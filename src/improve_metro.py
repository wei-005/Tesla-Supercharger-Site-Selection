"""Cheap, fast-to-verify improvements for the metro ranking (current AUC 0.757 vs demand-only 0.751).

All features use only information available on 1 Jan of year T; evaluation = same metro segment,
labels, AUC and precision@35 (15 km hit in T or T+1) as backtest.py, rolling 2019-2025.

Ideas tested (each adds a few BallTree radius counts; no new heavy computation):
  A  multi-scale demand      demand potential at several distance-decay exponents (already cached)
  B  supply as a signal      Tesla stalls within 5 / 15 / 30 km (infill: Tesla adds where it already is)
  C  build momentum          sites Tesla opened in the previous 2 years within 15 / 40 km
  D  competitor presence     AFDC effective ports within 15 km
  E  learned combination     logistic regression / gradient boosting on A-D + existing components,
                             retrained each year on years < T only (rolling, no look-ahead)
"""
import json, pickle, time
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from config import OUT, PROC, COMP_ATTR_MULT
from backtest import segments, seg_metrics, K_METRO, HIT_KM
from model import World, comp_ports

R = 6371.0
YEARS = list(range(2016, 2026))
EVAL = list(range(2019, 2026))


def radius_sum(tree_pts, weights, query, km):
    if len(tree_pts) == 0:
        return np.zeros(len(query))
    t = BallTree(np.radians(tree_pts), metric="haversine")
    idx = t.query_radius(np.radians(query), r=km / R)
    return np.array([weights[i].sum() for i in idx])


def build(world, cache):
    sc = world.sc
    frames = []
    for y in YEARS:
        base = cache[(1.5, 160.0, y)]
        m_metro, _ = segments(base)
        df = base.loc[m_metro, ["cell6", "lat", "lng", "local_net", "local_gross", "demand_pot",
                                "d_nearest_tesla", "flow_through", "anx_gain"]].copy()
        for b in (2.0, 2.5, 3.0):                                    # A: multi-scale demand
            df[f"demand_pot_b{b}"] = cache[(b, 160.0, y)].loc[m_metro, "demand_pot"].values
        q = df[["lat", "lng"]].values
        open_ = sc[sc.year_opened < y]
        st = np.where(open_.stalls > 0, open_.stalls, 12).astype(float)
        for km in (5, 15, 30):                                       # B: supply
            df[f"stalls_{km}km"] = radius_sum(open_[["lat", "lng"]].values, st, q, km)
        recent = sc[sc.year_opened.between(y - 2, y - 1)]
        for km in (15, 40):                                          # C: momentum
            df[f"recent_open_{km}km"] = radius_sum(recent[["lat", "lng"]].values, np.ones(len(recent)), q, km)
        cp = comp_ports(world, y); m = cp > 0                         # D: competitors
        df["comp_ports_15km"] = radius_sum(world.comp[m][["lat", "lng"]].values, cp[m] * COMP_ATTR_MULT, q, 15)
        df["year"] = y
        df["label"] = df.cell6.isin(set(sc[sc.year_opened == y].cell6)).astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


LOGF = lambda X: np.log1p(np.maximum(X, 0))
FEATS_ALL = ["demand_pot", "demand_pot_b2.0", "demand_pot_b2.5", "demand_pot_b3.0", "local_net", "local_gross",
             "stalls_5km", "stalls_15km", "stalls_30km", "recent_open_15km", "recent_open_40km",
             "comp_ports_15km", "d_nearest_tesla", "flow_through"]


def evaluate(world, D, cache, name, scorer):
    rows = []
    for y in EVAL:
        te = D[D.year == y]
        s = scorer(D, y, te)
        full = cache[(1.5, 160.0, y)]
        m_metro, _ = segments(full)
        sfull = np.full(len(full), -1e18); sfull[np.where(m_metro)[0]] = s
        m = seg_metrics(world, full, sfull, y, m_metro, K_METRO, HIT_KM["metro"])
        rows.append(dict(method=name, year=y, auc=m["auc"], precision=m["precision"]))
    return rows


def single(col):
    return lambda D, y, te: te[col].values.astype(float)


def product(cols):
    return lambda D, y, te: np.sum([np.log1p(te[c].values) for c in cols], axis=0)


def learned(kind, feats):
    def f(D, y, te):
        tr = D[D.year < y]
        rw = 0.75 ** (y - 1 - tr.year.values)                       # recency weights, as in backtest
        X, Xt = LOGF(tr[feats].values), LOGF(te[feats].values)
        if kind == "logit":
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))
            mdl.fit(X, tr.label.values, logisticregression__sample_weight=rw)
        else:
            mdl = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                 l2_regularization=1.0, class_weight="balanced")
            mdl.fit(X, tr.label.values, sample_weight=rw)
        return mdl.predict_proba(Xt)[:, 1]
    return f


if __name__ == "__main__":
    t0 = time.time()
    world = World()
    cache = pickle.load(open(PROC / "backtest_components.pkl", "rb"))
    D = build(world, cache)
    print(f"features built in {time.time() - t0:.0f}s, rows {len(D)}, positives {D.label.sum()}")
    rows = []
    rows += evaluate(world, D, cache, "demand only (beta 1.5)", single("demand_pot"))
    rows += evaluate(world, D, cache, "A best single scale (beta 2.5)", single("demand_pot_b2.5"))
    rows += evaluate(world, D, cache, "B demand x supply15 (log-sum)", product(["demand_pot", "stalls_15km"]))
    rows += evaluate(world, D, cache, "C demand x momentum40 (log-sum)", product(["demand_pot", "recent_open_40km"]))
    rows += evaluate(world, D, cache, "E1 logistic, demand scales only", learned("logit", ["demand_pot", "demand_pot_b2.0", "demand_pot_b2.5", "demand_pot_b3.0"]))
    rows += evaluate(world, D, cache, "E2 logistic, all cheap features", learned("logit", FEATS_ALL))
    rows += evaluate(world, D, cache, "E3 boosted trees, all cheap features", learned("gbm", FEATS_ALL))
    rows += evaluate(world, D, cache, "E4 boosted trees, without momentum", learned("gbm", [f for f in FEATS_ALL if not f.startswith("recent")]))
    R_ = pd.DataFrame(rows)
    bt = pd.read_csv(OUT / "backtest_by_year.csv")
    cur = bt[bt.split == "rolling_oos"][["year", "metro_auc", "metro_precision"]]
    R_ = pd.concat([R_, cur.rename(columns={"metro_auc": "auc", "metro_precision": "precision"}).assign(method="current method")])
    R_.to_csv(OUT / "improve_metro_by_year.csv", index=False)
    S = R_.groupby("method")[["auc", "precision"]].mean().sort_values("auc", ascending=False).round(3)
    base = R_[R_.method == "demand only (beta 1.5)"].set_index("year")
    wins = {m: int((g.set_index("year").auc > base.auc).sum()) for m, g in R_.groupby("method")}
    S["years_auc_beats_demand_only"] = S.index.map(wins)
    print(S.to_string())
    # permutation importance of the best learned model, refit on all years (for interpretation only)
    tr = D[D.year < 2026]
    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))
    mdl.fit(LOGF(tr[FEATS_ALL].values), tr.label.values)
    coef = pd.Series(mdl[-1].coef_[0], index=FEATS_ALL).sort_values(key=np.abs, ascending=False).round(3)
    print("logistic coefficients (standardised log features):\n", coef.to_string())
    json.dump(dict(summary=S.reset_index().to_dict(orient="records"), logit_coef=coef.to_dict()),
              open(OUT / "improve_metro_summary.json", "w"), indent=2, default=float)
    print(f"total {time.time() - t0:.0f}s")
