"""Step 4: rolling-origin backtest on Tesla's real 2016-2025 openings (two-segment model).

For each year T the model only sees the network open on 1 Jan T (supercharge.info open dates) and the
nearest earlier AFDC competitor snapshot. It scores every candidate H3 cell; we then check where Tesla
actually opened sites in T (and T+1 for the top-K hit test).

Tesla builds two different products, so the score is segmented:
  metro    - candidates in the densest 40% of local demand potential: MCI-Huff components
  corridor - other candidates crossed by OD paths: flow-refuelling components
Calibration (beta, comfort range, weights per segment) uses 2016-2021 only; 2022-2025 is held out.
"""
import itertools, json, pickle, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import BallTree
from config import *
from model import World, Params, evaluate_year, norm

METRO_KEYS = ["local_gross", "local", "pot", "dnear"]
CORR_KEYS = ["corr_gross", "anx", "corr", "flow", "dnear"]
COLS = {"local": "local_net", "local_gross": "local_gross", "corr": "corr_net", "corr_gross": "corr_gross",
        "feas": "feas_gain", "anx": "anx_gain", "flow": "flow_through", "pot": "demand_pot", "dnear": "dnear_cap"}
RECENCY = 0.75   # weight of a calibration year = RECENCY ** (age in years)
K_METRO, K_CORR = 35, 15
HIT_KM = {"metro": 15, "corridor": 25}


def segments(df):
    metro = df.demand_pot.values >= np.quantile(df.demand_pot.values, 0.6)
    corr = (~metro) & (df.flow_through.values > 0)
    return metro, corr


def seg_score(df, w):
    if "dnear_cap" not in df:
        df["dnear_cap"] = np.minimum(df.d_nearest_tesla.values, 150.0)
    s = np.zeros(len(df))
    for k, v in w.items():
        if v:
            s += v * norm(df[COLS[k]].values)
    return s


def pick_topk(lat, lng, s, k, nms_km=NMS_KM, mask=None):
    order = np.argsort(-s)
    if mask is not None:
        order = order[mask[order]]
    la, ln = np.radians(lat), np.radians(lng)
    chosen = []
    for i in order:
        if len(chosen) == k:
            break
        if chosen:
            c = np.array(chosen)
            d = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la[c] - la[i]) / 2) ** 2 +
                                             np.cos(la[c]) * np.cos(la[i]) * np.sin((ln[c] - ln[i]) / 2) ** 2))
            if (d < nms_km).any():
                continue
        chosen.append(i)
    return np.array(chosen, dtype=int)


def seg_metrics(world, df, s, year, mask, k, hit_km):
    sc = world.sc
    opened_y = sc[sc.year_opened == year]
    win = sc[sc.year_opened.between(year, year + 1)]
    y = df.cell6.isin(set(opened_y.cell6)).values
    auc = roc_auc_score(y[mask], s[mask]) if 0 < y[mask].sum() < mask.sum() else np.nan
    picks = pick_topk(df.lat.values, df.lng.values, s, k, mask=mask)
    tree = BallTree(np.radians(win[["lat", "lng"]].values), metric="haversine")
    d = tree.query(np.radians(df.iloc[picks][["lat", "lng"]].values), k=1)[0][:, 0] * 6371
    return dict(auc=auc, precision=float((d <= hit_km).mean()), n_pos=int(y[mask].sum()))


def random_precision(world, df, year, mask, k, hit_km, n=30):
    rng = np.random.default_rng(0)
    return float(np.mean([seg_metrics(world, df, rng.random(len(df)), year, mask, k, hit_km)["precision"]
                          for _ in range(n)]))


def simplex(keys, step=0.25):
    n = int(round(1 / step))
    def rec(i, left):
        if i == len(keys) - 1:
            yield [left]; return
        for v in range(left + 1):
            for rest in rec(i + 1, left - v):
                yield [v] + rest
    for comb in rec(0, n):
        yield {k: c * step for k, c in zip(keys, comb)}


def components(world, betas, comforts, years, reuse=True):
    f = PROC / "backtest_components.pkl"
    cache = pickle.load(open(f, "rb")) if (reuse and f.exists()) else {}
    for beta, rc in itertools.product(betas, comforts):
        for y in years:
            if (beta, rc, y) not in cache:
                cache[(beta, rc, y)] = evaluate_year(world, y, Params(beta=beta, r_comfort=rc))
    pickle.dump(cache, open(f, "wb"))
    return cache


def calibrate(world, cache, betas, comforts, years=None, save=True):
    years = years or CALIB_YEARS
    rw = np.array([RECENCY ** (max(years) - y) for y in years]); rw = rw / rw.sum()
    res = []
    for beta, rc in itertools.product(betas, comforts):
        for seg, keys, k in (("metro", METRO_KEYS, K_METRO), ("corridor", CORR_KEYS, K_CORR)):
            if seg == "metro" and rc != comforts[0]:
                continue          # comfort range does not enter the metro score
            if seg == "corridor" and beta != betas[0]:
                continue          # distance decay does not enter the corridor score
            for w in simplex(keys):
                ms = []
                for y in years:
                    df = cache[(beta, rc, y)]
                    m_metro, m_corr = segments(df)
                    mask = m_metro if seg == "metro" else m_corr
                    ms.append(seg_metrics(world, df, seg_score(df, w), y, mask, k, HIT_KM[seg]))
                a = np.nansum(rw * np.array([m["auc"] for m in ms])); p = float(np.sum(rw * np.array([m["precision"] for m in ms])))
                res.append(dict(segment=seg, beta=beta, r_comfort=rc, weights=json.dumps(w),
                                calib_auc=a, calib_precision=p, J=a + p))
    grid = pd.DataFrame(res)
    if save:
        grid.sort_values(["segment", "J"], ascending=[True, False]).to_csv(OUT / "backtest_calibration_grid.csv", index=False)
    # beta only affects metro, comfort range only affects corridor -> pick each segment's best jointly
    bm = grid[grid.segment == "metro"].sort_values("J", ascending=False).iloc[0]
    bc = grid[grid.segment == "corridor"].sort_values("J", ascending=False).iloc[0]
    return dict(beta=float(bm.beta), r_comfort=float(bc.r_comfort),
                w_metro=json.loads(bm.weights), w_corridor=json.loads(bc.weights))


def evaluate(world, cache, calib, rolling=None):
    rows = []
    for y in BACKTEST_YEARS:
        if rolling is not None:
            if y not in rolling:
                continue
            calib = rolling[y]
        df = cache[(calib["beta"], calib["r_comfort"], y)]
        m_metro, m_corr = segments(df)
        r = dict(year=y, split=("rolling_oos" if rolling is not None else
                                ("calibration" if y in CALIB_YEARS else "holdout")), n_existing=df.attrs["n_sites"],
                 n_opened=int((world.sc.year_opened == y).sum()))
        for seg, mask, k, w in (("metro", m_metro, K_METRO, calib["w_metro"]),
                                ("corridor", m_corr, K_CORR, calib["w_corridor"])):
            hk = HIT_KM[seg]
            m = seg_metrics(world, df, seg_score(df, w), y, mask, k, hk)
            pop = seg_metrics(world, df, df.demand_pot.values, y, mask, k, hk)
            gap = norm(df.d_nearest_tesla.values) * (norm(df.flow_through.values) + norm(df.demand_pot.values))
            g = seg_metrics(world, df, gap, y, mask, k, hk)
            r.update({f"{seg}_n_pos": m["n_pos"], f"{seg}_auc": m["auc"], f"{seg}_precision": m["precision"],
                      f"{seg}_auc_pop": pop["auc"], f"{seg}_precision_pop": pop["precision"],
                      f"{seg}_auc_gap": g["auc"], f"{seg}_precision_gap": g["precision"],
                      f"{seg}_precision_random": random_precision(world, df, y, mask, k, hk)})
        rows.append(r)
    bt = pd.DataFrame(rows)
    cols = [c for c in bt.columns if c.endswith(("auc", "precision", "_pop", "_gap", "_random"))]
    summary = bt.groupby("split")[cols].mean()
    return bt, summary


def run(reuse=True):
    t0 = time.time()
    world = World()
    betas, comforts = [1.5, 2.0, 2.5, 3.0], [160, 200, 240]
    cache = components(world, betas, comforts, BACKTEST_YEARS, reuse)
    print(f"components ready in {time.time() - t0:.0f}s")
    # (a) fixed split: calibrate 2016-2021, hold out 2022-2025
    calib = calibrate(world, cache, betas, comforts)
    print("fixed-split calibration:", calib)
    bt_fixed, sum_fixed = evaluate(world, cache, calib)
    # (b) rolling origin: for each year T >= 2019 refit on 2016..T-1 (recency weighted), predict T
    rolling = {}
    for y in range(2019, max(BACKTEST_YEARS) + 1):
        rolling[y] = calibrate(world, cache, betas, comforts, years=list(range(2016, y)), save=False)
        print(y, rolling[y])
    bt_roll, sum_roll = evaluate(world, cache, None, rolling)
    for y, c in rolling.items():
        bt_roll.loc[bt_roll.year == y, "weights_used"] = json.dumps(dict(beta=c["beta"], r_comfort=c["r_comfort"],
                                                                         metro=c["w_metro"], corridor=c["w_corridor"]))
    bt = pd.concat([bt_fixed, bt_roll], ignore_index=True)
    bt.to_csv(OUT / "backtest_by_year.csv", index=False)
    summary = pd.concat([sum_fixed, sum_roll])
    summary.to_csv(OUT / "backtest_summary.csv")
    pd.set_option("display.width", 250)
    print(bt.drop(columns=["weights_used"], errors="ignore").round(3).to_string())
    print(summary.round(3).T.to_string())
    # (c) final weights for the 2026 decision: refit on all years 2016-2025, recency weighted
    final = calibrate(world, cache, betas, comforts, years=BACKTEST_YEARS, save=True)
    final["fixed_split"] = calib
    final["summary"] = summary.round(4).to_dict(orient="index")
    json.dump(final, open(OUT / "calibrated_params.json", "w"), indent=2, default=float)
    print("final (2026) weights:", {k: final[k] for k in ["beta", "r_comfort", "w_metro", "w_corridor"]})
    return final


if __name__ == "__main__":
    run()
