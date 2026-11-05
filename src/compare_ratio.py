"""Benchmark: the current method vs the classic heuristic score = demand / supply ** alpha (alpha = 0.5).

(1) Backtest: same candidates, years, segments, metrics and hit rules as backtest.py (2019-2025).
    demand  metro    = distance-decayed EV charging demand around the cell (demand_pot)
            corridor = gravity OD flow through the cell (flow_through)
    supply  = Tesla stalls open on 1 Jan T within 15 km (metro) / 25 km (corridor), +1 to avoid /0
            variant "+comp" also counts AFDC competitor effective ports x 0.55
(2) Decision: pick 10 sites for 2026 with the ratio under the same business rules, then value both
    portfolios with the same operating/NPV model (note: that yardstick is the current method's own model).
"""
import json, pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import config
from config import OUT, PROC
from backtest import segments, seg_metrics, pick_topk, random_precision, K_METRO, K_CORR, HIT_KM
from model import World, comp_ports

R_EARTH = 6371.0
RADIUS = {"metro": 15.0, "corridor": 25.0}


def supply_within(world, df, sids, year, radius_km, with_comp):
    sc = world.sc.iloc[sids]
    stalls = np.where(sc.stalls.values > 0, sc.stalls.values, 12).astype(float)
    pts = np.radians(df[["lat", "lng"]].values)
    tree = BallTree(np.radians(sc[["lat", "lng"]].values), metric="haversine")
    idx = tree.query_radius(pts, r=radius_km / R_EARTH)
    s = np.array([stalls[i].sum() for i in idx])
    if with_comp:
        cp = comp_ports(world, year)
        m = cp > 0
        ct = BallTree(np.radians(world.comp[m][["lat", "lng"]].values), metric="haversine")
        cidx = ct.query_radius(pts, r=radius_km / R_EARTH)
        w = cp[m] * config.COMP_ATTR_MULT
        s = s + np.array([w[i].sum() for i in cidx])
    return s


def ratio_score(world, df, year, alpha, with_comp, include_pipeline=False):
    if include_pipeline:
        sids = np.where(world.sc.status.isin(["OPEN", "EXPANDING", "CONSTRUCTION", "PERMIT", "PLAN", "CLOSED_TEMP"]).values)[0]
    else:
        sids = np.where((world.sc.year_opened.notna() & (world.sc.year_opened < year)).values)[0]
    m_metro, m_corr = segments(df)
    s = np.zeros(len(df))
    for seg, mask, dcol in (("metro", m_metro, "demand_pot"), ("corridor", m_corr, "flow_through")):
        sup = supply_within(world, df, sids, year, RADIUS[seg], with_comp)
        s = np.where(mask, df[dcol].values / (sup + 1.0) ** alpha, s)
    return s


def backtest_compare(world, cache):
    rows = []
    for y in range(2019, 2026):
        df = cache[(1.5, 160.0, y)]
        m_metro, m_corr = segments(df)
        for name, alpha, comp in (("D/S^0.5", 0.5, False), ("D/S^0.5 +comp", 0.5, True),
                                  ("D/S^0 (demand only)", 0.0, False), ("D/S^1", 1.0, False)):
            s = ratio_score(world, df, y, alpha, comp)
            for seg, mask, k in (("metro", m_metro, K_METRO), ("corridor", m_corr, K_CORR)):
                m = seg_metrics(world, df, s, y, mask, k, HIT_KM[seg])
                rows.append(dict(year=y, method=name, segment=seg, auc=m["auc"], precision=m["precision"]))
    R = pd.DataFrame(rows)
    bt = pd.read_csv(OUT / "backtest_by_year.csv")
    cur = bt[bt.split == "rolling_oos"]
    for seg in ("metro", "corridor"):
        for _, r in cur.iterrows():
            R.loc[len(R)] = dict(year=r.year, method="current (rolling recalibrated)", segment=seg,
                                 auc=r[f"{seg}_auc"], precision=r[f"{seg}_precision"])
            R.loc[len(R)] = dict(year=r.year, method="random", segment=seg, auc=0.5,
                                 precision=r[f"{seg}_precision_random"])
    R.to_csv(OUT / "compare_ratio_backtest_by_year.csv", index=False)
    S = R.groupby(["segment", "method"])[["auc", "precision"]].mean().round(3)
    # year-by-year wins of current vs D/S^0.5
    piv = R[R.method.isin(["current (rolling recalibrated)", "D/S^0.5"])].pivot_table(
        index=["segment", "year"], columns="method", values=["auc", "precision"])
    wins = {}
    for seg in ("metro", "corridor"):
        p = piv.loc[seg]
        wins[seg] = dict(
            auc_years_current_better=int((p[("auc", "current (rolling recalibrated)")] > p[("auc", "D/S^0.5")]).sum()),
            precision_years_current_better=int((p[("precision", "current (rolling recalibrated)")] > p[("precision", "D/S^0.5")]).sum()),
            precision_years_tied=int((p[("precision", "current (rolling recalibrated)")] == p[("precision", "D/S^0.5")]).sum()))
    return S, wins


def decision_compare(world):
    import select_sites as ss
    st = pickle.load(open(PROC / "decision_state.pkl", "rb"))
    df = st["df"]
    ss.TRIP_SCALE = world.counties.ev_2023.sum() * config.CORRIDOR_TRIPS_PER_EV_YEAR / 1e6
    s = ratio_score(world, df, 2027, 0.5, False, include_pipeline=True)
    m_metro, m_corr = segments(df)
    elig = (df.d_nearest_tesla.values >= config.MIN_DIST_EXISTING_KM) & (m_metro | m_corr) & ~df.cell6.isin(config.EXCLUDE_CELLS).values
    states = ss.nearest_states(world, df.lat.values, df.lng.values)
    # same mix as the current pick (9 metro + 1 corridor), 40 km spacing, <= 3 per state
    picks = []
    for seg_mask, n in ((m_metro, 9), (m_corr, 1)):
        order = pick_topk(df.lat.values, df.lng.values, s, 400, nms_km=config.MIN_DIST_BETWEEN_PICKS_KM, mask=elig & seg_mask)
        cnt = {}
        for i in picks:
            cnt[states[i]] = cnt.get(states[i], 0) + 1
        got = 0
        for i in order:
            if got == n:
                break
            la, ln = np.radians(df.lat.values), np.radians(df.lng.values)
            if picks:
                c = np.array(picks)
                d = 2 * R_EARTH * np.arcsin(np.sqrt(np.sin((la[c] - la[i]) / 2) ** 2 + np.cos(la[c]) * np.cos(la[i]) * np.sin((ln[c] - ln[i]) / 2) ** 2))
                if (d < config.MIN_DIST_BETWEEN_PICKS_KM).any():
                    continue
            if cnt.get(states[i], 0) >= config.MAX_SITES_PER_STATE:
                continue
            picks.append(i); cnt[states[i]] = cnt.get(states[i], 0) + 1; got += 1
    q90 = np.quantile(df.demand_pot.values[m_metro], 0.9)
    rows = []
    for i in picks:
        seg = "metro" if m_metro[i] else "corridor"
        area = "rural" if seg == "corridor" else ("urban" if df.demand_pot.values[i] >= q90 else "suburban")
        f_loc, comp_share = ss.site_local_fn(df, i, world)
        site = dict(area=area, state=states[i], corr_kwh12=df.corr_gross.values[i] * ss.TRIP_SCALE, local_kwh_by_n=f_loc)
        best = None
        for n in config.STALL_OPTIONS:
            for opt in config.INTERCONNECT_OPTIONS:
                for mp in (0, 1, 2):
                    r = ss.npv_of(site, n, opt, mp, 8 if seg == "metro" else 4)
                    if r and (best is None or r["npv"] > best["npv"]):
                        best = dict(r, stalls=n, interconnect=opt, megapacks=mp)
        rows.append(dict(cell6=df.cell6.values[i], segment=seg, state=states[i], lat=df.lat.values[i], lng=df.lng.values[i],
                         d_nearest_tesla_km=round(df.d_nearest_tesla.values[i], 1), ratio_score=s[i],
                         stalls=best["stalls"], interconnect=best["interconnect"], npv_musd=best["npv"] / 1e6,
                         corridor_gwh=best["corridor_kwh_y1"] / 1e6, utilization=best["utilization"],
                         competitor_share=comp_share))
    P = pd.DataFrame(rows)
    cur = pd.read_csv(OUT / "selected_10_cells.csv")
    la1, ln1 = np.radians(P.lat.values), np.radians(P.lng.values)
    la2, ln2 = np.radians(cur.lat.values), np.radians(cur.lng.values)
    D = 2 * R_EARTH * np.arcsin(np.sqrt(np.sin((la1[:, None] - la2[None]) / 2) ** 2 + np.cos(la1[:, None]) * np.cos(la2[None]) * np.sin((ln1[:, None] - ln2[None]) / 2) ** 2))
    P["km_to_nearest_current_pick"] = D.min(axis=1).round(1)
    P.to_csv(OUT / "compare_ratio_portfolio.csv", index=False)
    summ = dict(
        ratio_npv_musd=round(P.npv_musd.sum(), 1), current_npv_musd=round(cur.npv.sum() / 1e6, 1),
        ratio_corridor_gwh=round(P.corridor_gwh.sum(), 2), current_corridor_gwh=round(cur.corridor_kwh_y1.sum() / 1e6, 2),
        overlap_within_15km=int((P.km_to_nearest_current_pick <= 15).sum()),
        ratio_negative_npv_sites=int((P.npv_musd <= 0).sum()),
        ratio_mean_competitor_share=round(P.competitor_share.mean(), 2),
        current_mean_competitor_share=round(cur.competitor_share_catchment.mean(), 2))
    return P, summ


if __name__ == "__main__":
    world = World()
    cache = pickle.load(open(PROC / "backtest_components.pkl", "rb"))
    S, wins = backtest_compare(world, cache)
    del cache
    print(S.to_string()); print(wins)
    P, summ = decision_compare(world)
    pd.set_option("display.width", 220)
    print(P[["segment", "state", "lat", "lng", "d_nearest_tesla_km", "stalls", "npv_musd", "utilization",
             "competitor_share", "km_to_nearest_current_pick"]].round(2).to_string())
    print(summ)
    json.dump(dict(backtest=S.reset_index().to_dict(orient="records"), wins=wins, decision=summ),
              open(OUT / "compare_ratio_summary.json", "w"), indent=2, default=float)
