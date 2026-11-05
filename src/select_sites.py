"""Step 5: the 2026 decision (method C).

1. Screen    - score every candidate with the backtest-calibrated segment weights on today's network
               (open + construction + permit + plan), drop cells within MIN_DIST_EXISTING_KM of any
               Tesla site, keep a shortlist per segment (metro / corridor).
2. Columns   - for every shortlisted cell enumerate configurations (stalls x interconnection option x
               Megapack units) and run an hourly operating model on representative days against a
               flexible-interconnection envelope. Each configuration becomes a column with NPV, time to
               energise, curtailed kWh, service level.
3. Master    - MILP over (site, configuration) columns: maximise NPV subject to choosing 10 sites,
               40 km spacing, a pairwise cannibalisation correction and an epsilon-constraint on
               corridor comfort (objective 2). The epsilon sweep gives the Pareto frontier; we take the
               largest comfort gain that costs <= 5% of the unconstrained NPV.
4. Forward check - rank on the *open-only* network and test how often the shortlist lands on sites that
               Tesla itself has under construction / permit / plan today (data the model never sees).
"""
import json, pickle
import numpy as np
import pandas as pd
import h3
from scipy.optimize import milp, LinearConstraint, Bounds
from sklearn.neighbors import BallTree
from config import *
from model import World, Params, evaluate_year, norm, size_attr
from backtest import segments, seg_score, pick_topk

DECISION_YEAR = 2027          # sites chosen now energise 2027-2028
CAL = json.load(open(OUT / "calibrated_params.json"))
PRM = Params(beta=CAL["beta"], r_comfort=CAL["r_comfort"])
TRIP_SCALE = None             # set in main: real trips per normalised flow unit

# ------------------------------------------------------------------ representative days
DAY_TYPES = {  # name: (days per year, season)
    "summer_weekday": (65, "summer"), "summer_weekend_holiday": (27, "summer"),
    "shoulder_weekday": (85, "shoulder"), "shoulder_weekend": (32, "shoulder"),
    "winter_weekday": (110, "winter"), "winter_weekend": (46, "winter")}
H = np.arange(24)


def _bump(c, w):
    return np.exp(-0.5 * ((H - c) / w) ** 2)


METRO_SHAPE = 0.15 + 0.9 * _bump(12.5, 2.2) + 1.0 * _bump(18.5, 2.0)
CORR_SHAPE = 0.08 + 1.0 * _bump(13.5, 3.5)
DEMAND_MULT = {"metro": {"weekday": 1.0, "weekend": 1.1, "weekend_holiday": 1.1},
               "corridor": {"weekday": 0.8, "weekend": 1.25, "weekend_holiday": 1.7}}
WINTER_KWH_MULT = 1.15        # more energy per session in cold weather
# feeder load (share of feeder peak) by day type: residential/commercial mix
FEEDER = {
    "summer_weekday": 0.45 + 0.55 * _bump(18.5, 2.5),
    "summer_weekend_holiday": 0.40 + 0.45 * _bump(18, 3.0),
    "shoulder_weekday": 0.40 + 0.30 * _bump(8, 1.5) + 0.35 * _bump(19, 2.0),
    "shoulder_weekend": 0.35 + 0.30 * _bump(18, 3.0),
    "winter_weekday": 0.45 + 0.40 * _bump(7.5, 1.5) + 0.45 * _bump(18.5, 2.0),
    "winter_weekend": 0.40 + 0.35 * _bump(18, 3.0)}
FEEDER = {k: v / max(f.max() for f in FEEDER.values()) for k, v in FEEDER.items()}
KWH_PER_STALL_HOUR = 80       # service capacity of one stall incl. taper and turnover


def day_profiles(annual_metro_kwh, annual_corr_kwh):
    """kWh per hour for each day type, scaled so the annual sum equals the inputs."""
    prof = {}
    for d, (n, season) in DAY_TYPES.items():
        kind = "weekend_holiday" if "holiday" in d else ("weekend" if "weekend" in d else "weekday")
        wm = DEMAND_MULT["metro"][kind] * (WINTER_KWH_MULT if season == "winter" else 1.0)
        wc = DEMAND_MULT["corridor"][kind] * (0.75 if season == "winter" else (1.15 if season == "summer" else 1.0))
        prof[d] = (wm * METRO_SHAPE / METRO_SHAPE.sum(), wc * CORR_SHAPE / CORR_SHAPE.sum())
    zm = sum(DAY_TYPES[d][0] * prof[d][0].sum() for d in prof)
    zc = sum(DAY_TYPES[d][0] * prof[d][1].sum() for d in prof)
    return {d: annual_metro_kwh / zm * pm + annual_corr_kwh / zc * pc for d, (pm, pc) in prof.items()}


def envelope(option, area, request_kw, state):
    """hourly grid-capacity envelope (kW) per day type, months to energise, extra capex, availability."""
    head = AREA_HEADROOM_KW[area]
    h = {d: head * (1 - 0.6 * FEEDER[d]) for d in DAY_TYPES}
    h_min = min(v.min() for v in h.values())
    flex_ok = state in STATES_WITH_FLEX_PROGRAMS
    deficit = max(0.0, request_kw - h_min)
    if option == "firm":
        lead = 9 if deficit == 0 else FIRM_LEADTIME_MONTHS[area]
        return {d: np.full(24, request_kw) for d in DAY_TYPES}, lead, deficit * FIRM_UPGRADE_COST_PER_KW, True
    if option == "static_flex":
        cap = min(request_kw, h_min)
        return {d: np.full(24, cap) for d in DAY_TYPES}, FLEX_LEADTIME_MONTHS, 20_000, True
    if option == "dynamic_flex":
        env = {d: np.clip(h[d], FLEX_FLOOR_SHARE * request_kw, request_kw) for d in DAY_TYPES}
        return env, FLEX_LEADTIME_MONTHS, 60_000, flex_ok
    if option == "bridge":   # dynamic now, firm once the upgrade lands
        env = {d: np.clip(h[d], FLEX_FLOOR_SHARE * request_kw, request_kw) for d in DAY_TYPES}
        lead_firm = 9 if deficit == 0 else FIRM_LEADTIME_MONTHS[area]
        return env, (FLEX_LEADTIME_MONTHS, lead_firm), 60_000 + deficit * FIRM_UPGRADE_COST_PER_KW, flex_ok and deficit > 0
    raise ValueError(option)


def operate(dem, env, n_stalls, mp_units):
    """greedy hourly dispatch with Megapack: serve demand up to stall capacity, grid draw <= envelope,
    battery covers the deficit and refills from spare headroom. Returns per-day served/lost/peak kW."""
    e_cap, p_cap = mp_units * MEGAPACK_UNIT_KWH, mp_units * MEGAPACK_UNIT_KW
    out = {}
    for d in DAY_TYPES:
        want = np.minimum(dem[d], n_stalls * KWH_PER_STALL_HOUR)
        soc = 0.6 * e_cap
        served = np.zeros(24); grid = np.zeros(24)
        for _ in range(2):             # two passes so the start-of-day SOC is periodic
            for t in range(24):
                g = min(want[t], env[d][t])
                deficit = want[t] - g
                dis = min(deficit, p_cap, soc * 0.95)
                soc -= dis / 0.95
                spare = env[d][t] - g
                ch = min(spare, p_cap, (e_cap - soc) / 0.95) if e_cap else 0.0
                soc += ch * 0.95
                served[t] = g + dis; grid[t] = g + ch
        out[d] = dict(served=served.sum(), lost=dem[d].sum() - served.sum(),
                      curtailed=(want - np.minimum(want, env[d])).sum(), peak_kw=grid.max(), grid=grid.sum())
    return out


def annual(op):
    res = dict(served=0.0, lost=0.0, curtailed=0.0, grid=0.0)
    for d, (n, season) in DAY_TYPES.items():
        for k in res:
            res[k] += n * op[d][k]
    peaks = {}
    for d, (n, season) in DAY_TYPES.items():
        peaks[season] = max(peaks.get(season, 0), op[d]["peak_kw"])
    res["demand_charge"] = DEMAND_CHARGE_PER_KW_MONTH * (3 * peaks["summer"] + 4 * peaks["shoulder"] + 5 * peaks["winter"])
    return res


def npv_of(site, n_stalls, option, mp_units, n_l2):
    req = n_stalls * STALL_KW_DESIGN * COINCIDENCE
    env, lead, extra, ok = envelope(option, site["area"], req, site["state"])
    if not ok:
        return None
    mult = float(size_attr(n_stalls, PRM.gamma) / size_attr(12, PRM.gamma))
    loc = site["local_kwh_by_n"](n_stalls)
    cor = site["corr_kwh12"] * mult / (1 + 0.25 * (mult - 1))
    dem = day_profiles(loc, cor)
    if option == "bridge":
        env_firm, _, _, _ = envelope("firm", site["area"], req, site["state"])
        ops = {"early": annual(operate(dem, env, n_stalls, mp_units)), "late": annual(operate(dem, env_firm, n_stalls, mp_units))}
        lead_on, lead_firm = lead
    else:
        ops = {"early": annual(operate(dem, env, n_stalls, mp_units))}
        ops["late"] = ops["early"]; lead_on, lead_firm = lead, lead
    capex = (SITE_FIXED_CAPEX + n_stalls * CAPEX_PER_STALL + mp_units * MEGAPACK_UNIT_COST + n_l2 * L2_UNIT_COST + extra)
    price, ecost = REGION_ECON["CA"] if site["state"] == "CA" else (
        REGION_ECON["NE"] if site["state"] in NE_STATES else REGION_ECON["default"])
    margin = price - ecost
    npv = -capex; served_total = 0.0
    for t in range(LIFE_YEARS):
        growth = min(1.07 ** t, 1.45)
        m_on = np.clip(12 * (t + 1) - lead_on, 0, 12) / 12
        m_firm = np.clip(12 * (t + 1) - lead_firm, 0, 12) / 12 if option == "bridge" else m_on
        for key, frac in (("early", m_on - m_firm), ("late", m_firm)):
            if frac <= 0:
                continue
            a = ops[key]
            cf = frac * (growth * (a["served"] * margin - a["lost"] * (margin + LOST_SESSION_PENALTY))
                         - a["demand_charge"] - n_stalls * OTHER_OPEX_PER_STALL_YEAR)
            npv += cf / (1 + DISCOUNT_RATE) ** (t + 0.5)
            served_total += frac * growth * a["served"]
        npv += n_l2 * 900 / (1 + DISCOUNT_RATE) ** (t + 0.5) * (m_on > 0)   # L2 destination margin
    a = ops["late"]
    util = a["served"] / (n_stalls * KWH_PER_STALL_HOUR * 24 * 365)
    return dict(npv=npv, capex=capex, months_to_energise=lead_on, months_to_full_power=lead_firm,
                price_per_kwh=price, energy_cost_per_kwh=ecost, corridor_kwh_y1=cor, local_kwh_y1=loc,
                served_kwh_y1=ops["early"]["served"], lost_kwh_y1=ops["early"]["lost"],
                curtailed_kwh_y1=ops["early"]["curtailed"], utilization=util,
                service_level=a["served"] / max(1.0, a["served"] + a["lost"]),
                request_kw=req, demand_charge=a["demand_charge"])


def site_local_fn(df, k_rows, world):
    """local gross kWh/yr for a site at candidate index k with n stalls (Huff on today's network)."""
    Q, denom = df.attrs["Q"], df.attrs["denom"]
    m = world.ck_k == k_rows
    c, d = world.ck_c[m], world.ck_d[m]
    dec = (d + 3.0) ** (-PRM.beta)
    def f(n):
        a = float(size_attr(n, PRM.gamma)) * dec
        return float((Q[c] * a / (denom[c] + a)).sum())
    S_C, S_T = df.attrs["S_C"], df.attrs["S_T"]
    w = Q[c] * dec
    comp_share = float((w * S_C[c] / (S_C[c] + S_T[c] + 1e-12)).sum() / max(w.sum(), 1e-12))
    return f, comp_share


def area_type(row, q90):
    if row.segment == "corridor":
        return "rural"
    return "urban" if row.demand_pot >= q90 else "suburban"


def nearest_states(world, lats, lngs):
    tree = BallTree(np.radians(world.counties[["lat", "lng"]].values), metric="haversine")
    i = tree.query(np.radians(np.c_[lats, lngs]), k=1)[1][:, 0]
    return world.counties.state.values[i]


def nearest_state(world, lat, lng):
    tree = BallTree(np.radians(world.counties[["lat", "lng"]].values), metric="haversine")
    i = tree.query(np.radians([[lat, lng]]), k=1)[1][0][0]
    r = world.counties.iloc[i]
    return r.state, r.County


def main():
    global TRIP_SCALE
    world = World()
    TRIP_SCALE = world.counties.ev_2023.sum() * CORRIDOR_TRIPS_PER_EV_YEAR / 1e6
    df = evaluate_year(world, DECISION_YEAR, PRM, include_pipeline=True, comp_year=2027)
    df["dnear_cap"] = np.minimum(df.d_nearest_tesla.values, 150)
    m_metro, m_corr = segments(df)
    df["segment"] = np.where(m_metro, "metro", np.where(m_corr, "corridor", "other"))
    df["screen_score"] = np.where(m_metro, seg_score(df, CAL["w_metro"]), seg_score(df, CAL["w_corridor"]))
    if METRO_SCREEN == "learned":
        import learned_screen
        lmodel = learned_screen.fit(world)
        df["screen_score"] = np.where(m_metro, learned_screen.score(world, df, m_metro, DECISION_YEAR, True, PRM.beta, lmodel),
                                      df.screen_score.values)
    df.loc[df.segment == "other", "screen_score"] = 0

    # ---- forward validation on the open-only network (pipeline sites are the "future" truth)
    fwd = evaluate_year(world, DECISION_YEAR, PRM, include_pipeline=False, comp_year=2027)
    fwd["dnear_cap"] = np.minimum(fwd.d_nearest_tesla.values, 150)
    fm, fc = segments(fwd)
    pipe = world.sc[world.sc.status.isin(["CONSTRUCTION", "PERMIT", "PLAN"])]
    tree = BallTree(np.radians(pipe[["lat", "lng"]].values), metric="haversine")
    fwd_rows = []
    for seg, mask, w, k in (("metro", fm, CAL["w_metro"], 50), ("corridor", fc, CAL["w_corridor"], 30)):
        s = seg_score(fwd, w)
        rankings = [("model" if not (seg == "metro" and METRO_SCREEN == "learned") else "demand_pot (v1)", s),
                    ("random", np.random.default_rng(1).random(len(fwd)))]
        if seg == "metro" and METRO_SCREEN == "learned":
            rankings.insert(0, ("learned logit (v2)", learned_screen.score(world, fwd, mask, DECISION_YEAR, False, PRM.beta, lmodel)))
        for name, sc_ in rankings:
            pk = pick_topk(fwd.lat.values, fwd.lng.values, sc_, k, mask=mask & (fwd.d_nearest_tesla.values > 3))
            d = tree.query(np.radians(fwd.iloc[pk][["lat", "lng"]].values), k=1)[0][:, 0] * 6371
            fwd_rows.append(dict(segment=seg, ranking=name, k=k, share_within_15km_of_tesla_pipeline=float((d <= 15).mean())))
    fwd_df = pd.DataFrame(fwd_rows); fwd_df.to_csv(OUT / "forward_validation_pipeline.csv", index=False)
    print(fwd_df)

    # ---- screening shortlist on today's network incl. pipeline
    elig = (df.d_nearest_tesla.values >= MIN_DIST_EXISTING_KM) & (df.segment != "other")
    q90 = np.quantile(df.demand_pot.values[m_metro], 0.9)
    states_all = nearest_states(world, df.lat.values, df.lng.values)
    short = []
    for seg, k in (("metro", int(SHORTLIST * 0.6)), ("corridor", int(SHORTLIST * 0.4))):
        pk = pick_topk(df.lat.values, df.lng.values, df.screen_score.values, 4 * k, nms_km=20,
                       mask=elig & (df.segment.values == seg))
        cnt, keep = {}, []
        for i in pk:
            stt = states_all[i]
            if cnt.get(stt, 0) < SHORTLIST_STATE_CAP[seg]:
                keep.append(i); cnt[stt] = cnt.get(stt, 0) + 1
            if len(keep) == k:
                break
        short.extend(keep)
    S = df.iloc[short].copy().reset_index().rename(columns={"index": "k"})
    S = S[~S.cell6.isin(EXCLUDE_CELLS)].reset_index(drop=True)
    S["area"] = [area_type(r, q90) for r in S.itertuples()]
    st = [nearest_state(world, a, b) for a, b in zip(S.lat, S.lng)]
    S["state"] = [x[0] for x in st]; S["county"] = [x[1] for x in st]
    S["corr_kwh12"] = S.corr_gross * TRIP_SCALE

    # ---- column generation: configurations per site
    cols = []
    for r in S.itertuples():
        f_loc, comp_share = site_local_fn(df, r.k, world)
        site = dict(area=r.area, state=r.state, corr_kwh12=r.corr_kwh12, local_kwh_by_n=f_loc)
        S.loc[S.k == r.k, "competitor_share_catchment"] = comp_share
        S.loc[S.k == r.k, "local_kwh_12"] = f_loc(12)
        for n in STALL_OPTIONS:
            for opt in INTERCONNECT_OPTIONS:
                for mp in (0, 1, 2):
                    n_l2 = 8 if r.segment == "metro" else 4
                    res = npv_of(site, n, opt, mp, n_l2)
                    if res is None:
                        continue
                    cols.append(dict(k=r.k, cell6=r.cell6, stalls=n, interconnect=opt, megapacks=mp, l2=n_l2, **res))
    C = pd.DataFrame(cols)
    C = C[C.npv > 0]
    # keep the non-dominated configurations per site (NPV up, months down, service up)
    keep = []
    for k, g in C.groupby("k"):
        g = g.sort_values("npv", ascending=False)
        best_by = g.groupby("months_to_energise").head(1)
        keep.append(pd.concat([g.head(3), best_by]).drop_duplicates())
    C = pd.concat(keep, ignore_index=True)
    C.to_csv(OUT / "config_columns.csv", index=False)
    S.to_csv(OUT / "shortlist.csv", index=False)
    print("shortlist", len(S), "columns", len(C))

    # ---- master MILP with epsilon-constraint on corridor comfort
    sites = S.k.values
    pos = {k: i for i, k in enumerate(sites)}
    nC, nS = len(C), len(S)
    corridor = C.corridor_kwh_y1.values / 1e6                         # objective 1: corridor flow served (GWh/yr)
    comfort = S.set_index("k").anx_gain.reindex(C.k).values          # objective 2: comfort-gap reduction
    feas = S.set_index("k").feas_gain.reindex(C.k).values             # objective 1b: newly drivable OD flow
    lat, lng = np.radians(S.lat.values), np.radians(S.lng.values)
    D = 2 * 6371 * np.arcsin(np.sqrt(np.sin((lat[:, None] - lat[None]) / 2) ** 2 +
                                     np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin((lng[:, None] - lng[None]) / 2) ** 2))
    A_site = np.zeros((nS, nC))
    for j, k in enumerate(C.k.values):
        A_site[pos[k], j] = 1
    rows, lbs, ubs = [A_site], [np.zeros(nS)], [np.ones(nS)]
    rows.append(np.ones((1, nC))); lbs.append([N_SITES]); ubs.append([N_SITES])
    close = [(a, b) for a in range(nS) for b in range(a + 1, nS) if D[a, b] < MIN_DIST_BETWEEN_PICKS_KM]
    if close:
        R = np.zeros((len(close), nC))
        for i, (a, b) in enumerate(close):
            R[i] = A_site[a] + A_site[b]
        rows.append(R); lbs.append(np.zeros(len(close))); ubs.append(np.ones(len(close)))
    stv = S.state.values
    for stt in np.unique(stv):
        R = A_site[stv == stt].sum(axis=0, keepdims=True)
        rows.append(R); lbs.append([0]); ubs.append([MAX_SITES_PER_STATE])

    def solve(eps=None):
        Aeq = list(rows); lb = list(lbs); ub = list(ubs)
        if eps is not None:
            Aeq.append(corridor[None, :]); lb.append([eps]); ub.append([np.inf])
        res = milp(c=-C.npv.values, constraints=LinearConstraint(np.vstack(Aeq), np.concatenate(lb), np.concatenate(ub)),
                   integrality=np.ones(nC), bounds=Bounds(0, 1), options=dict(time_limit=120))
        if res.x is None:
            return None
        sel = np.where(res.x > 0.5)[0]
        return sel, C.npv.values[sel].sum(), corridor[sel].sum(), comfort[sel].sum(), feas[sel].sum()

    base = solve()
    cmax_res = milp(c=-corridor, constraints=LinearConstraint(np.vstack(rows), np.concatenate(lbs), np.concatenate(ubs)),
                    integrality=np.ones(nC), bounds=Bounds(0, 1), options=dict(time_limit=120))
    cmax = -cmax_res.fun
    pareto = []
    for frac in np.linspace(0, 1, 11):
        r = solve(frac * cmax)
        if r is None:
            continue
        pareto.append(dict(eps_share=frac, npv_musd=r[1] / 1e6, corridor_gwh=r[2], comfort_gain=r[3],
                           feasibility_gain=r[4], n_corridor_sites=int((S.set_index("k").segment.reindex(C.k.values[r[0]]) == "corridor").sum()),
                           sites=",".join(C.cell6.values[r[0]])))
    P = pd.DataFrame(pareto).drop_duplicates(subset=["sites"])
    P["pick_score"] = PROFIT_WEIGHT * P.npv_musd / P.npv_musd.max() + (1 - PROFIT_WEIGHT) * P.corridor_gwh / P.corridor_gwh.max()
    P.to_csv(OUT / "pareto_frontier.csv", index=False)
    print(P[["eps_share", "npv_musd", "corridor_gwh", "comfort_gain", "n_corridor_sites", "pick_score"]])
    chosen = P.sort_values("pick_score", ascending=False).iloc[0]
    print("chosen epsilon share", chosen.eps_share)
    sel = solve(chosen.eps_share * cmax)[0]
    pick = C.iloc[sel].merge(S[["k", "segment", "area", "state", "county", "lat", "lng", "local_kwh_12", "corr_kwh12",
                                "competitor_share_catchment", "anx_gain", "feas_gain", "screen_score", "demand_pot",
                                "d_nearest_tesla"]], on="k")
    pick = pick.sort_values("npv", ascending=False).reset_index(drop=True)
    pick.to_csv(OUT / "selected_10_cells.csv", index=False)
    json.dump(dict(chosen_eps_share=float(chosen.eps_share), npv_unconstrained_musd=float(P.npv_musd.max()),
                   npv_chosen_musd=float(chosen.npv_musd), corridor_gwh_chosen=float(chosen.corridor_gwh),
                   corridor_gwh_max=float(cmax), profit_weight=PROFIT_WEIGHT,
                   max_sites_per_state=MAX_SITES_PER_STATE),
              open(OUT / "selection_summary.json", "w"), indent=2)
    pd.set_option("display.width", 250)
    print(pick[["cell6", "segment", "area", "state", "county", "lat", "lng", "stalls", "interconnect", "megapacks",
                "months_to_energise", "npv", "utilization", "service_level", "d_nearest_tesla"]].round(2).to_string())
    pickle.dump(dict(df=df, S=S, C=C), open(PROC / "decision_state.pkl", "wb"))


if __name__ == "__main__":
    main()
