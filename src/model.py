"""Step 3 core: per-year marginal value of opening a Supercharger at every candidate H3 cell.

Two demand channels, both on the H3 base:
  corridor  - flow-refuelling on the 7,000 gravity OD paths. A trip is made by EV with probability
              pi(max gap) (0 if any gap > usable range, decaying beyond comfort range). A new site splits
              the gap it lands in. Outputs: feasibility gain (objective 1), comfort/anxiety gain
              (objective 2), net-new corridor kWh and gross corridor kWh captured (Huff along the path).
  local     - MCI/Huff choice between Tesla sites, competitor DCFC (AFDC) and an outside option, for
              every county within ~35 km. Net-new Tesla kWh = gain in Tesla share (own cannibalisation
              and congestion relief are netted inside the Huff denominator).
"""
import pickle
from dataclasses import dataclass
import numpy as np
import pandas as pd
import h3
from config import *
from network import hav


@dataclass
class Params:
    beta: float = HUFF_BETA
    r_comfort: float = RANGE_COMFORT_KM
    r_usable: float = RANGE_USABLE_KM
    tau: float = 60.0
    gamma: float = HUFF_GAMMA_STALLS
    u0_mult: float = OUTSIDE_OPTION
    new_stalls: int = 12


class World:
    """Static, year-independent structures (built once)."""

    def __init__(self):
        self.sc = pd.read_csv(PROC / "superchargers.csv")
        self.counties = pd.read_csv(PROC / "counties.csv")
        self.comp = pd.read_csv(PROC / "competitors_grid.csv")
        self.cand = pd.read_csv(PROC / "candidates.csv")
        with open(PROC / "paths.pkl", "rb") as fh:
            self.paths = pickle.load(fh)
        self.cand_idx = {c: i for i, c in enumerate(self.cand.cell6)}
        self._build_county_links()
        self._build_path_tables()

    # ---------- county <-> cell links for Huff
    def _build_county_links(self):
        cell_to_cty = {}
        for i, c in enumerate(self.counties.cell6):
            for nb in h3.grid_disk(c, LOCAL_RADIUS_RING):
                cell_to_cty.setdefault(nb, []).append(i)
        self.cell_to_cty = cell_to_cty
        clat, clng = self.counties.lat.values, self.counties.lng.values
        rows_c, rows_k, rows_d = [], [], []
        for k, (cell, lat, lng) in enumerate(zip(self.cand.cell6, self.cand.lat, self.cand.lng)):
            for i in cell_to_cty.get(cell, ()):
                rows_c.append(i); rows_k.append(k)
                rows_d.append(hav(lat, lng, clat[i], clng[i]) * DETOUR_FACTOR)
        self.ck_c = np.array(rows_c); self.ck_k = np.array(rows_k); self.ck_d = np.array(rows_d)

    def point_links(self, lats, lngs, cells):
        """county links for arbitrary points (stations, competitor cells): (point_idx, county_idx, dist)"""
        clat, clng = self.counties.lat.values, self.counties.lng.values
        P, C, D = [], [], []
        for p, (cell, lat, lng) in enumerate(zip(cells, lats, lngs)):
            for i in self.cell_to_cty.get(cell, ()):
                P.append(p); C.append(i); D.append(hav(lat, lng, clat[i], clng[i]) * DETOUR_FACTOR)
        return np.array(P, dtype=int), np.array(C, dtype=int), np.array(D)

    # ---------- path tables for the flow-refuelling model
    def _build_path_tables(self):
        pid, cell, pos = [], [], []
        for q, p in enumerate(self.paths):
            pid.extend([q] * len(p["cells"])); cell.extend(p["cells"]); pos.extend(p["pos"].tolist())
        P = pd.DataFrame(dict(pid=pid, cell=cell, pos=pos))
        P["k"] = P.cell.map(self.cand_idx).fillna(-1).astype(int)
        self.P = P
        # station -> path matches (station within DETOUR_RING of a path cell)
        rows = []
        for s, c in enumerate(self.sc.cell6):
            for nb in h3.grid_disk(c, DETOUR_RING):
                rows.append((s, nb))
        S = pd.DataFrame(rows, columns=["sid", "cell"]).merge(P[["pid", "cell", "pos"]], on="cell")
        self.SP = S.groupby(["sid", "pid"], as_index=False).pos.min()
        self.path_len = np.array([p["length_km"] for p in self.paths])
        self.path_flow = np.array([p["flow"] for p in self.paths])
        self.path_groups = {q: g for q, g in P[P.k >= 0].groupby("pid")}


def network_state(world, year, include_pipeline=False):
    """Tesla sites existing at 1 Jan `year` (or today's network incl. pipeline)."""
    sc = world.sc
    if include_pipeline:
        m = sc.status.isin(["OPEN", "EXPANDING", "CONSTRUCTION", "PERMIT", "PLAN", "CLOSED_TEMP"])
    else:
        m = sc.year_opened.notna() & (sc.year_opened < year)
    return np.where(m.values)[0]


def comp_ports(world, year):
    col = ("eff_ports_2019" if year <= 2020 else "eff_ports_2022" if year <= 2023
           else "eff_ports_2025" if year <= 2026 else "eff_ports_now")
    return world.comp[col].values.astype(float)


def ev_index(year):
    return EV_INDEX.get(year, EV_INDEX[max(EV_INDEX)])


def trip_prob(maxgap, prm):
    return np.where(maxgap > prm.r_usable, 0.0, np.exp(-np.maximum(0.0, maxgap - prm.r_comfort) / prm.tau))


def size_attr(n, gamma):
    """MCI size term with saturation: elastic up to ~16 stalls, nearly flat beyond (size then buys queue
    capacity, not extra drivers)."""
    n = np.maximum(np.asarray(n, dtype=float), 2)
    return np.minimum(n, ATTR_SAT_STALLS) ** gamma * (np.maximum(n, ATTR_SAT_STALLS) / ATTR_SAT_STALLS) ** (gamma / 4)


def attractiveness(stalls, fac, prm):
    mult = np.array([FACILITY_MULT.get(f, 0.95) for f in fac]) if fac is not None else 1.0
    return size_attr(stalls, prm.gamma) * mult


def evaluate_year(world, year, prm=Params(), include_pipeline=False, comp_year=None):
    """Return per-candidate components for opening one `prm.new_stalls`-stall site in `year`."""
    sids = network_state(world, year, include_pipeline)
    sc = world.sc.iloc[sids].copy()
    sc.loc[sc.stalls <= 0, "stalls"] = 12
    nK = len(world.cand)
    idx = ev_index(year)
    Q = world.counties.ev_2023.values * idx * PUBLIC_DCFC_KWH_PER_EV_YEAR     # kWh/yr public DCFC demand
    decay = lambda d: (d + 3.0) ** (-prm.beta)
    U0 = prm.u0_mult * size_attr(8, prm.gamma) * decay(15.0)

    # ---------------- local Huff baseline sums (Tesla, competitor) per county
    A_t = attractiveness(sc.stalls.values, sc.facility.values, prm)
    sp, sc_c, sd = world.point_links(sc.lat.values, sc.lng.values, sc.cell6.values)
    cp_ports = comp_ports(world, comp_year or year)
    m = cp_ports > 0
    comp = world.comp[m]
    cp, cc, cd = world.point_links(comp.lat.values, comp.lng.values, comp.cell6.values)
    A_c = (cp_ports[m] * COMP_ATTR_MULT).clip(min=0.5) ** prm.gamma
    S_C = np.bincount(cc, weights=A_c[cp] * decay(cd), minlength=len(Q))

    # congestion fixed point on Tesla attractiveness (one pass is enough at this resolution)
    cong = np.ones(len(sc))
    for _ in range(2):
        S_T = np.bincount(sc_c, weights=(A_t * cong)[sp] * decay(sd), minlength=len(Q))
        denom = S_T + S_C + U0
        load = np.bincount(sp, weights=Q[sc_c] * (A_t * cong)[sp] * decay(sd) / denom[sc_c], minlength=len(sc))
        util = load / 365.0 / (np.maximum(sc.stalls.values, 2) * STALL_KWH_CAP_DAY)
        cong = np.exp(-CONGESTION_KAPPA * np.maximum(0.0, util - 1.0))
    S_T = np.bincount(sc_c, weights=(A_t * cong)[sp] * decay(sd), minlength=len(Q))
    denom = S_T + S_C + U0

    a_new = size_attr(prm.new_stalls, prm.gamma) * decay(world.ck_d)
    c = world.ck_c
    gain = Q[c] * ((S_T[c] + a_new) / (denom[c] + a_new) - S_T[c] / denom[c])
    gross = Q[c] * a_new / (denom[c] + a_new)
    local_net = np.bincount(world.ck_k, weights=gain, minlength=nK)
    local_gross = np.bincount(world.ck_k, weights=gross, minlength=nK)
    demand_pot = np.bincount(world.ck_k, weights=Q[c] * decay(world.ck_d), minlength=nK)

    # ---------------- corridor flow-refuelling
    SP = world.SP[world.SP.sid.isin(sids)]
    st_attr = pd.Series(A_t * cong, index=sids)
    SP = SP.assign(A=st_attr.reindex(SP.sid.values).values)
    st_by_path = {q: (g.pos.values, g.A.values) for q, g in SP.groupby("pid")}
    corr_net = np.zeros(nK); corr_gross = np.zeros(nK); feas = np.zeros(nK); anx = np.zeros(nK)
    flow_through = np.zeros(nK)
    trips_ok = 0.0
    A_new = float(size_attr(prm.new_stalls, prm.gamma))
    for q, g in world.path_groups.items():
        L = world.path_len[q]; f = world.path_flow[q] * idx
        E = KWH_PER_KM * max(0.0, L - 0.6 * prm.r_usable)
        pos, A = st_by_path.get(q, (np.array([]), np.array([])))
        b = np.sort(np.concatenate([[0.0], pos[(pos > 0) & (pos < L)], [L]]))
        gaps = np.diff(b)
        gi = int(np.argmax(gaps)); gmax = gaps[gi]
        g2 = np.partition(gaps, -2)[-2] if len(gaps) > 1 else 0.0
        p0 = float(trip_prob(np.array([gmax]), prm)[0])
        trips_ok += f * p0
        x = g.pos.values.astype(float); ks = g.k.values
        j = np.clip(np.searchsorted(b, x, side="right") - 1, 0, len(gaps) - 1)
        left = x - b[j]; right = b[j + 1] - x
        other = np.where(j == gi, g2, gmax)
        newmax = np.maximum(other, np.maximum(left, right))
        p1 = trip_prob(newmax, prm)
        np.add.at(corr_net, ks, f * E * (p1 - p0))
        np.add.at(corr_gross, ks, f * E * p1 * A_new / (A.sum() + A_new))
        np.add.at(feas, ks, f * ((gmax > prm.r_usable) & (newmax <= prm.r_usable)))
        relu = lambda v: np.maximum(0.0, v - prm.r_comfort)
        np.add.at(anx, ks, f * (relu(gaps[j]) - relu(left) - relu(right)))
        np.add.at(flow_through, ks, f)

    # distance to nearest existing Tesla site (km)
    from sklearn.neighbors import BallTree
    tree = BallTree(np.radians(sc[["lat", "lng"]].values), metric="haversine")
    dnear = tree.query(np.radians(world.cand[["lat", "lng"]].values), k=1)[0][:, 0] * 6371.0

    out = pd.DataFrame(dict(cell6=world.cand.cell6, lat=world.cand.lat, lng=world.cand.lng,
                            local_net=local_net, local_gross=local_gross, demand_pot=demand_pot,
                            corr_net=corr_net, corr_gross=corr_gross, feas_gain=feas, anx_gain=anx,
                            flow_through=flow_through, d_nearest_tesla=dnear))
    out.attrs.update(dict(year=year, n_sites=len(sc), trips_enabled=trips_ok,
                          util_mean=float(np.mean(util)), util_p90=float(np.quantile(util, 0.9))))
    # kept for the decision stage (site-level Huff with other stall counts, competitor overlap)
    out.attrs.update(dict(Q=Q, S_T=S_T, S_C=S_C, U0=U0, denom=denom, station_util=util, station_ids=sids))
    return out


def norm(v):
    s = np.quantile(v[v > 0], 0.99) if (v > 0).any() else 1.0
    return np.clip(v / s, 0, 3)


COMPONENTS = {"local": "local_net", "local_gross": "local_gross", "corr": "corr_net",
              "corr_gross": "corr_gross", "feas": "feas_gain", "anx": "anx_gain"}


def score(df, w):
    """single-site value used for ranking (weights calibrated in backtest)"""
    s = np.zeros(len(df))
    for k, col in COMPONENTS.items():
        if w.get(k, 0):
            s += w[k] * norm(df[col].values)
    return s
