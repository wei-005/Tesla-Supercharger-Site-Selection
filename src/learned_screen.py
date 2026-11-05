"""Learned metro screening score (validated in improve_metro.py: rolling AUC 0.836 vs 0.751 demand-only).

Logistic regression on log features, fitted on all backtest years 2016-2025 (recency weighted 0.75^age),
applied to any network state. Features are multi-scale demand potential, Huff gross/net capture, Tesla
stalls within 5/15/30 km, recent openings within 15/40 km, competitor ports within 15 km, distance to the
nearest Tesla site and corridor flow through the cell.
Ranking is invariant to the year's EV-stock level (a constant shift of a log feature), so a model fitted on
2016-2025 can score the 2027 decision state.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import PROC, COMP_ATTR_MULT
from model import comp_ports
from improve_metro import FEATS_ALL, LOGF, radius_sum

FEAT_FILE = PROC / "metro_features.pkl"
PIPELINE = ["OPEN", "EXPANDING", "CONSTRUCTION", "PERMIT", "PLAN", "CLOSED_TEMP"]


def training_table(world):
    if FEAT_FILE.exists():
        return pd.read_pickle(FEAT_FILE)
    from improve_metro import build
    cache = pickle.load(open(PROC / "backtest_components.pkl", "rb"))
    D = build(world, cache)
    D.to_pickle(FEAT_FILE)
    return D


def fit(world, last_year=2025):
    D = training_table(world)
    tr = D[D.year <= last_year]
    rw = 0.75 ** (last_year - tr.year.values)
    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))
    mdl.fit(LOGF(tr[FEATS_ALL].values), tr.label.values, logisticregression__sample_weight=rw)
    return mdl


def features(world, full, mask, year, include_pipeline, beta_base):
    """same feature columns as the training table, for the rows `mask` of an evaluate_year() frame"""
    Q = full.attrs["Q"]
    c, d, k = world.ck_c, world.ck_d, world.ck_k
    df = full[mask]
    F = pd.DataFrame(index=df.index)
    for b, col in ((beta_base, "demand_pot"), (2.0, "demand_pot_b2.0"), (2.5, "demand_pot_b2.5"), (3.0, "demand_pot_b3.0")):
        F[col] = np.bincount(k, weights=Q[c] * (d + 3.0) ** (-b), minlength=len(full))[mask]
    for col in ("local_net", "local_gross", "d_nearest_tesla", "flow_through"):
        F[col] = df[col].values
    sc = world.sc
    net = sc[sc.status.isin(PIPELINE)] if include_pipeline else sc[sc.year_opened < year]
    st = np.where(net.stalls > 0, net.stalls, 12).astype(float)
    q = df[["lat", "lng"]].values
    for km in (5, 15, 30):
        F[f"stalls_{km}km"] = radius_sum(net[["lat", "lng"]].values, st, q, km)
    recent = sc[sc.year_opened.between(year - 2, year - 1)]
    for km in (15, 40):
        F[f"recent_open_{km}km"] = radius_sum(recent[["lat", "lng"]].values, np.ones(len(recent)), q, km)
    cp = comp_ports(world, year); m = cp > 0
    F["comp_ports_15km"] = radius_sum(world.comp[m][["lat", "lng"]].values, cp[m] * COMP_ATTR_MULT, q, 15)
    return F[FEATS_ALL]


def score(world, df, mask, year, include_pipeline, beta_base, mdl=None):
    mdl = mdl or fit(world)
    out = np.zeros(len(df))
    F = features(world, df, mask, year, include_pipeline, beta_base)
    out[np.where(mask)[0]] = mdl.predict_proba(LOGF(F.values))[:, 1]
    return out
