"""Grid-headroom stress test on the selected sites: best configuration when feeder headroom is
lower than the base assumption (no public national hosting-capacity data exists)."""
import pickle
import numpy as np
import pandas as pd
import config
import select_sites as ss
from model import World

def main():
    st = pickle.load(open(config.PROC / "decision_state.pkl", "rb"))
    df, S = st["df"], st["S"]
    world = World()
    ss.TRIP_SCALE = world.counties.ev_2023.sum() * config.CORRIDOR_TRIPS_PER_EV_YEAR / 1e6
    sel = pd.read_csv(config.OUT / "selected_10_cells.csv")
    base = dict(config.AREA_HEADROOM_KW)
    rows = []
    for mult in (1.0, 0.5, 0.3):
        ss.AREA_HEADROOM_KW.update({k: v * mult for k, v in base.items()})
        for r in sel.itertuples():
            s = S[S.cell6 == r.cell6].iloc[0]
            f_loc, _ = ss.site_local_fn(df, s.k, world)
            site = dict(area=s.area, state=s.state, corr_kwh12=s.corr_kwh12, local_kwh_by_n=f_loc)
            best = None
            for opt in config.INTERCONNECT_OPTIONS:
                for mp in (0, 1, 2):
                    res = ss.npv_of(site, int(r.stalls), opt, mp, int(r.l2))
                    if res and (best is None or res["npv"] > best["npv"]):
                        best = dict(res, interconnect=opt, megapacks=mp)
            rows.append(dict(headroom_mult=mult, cell6=r.cell6, state=r.state, county=r.county, stalls=r.stalls,
                             best_interconnect=best["interconnect"], megapacks=best["megapacks"],
                             months_to_energise=best["months_to_energise"], npv_musd=round(best["npv"] / 1e6, 2),
                             service_level=round(best["service_level"], 3), curtailed_mwh_y1=round(best["curtailed_kwh_y1"] / 1e3)))
    ss.AREA_HEADROOM_KW.update(base)
    out = pd.DataFrame(rows)
    out.to_csv(config.OUT / "grid_headroom_sensitivity.csv", index=False)
    print(out.pivot_table(index=["state", "county"], columns="headroom_mult",
                          values=["best_interconnect", "megapacks", "npv_musd"], aggfunc="first").to_string())

if __name__ == "__main__":
    main()
