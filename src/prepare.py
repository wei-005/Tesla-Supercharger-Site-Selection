"""Step 1: clean raw inputs into processed tables keyed on H3 cells."""
import json
import numpy as np
import pandas as pd
import h3
from config import RAW, PROC, H3_CAND_RES, H3_NODE_RES, COMP_GRID_DEG

STATE_ABBR = {'Alabama':'AL','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT',
 'Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA','Idaho':'ID','Illinois':'IL','Indiana':'IN',
 'Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA',
 'Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV',
 'New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
 'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
 'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA',
 'West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY','Alaska':'AK','Hawaii':'HI'}

# AFDC light-duty EV registrations by state, Dec 2023 (afdc.energy.gov/data/10962)
EV_REG_2023 = {'AL':13047,'AZ':89798,'AR':7108,'CA':1256646,'CO':90083,'CT':31557,'DE':8435,'DC':8066,'FL':254878,
 'GA':92368,'ID':8501,'IL':99573,'IN':26101,'IA':9031,'KS':11271,'KY':11617,'LA':8150,'ME':7377,'MD':72139,'MA':73768,
 'MI':50284,'MN':37050,'MS':3590,'MO':26861,'MT':4608,'NE':6920,'NV':47361,'NH':9861,'NJ':134753,'NM':10276,'NY':131250,
 'NC':70164,'ND':959,'OH':50393,'OK':22843,'OR':64361,'PA':70154,'RI':6396,'SC':20873,'SD':1675,'TN':33221,'TX':230125,
 'UT':39998,'VT':7816,'VA':84936,'WA':152101,'WV':2758,'WI':24943,'WY':1139}


def prep_superchargers():
    sc = pd.read_csv(PROC / "superchargers_conus.csv")
    sc["year_opened"] = pd.to_datetime(sc["date_opened"]).dt.year
    sc["cell6"] = [h3.latlng_to_cell(a, b, H3_CAND_RES) for a, b in zip(sc.lat, sc.lng)]
    sc["cell8"] = [h3.latlng_to_cell(a, b, H3_NODE_RES) for a, b in zip(sc.lat, sc.lng)]
    sc.to_csv(PROC / "superchargers.csv", index=False)
    return sc


def prep_counties():
    c = pd.read_csv(RAW / "dl_counties2" / "counties.csv", dtype={"FIPS Code": str})
    c["fips"] = c["FIPS Code"].str.zfill(5)
    cc = pd.read_csv(RAW / "dl_counties" / "county_centers.csv", dtype={"fips": str})
    cc["fips"] = cc.fips.str.zfill(5)
    m = c.merge(cc[["fips", "pclat10", "pclon10"]], on="fips", how="inner")
    m = m.rename(columns={"pclat10": "lat", "pclon10": "lng", "Population": "pop", "Area": "area_sqmi"})
    m["state"] = m.State.map(STATE_ABBR)
    m = m[(m.lat > 24) & (m.lat < 49.6) & (m.lng > -125) & (m.lng < -66.5) & m.state.notna()].copy()
    m["pop"] = pd.to_numeric(m["pop"], errors="coerce").fillna(0)
    st_pop = m.groupby("state")["pop"].sum()
    m["ev_2023"] = m["pop"] * m.state.map(lambda s: EV_REG_2023.get(s, 0) / max(st_pop.get(s, 1), 1))
    m["density_km2"] = m["pop"] / (pd.to_numeric(m["area_sqmi"], errors="coerce").clip(lower=1) * 2.59)
    m["cell6"] = [h3.latlng_to_cell(a, b, H3_CAND_RES) for a, b in zip(m.lat, m.lng)]
    m = m[["fips", "County", "state", "lat", "lng", "pop", "area_sqmi", "density_km2", "ev_2023", "cell6"]]
    m.to_csv(PROC / "counties.csv", index=False)
    return m


def prep_competitors():
    """Decode AFDC non-Tesla public DCFC, aggregated to a 0.3 deg grid; values are effective ports
    (ports x power weight: >=150kW 1.0, 100-149kW 0.7, <100kW 0.4), cumulative at end of 2019/2022/2025/now."""
    s = open(RAW / "competitor_dcfc_grid.txt").read().strip()
    rows = []
    for i in range(0, len(s), 11):
        r = s[i:i + 11]
        key = int(r[0:3], 36)
        gi, gj = divmod(key, 200)
        vals = [int(r[3 + 2 * b:5 + 2 * b], 36) for b in range(4)]
        lat = 24 + (gi + 0.5) * COMP_GRID_DEG
        lng = -125 + (gj + 0.5) * COMP_GRID_DEG
        rows.append(dict(lat=lat, lng=lng, eff_ports_2019=vals[0], eff_ports_2022=vals[1],
                         eff_ports_2025=vals[2], eff_ports_now=vals[3]))
    df = pd.DataFrame(rows)
    df["cell6"] = [h3.latlng_to_cell(a, b, H3_CAND_RES) for a, b in zip(df.lat, df.lng)]
    df.to_csv(PROC / "competitors_grid.csv", index=False)
    return df


def prep_roads():
    src = RAW / "dl_roads" / "ne_10m_roads.geojson"
    if not src.exists() and (PROC / "roads_us.json").exists():
        print("roads: raw Natural Earth file not present, reusing data/processed/roads_us.json")
        return json.load(open(PROC / "roads_us.json"))
    g = json.load(open(src))
    speed = {"Interstate": 105, "Federal": 85, "State": 75}
    out = []
    for f in g["features"]:
        p = f["properties"]
        if p.get("sov_a3") != "USA" or p.get("type") == "Ferry Route":
            continue
        geom = f["geometry"]
        lines = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        v = speed.get(p.get("level"), 70) + (10 if p.get("expressway") == 1 else 0)
        is_inter = p.get("level") == "Interstate"
        for ln in lines:
            out.append(dict(speed=v, interstate=is_inter, coords=ln))
    with open(PROC / "roads_us.json", "w") as fh:
        json.dump(out, fh)
    return out


if __name__ == "__main__":
    sc = prep_superchargers(); print("superchargers", sc.shape)
    co = prep_counties(); print("counties", co.shape, "pop", int(co["pop"].sum()), "EV", int(co.ev_2023.sum()))
    cp = prep_competitors(); print("competitor cells", cp.shape, cp[["eff_ports_2019", "eff_ports_now"]].sum().to_dict())
    rd = prep_roads(); print("road lines", len(rd))
