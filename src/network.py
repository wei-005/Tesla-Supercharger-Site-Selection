"""Step 2: H3 road graph, gravity OD flows, shortest paths, candidate cells.

- Road vertices are snapped to H3 res-8 cells, so separately digitised segments join into one graph.
- County population is aggregated to H3 res-4 OD zones; long-distance flows follow a gravity model.
- Each OD path is re-expressed as the sequence of H3 res-6 cells it crosses with a km position:
  those cells are both the "arcs" of the flow-refuelling model and the candidate sites.
"""
import json, pickle, time
import numpy as np
import pandas as pd
import networkx as nx
import h3
from config import (PROC, H3_NODE_RES, H3_CAND_RES, H3_OD_RES, TRIP_MIN_KM, TRIP_MAX_KM)

R_EARTH = 6371.0


def hav(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def build_graph():
    roads = json.load(open(PROC / "roads_us.json"))
    G = nx.Graph()
    for r in roads:
        cells = [h3.latlng_to_cell(lat, lng, H3_NODE_RES) for lng, lat in r["coords"]]
        pts = [(lat, lng) for lng, lat in r["coords"]]
        for (c1, p1), (c2, p2) in zip(zip(cells, pts), zip(cells[1:], pts[1:])):
            if c1 == c2:
                continue
            d = float(hav(p1[0], p1[1], p2[0], p2[1]))
            t = d / r["speed"]
            if G.has_edge(c1, c2):
                if G[c1][c2]["t"] <= t:
                    continue
            G.add_edge(c1, c2, km=d, t=t, inter=r["interstate"])
    # bridge tiny gaps: join nodes whose res-8 cells are adjacent (digitising breaks)
    nodes = list(G.nodes)
    nodeset = set(nodes)
    for n in nodes:
        for nb in h3.grid_disk(n, 1):
            if nb != n and nb in nodeset and not G.has_edge(n, nb):
                a = h3.cell_to_latlng(n); b = h3.cell_to_latlng(nb)
                d = float(hav(a[0], a[1], b[0], b[1]))
                G.add_edge(n, nb, km=d, t=d / 60, inter=False)
    comp = max(nx.connected_components(G), key=len)
    G = G.subgraph(comp).copy()
    return G


def od_zones(counties, G):
    counties = counties.copy()
    counties["z"] = [h3.cell_to_parent(c, H3_OD_RES) for c in counties.cell6]
    z = counties.groupby("z").apply(lambda d: pd.Series(dict(
        pop=d["pop"].sum(), ev=d.ev_2023.sum(),
        lat=np.average(d.lat, weights=d["pop"].clip(lower=1)),
        lng=np.average(d.lng, weights=d["pop"].clip(lower=1)),
        state=d.sort_values("pop").state.iloc[-1])), include_groups=False).reset_index()
    z = z[z["pop"] >= 60_000].reset_index(drop=True)
    # snap zone centroid to the nearest graph node (search outward on H3 rings)
    nodeset = set(G.nodes)
    snaps = []
    for lat, lng in zip(z.lat, z.lng):
        c = h3.latlng_to_cell(lat, lng, H3_NODE_RES)
        found = None
        for k in range(0, 60, 2):
            ring = [x for x in h3.grid_ring(c, k)] if k else [c]
            hit = [x for x in ring if x in nodeset]
            if hit:
                found = hit[0]; break
        snaps.append(found)
    z["node"] = snaps
    return z[z.node.notna()].reset_index(drop=True)


def gravity_pairs(z, max_pairs=7000):
    lat = z.lat.values; lng = z.lng.values
    D = hav(lat[:, None], lng[:, None], lat[None, :], lng[None, :]) * 1.2
    # EV-weighted gravity: long trips by EV owners; distance decay for long-distance travel
    w = z.ev.values.clip(min=50) ** 0.75 * z["pop"].values ** 0.25
    F = (w[:, None] * w[None, :]) * np.exp(-D / 380.0)
    mask = (D >= TRIP_MIN_KM) & (D <= TRIP_MAX_KM)
    iu = np.triu_indices(len(z), 1)
    sel = mask[iu]
    i, j = iu[0][sel], iu[1][sel]
    f = F[i, j]
    order = np.argsort(-f)[:max_pairs]
    pairs = pd.DataFrame(dict(o=i[order], d=j[order], flow=f[order], crow_km=D[i[order], j[order]]))
    pairs["flow"] = pairs.flow / pairs.flow.sum() * 1e6   # normalised flow units (EV trips/yr index)
    print(f"gravity: kept {len(pairs)} pairs covering {f[order].sum() / f.sum():.1%} of EV gravity flow")
    return pairs


def compute_paths(G, z, pairs):
    paths = []
    by_o = pairs.groupby("o")
    t0 = time.time()
    for o, grp in by_o:
        src = z.node[o]
        dist, pred = nx.single_source_dijkstra(G, src, weight="t")
        for _, r in grp.iterrows():
            dst = z.node[int(r.d)]
            if dst not in pred:
                continue
            nodes = pred[dst]
            km = np.concatenate([[0], np.cumsum([G[a][b]["km"] for a, b in zip(nodes, nodes[1:])])])
            inter = np.mean([G[a][b]["inter"] for a, b in zip(nodes, nodes[1:])]) if len(nodes) > 1 else 0
            cells, pos = [], []
            for n, k in zip(nodes, km):
                c = h3.cell_to_parent(n, H3_CAND_RES)
                if not cells or cells[-1] != c:
                    cells.append(c); pos.append(k)
            paths.append(dict(o=int(r.o), d=int(r.d), flow=float(r.flow), length_km=float(km[-1]),
                              interstate_share=float(inter), cells=cells, pos=np.array(pos, dtype=np.float32)))
    print(f"paths: {len(paths)} in {time.time() - t0:.0f}s")
    return paths


def candidates(paths, counties):
    rows = {}
    for p in paths:
        for c in p["cells"]:
            rows.setdefault(c, dict(on_path=True))
    # urban/suburban cells: ring around populated county centres
    for _, r in counties[counties["pop"] >= 40_000].iterrows():
        k = 3 if r["pop"] < 300_000 else (6 if r["pop"] < 1_500_000 else 9)
        for c in h3.grid_disk(r.cell6, k):
            rows.setdefault(c, dict(on_path=False))
    cand = pd.DataFrame([dict(cell6=c, **v) for c, v in rows.items()])
    ll = np.array([h3.cell_to_latlng(c) for c in cand.cell6])
    cand["lat"], cand["lng"] = ll[:, 0], ll[:, 1]
    cand = cand[(cand.lat > 24) & (cand.lat < 49.6) & (cand.lng > -125) & (cand.lng < -66.5)]
    return cand.reset_index(drop=True)


if __name__ == "__main__":
    t = time.time()
    G = build_graph(); print("graph", G.number_of_nodes(), G.number_of_edges(), f"{time.time() - t:.0f}s")
    counties = pd.read_csv(PROC / "counties.csv")
    z = od_zones(counties, G); print("OD zones", len(z))
    pairs = gravity_pairs(z)
    paths = compute_paths(G, z, pairs)
    cand = candidates(paths, counties); print("candidates", len(cand), "on path", int(cand.on_path.sum()))
    z.to_csv(PROC / "od_zones.csv", index=False)
    pairs.to_csv(PROC / "od_pairs.csv", index=False)
    cand.to_csv(PROC / "candidates.csv", index=False)
    with open(PROC / "paths.pkl", "wb") as fh:
        pickle.dump(paths, fh)
