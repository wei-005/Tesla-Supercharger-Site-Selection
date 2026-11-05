"""Minimal TopoJSON decoder + Albers equal-area projection (no GIS dependencies).

Boundary data in data/raw/geo/: us-atlas states-10m (US Census cartographic boundaries) and
world-atlas countries-50m/110m (Natural Earth), both public-domain derived.
"""
import json
import numpy as np
from config import RAW

GEO = RAW / "geo"


def load_topo(name):
    return json.load(open(GEO / name))


def _arcs(topo):
    t = topo.get("transform")
    out = []
    for arc in topo["arcs"]:
        a = np.array(arc, dtype=float)
        if t:
            a = np.cumsum(a, axis=0) * t["scale"] + t["translate"]
        out.append(a)
    return out


def features(topo, obj):
    """yield (properties, id, list of polygons, each = list of rings (N x 2 lon/lat arrays))"""
    arcs = _arcs(topo)

    def ring(idx):
        pts = []
        for i in idx:
            a = arcs[i] if i >= 0 else arcs[~i][::-1]
            pts.append(a if not pts else a[1:])
        return np.vstack(pts)

    def geom(g):
        if g["type"] == "Polygon":
            return [[ring(r) for r in g["arcs"]]]
        if g["type"] == "MultiPolygon":
            return [[ring(r) for r in p] for p in g["arcs"]]
        return []

    for g in topo["objects"][obj]["geometries"]:
        yield g.get("properties", {}), g.get("id"), geom(g)


def albers(lon, lat, lon0=-96.0, lat0=37.5, p1=29.5, p2=45.5):
    """Albers equal-area conic (sphere), returns x, y in radians-scale units (y up)."""
    lon, lat = np.radians(lon), np.radians(lat)
    l0, f0, f1, f2 = map(np.radians, (lon0, lat0, p1, p2))
    n = (np.sin(f1) + np.sin(f2)) / 2
    C = np.cos(f1) ** 2 + 2 * n * np.sin(f1)
    r0 = np.sqrt(C - 2 * n * np.sin(f0)) / n
    r = np.sqrt(np.maximum(C - 2 * n * np.sin(lat), 0)) / n
    th = n * (lon - l0)
    return r * np.sin(th), r0 - r * np.cos(th)
