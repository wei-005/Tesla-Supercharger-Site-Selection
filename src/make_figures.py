"""Static overview map (vector SVG) of the 10 recommended sites over the Tesla network, for the report.

    python src/make_figures.py   ->  outputs/fig_overview.svg
"""
import numpy as np
import pandas as pd
from config import OUT, PROC
from geo import load_topo, features, albers
from make_map import CITY, SHOW_STATUS

W, H = 1600, 900
VY0, VH = 120, 760          # visible window (crops empty Canada / ocean)
ABBR = {"Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT",
        "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
        "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
        "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
        "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
        "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
        "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
        "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}
SMALL = {"CT", "DE", "MA", "MD", "NH", "NJ", "RI", "VT"}          # too small to label at this scale
NUDGE = {"FL": (18, 10), "MI": (14, 22), "LA": (-10, 0), "CA": (-8, 10), "ID": (0, 22), "KY": (8, 2), "VA": (8, 4), "NY": (8, 6)}

FIG_OFFSET = {"862a94d07ffffff": (34, 52)}     # Toledo: below-right here, clear of the Colorado Springs label

# projection fitted to CONUS, with extra room on the Pacific side for left-hand labels
bx, by = albers(np.array([-124.8, -66.9, -124.8, -66.9, -96, -96]), np.array([24.5, 24.5, 49.4, 49.4, 24.5, 49.4]))
x0, x1, y0, y1 = bx.min(), bx.max(), by.min(), by.max()
PAD_L, PAD_R, PAD_T, PAD_B = 230, 150, 40, 50
k = min((W - PAD_L - PAD_R) / (x1 - x0), (H - PAD_T - PAD_B) / (y1 - y0))
ox = PAD_L + ((W - PAD_L - PAD_R) - k * (x1 - x0)) / 2
oy = PAD_T + ((H - PAD_T - PAD_B) - k * (y1 - y0)) / 2


def P(lon, lat):
    x, y = albers(np.asarray(lon, float), np.asarray(lat, float))
    return ox + (x - x0) * k, H - (oy + (y - y0) * k)


def path(polys, keep=lambda lon, lat: True):
    d = []
    for poly in polys:
        for r in poly:
            if not keep(r[:, 0], r[:, 1]):
                continue
            x, y = P(r[:, 0], r[:, 1])
            d.append("M" + "L".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y)) + "Z")
    return "".join(d)


def main():
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {VY0} {W} {VH}" font-family="Noto Sans CJK SC, Noto Sans SC, PingFang SC, Microsoft YaHei, sans-serif">',
           '<defs><clipPath id="vp"><rect y="%d" width="%d" height="%d" rx="14"/></clipPath>'
           '<radialGradient id="sea" cx="50%%" cy="45%%" r="75%%"><stop offset="0" stop-color="#0f1c34"/><stop offset="1" stop-color="#070d1b"/></radialGradient></defs>' % (VY0, W, VH),
           '<g clip-path="url(#vp)">', f'<rect width="{W}" height="{H}" fill="url(#sea)"/>']
    # neighbours (Canada, Mexico, Caribbean) - only rings in the North American window
    near = lambda lon, lat: (lon.max() > -140) and (lon.min() < -50) and (lat.max() > 5) and (lat.min() < 72)
    for props, _, polys in features(load_topo("countries-50m.json"), "countries"):
        if props.get("name") in ("United States of America",):
            continue
        d = path(polys, near)
        if d:
            out.append(f'<path d="{d}" fill="#1d3346" stroke="#3d5a73" stroke-width="0.8"/>')
    us = load_topo("states-10m.json")
    labels = []
    for props, _, polys in features(us, "states"):
        ab = ABBR.get(props["name"])
        if not ab:
            continue
        out.append(f'<path d="{path(polys)}" fill="#24304d" stroke="none"/>')
        big = max((p[0] for p in polys), key=len)
        x, y = P(big[:, 0], big[:, 1])
        if ab not in SMALL:
            dx, dy = NUDGE.get(ab, (0, 0))
            labels.append((x.mean() + dx, y.mean() + dy, ab))
    for props, _, polys in features(us, "states"):
        if ABBR.get(props["name"]):
            out.append(f'<path d="{path(polys)}" fill="none" stroke="#8ea3c2" stroke-opacity=".42" stroke-width="1" stroke-dasharray="5 4"/>')
    for x, y, ab in labels:
        out.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="15" fill="#8796ad" text-anchor="middle" dominant-baseline="middle">{ab}</text>')
    # Tesla network
    sc = pd.read_csv(PROC / "superchargers.csv")
    sc = sc[sc.status.isin(SHOW_STATUS)]
    for pipe, colr in ((False, "#8fa0bb"), (True, "#d8b46f")):
        s = sc[sc.status.map(SHOW_STATUS).eq("pipeline") == pipe]
        x, y = P(s.lng.values, s.lat.values)
        out.append(f'<g fill="{colr}" fill-opacity="{.5 if not pipe else .7}">' +
                   "".join(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="1.7"/>' for a, b in zip(x, y)) + "</g>")
    # 10 sites
    T = pd.read_csv(OUT / "final_10_sites.csv")
    halo = 'paint-order="stroke" stroke="#050a14" stroke-width="5" stroke-linejoin="round"'
    for _, r in T.iterrows():
        cn, en, dx, dy = CITY[r["H3_res6"]]
        dx, dy = FIG_OFFSET.get(r["H3_res6"], (dx * 1.25, dy * 1.25))
        colr = "#f6b26b" if r["类型"] == "高速走廊" else "#8ab4f8"
        x, y = P(r["经度"], r["纬度"])
        desc = f'{r["超充桩(V4)"]} 桩超充 · {r["慢充(目的地L2)"]} 慢充' + (" · I-5 走廊" if r["类型"] == "高速走廊" else "")
        anchor = "end" if dx < 0 else ("middle" if dx == 0 else "start")
        L = max(np.hypot(dx, dy), 1)
        lx, ly = (x + dx, y + dy - 14) if dx == 0 else (x + dx - (6 if dx > 0 else -6), y + dy)
        out.append(f'<line x1="{x + dx / L * 17:.1f}" y1="{y + dy / L * 17:.1f}" x2="{lx:.1f}" y2="{ly:.1f}" stroke="#ffffff" stroke-opacity=".6" stroke-width="1.2"/>')
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="19" fill="{colr}" fill-opacity=".22"/>'
                   f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{colr}" stroke="#ffffff" stroke-opacity=".55" stroke-width="2.5"/>'
                   f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="#0b1020"/>')
        tx, ty = x + dx, y + dy
        base = ty - 18 if dx != 0 else ty
        out.append(f'<g text-anchor="{anchor}" {halo}>'
                   f'<text x="{tx:.1f}" y="{base:.1f}" font-size="25" font-weight="700" fill="#ffffff">{cn}'
                   f'<tspan font-size="17" fill="{colr}" dx="7">{r["排名"]}</tspan></text>'
                   f'<text x="{tx:.1f}" y="{base + 21:.1f}" font-size="17" fill="#eef2f8">{en}</text>'
                   f'<text x="{tx:.1f}" y="{base + 40:.1f}" font-size="15" fill="#c9d4e4">{desc}</text></g>')
    # legend
    lg = [("#8ab4f8", "都市圈站（9）", 7), ("#f6b26b", "高速走廊站（1）", 7), ("#8fa0bb", "特斯拉已开通", 2.5), ("#d8b46f", "特斯拉在建/许可/规划", 2.5)]
    gx, gy = 34, VY0 + VH - 112
    out.append(f'<rect x="{gx - 14}" y="{gy - 24}" width="250" height="122" rx="10" fill="#0b1324" fill-opacity=".8" stroke="#ffffff" stroke-opacity=".12"/>')
    for i, (c, t, rad) in enumerate(lg):
        out.append(f'<circle cx="{gx + 6}" cy="{gy + i * 26}" r="{rad}" fill="{c}"/><text x="{gx + 24}" y="{gy + i * 26 + 5}" font-size="15" fill="#dfe6f0">{t}</text>')
    out.append(f'<text x="{W - 16}" y="{VY0 + VH - 12}" font-size="11" fill="#6f7f98" text-anchor="end">边界：US Census / Natural Earth · 特斯拉网络：supercharge.info · Albers 等积投影</text>')
    out.append("</g></svg>")
    (OUT / "fig_overview.svg").write_text("\n".join(out), encoding="utf-8")
    print("outputs/fig_overview.svg")


if __name__ == "__main__":
    main()
