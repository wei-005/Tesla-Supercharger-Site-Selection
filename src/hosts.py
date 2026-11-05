"""Step 6: map each selected H3 cell to a real host property (Google Places lookups done in the chat
session, recorded here) and verify distances to the cell centre and to every Tesla site incl. pipeline."""
import numpy as np, pandas as pd
from config import PROC, OUT

HOSTS = [  # cell6, host name, address, lat, lng, host type, alternate
 ("8629a01a7ffffff", "Lakeview Valero (24h fuel, truck parking)", "19560 Lakeview Ave, Nuevo, CA 92567", 33.8367451, -117.1231892, "fuel_travel_center", "Nuevo Village Shopping Center, 29616 Nuevo Rd, Nuevo, CA"),
 ("86283631fffffff", "CA Stop Gas Station (I-5 / Manthey Rd)", "2224 Manthey Rd, Stockton, CA 95206", 37.9255749, -121.2954002, "fuel_travel_center", "24 Hours Gas and Mart, 1901 S El Dorado St, Stockton, CA"),
 ("8629b6d37ffffff", "Arizona Center (garage + dining)", "455 N 3rd St, Phoenix, AZ 85004", 33.4525039, -112.0688993, "mall_shopping_center", "Safeway, 340 E McDowell Rd, Phoenix, AZ"),
 ("86268c60fffffff", "First & Main Town Center", "3305 Cinema Point, Colorado Springs, CO 80922", 38.8809172, -104.716419, "mall_shopping_center", "Safeway, 2890 N Powers Blvd, Colorado Springs, CO"),
 ("862a100afffffff", "Concourse Plaza (24h, structured parking)", "200 E 161st St, Bronx, NY 10451", 40.8252165, -73.9202784, "mall_shopping_center", "Bronx Terminal Market, 610 Exterior St, Bronx, NY"),
 ("862ab2cc7ffffff", "Fairlane Town Center", "18900 Michigan Ave, Dearborn, MI 48126", 42.3168762, -83.2228274, "mall_shopping_center", "Westborn Mall, 23001 Michigan Ave, Dearborn, MI"),
 ("86441a597ffffff", "Walmart Supercenter (US-92)", "2120 US-92, Auburndale, FL 33823", 28.0600288, -81.8159512, "grocery_bigbox", "Olympia Plaza, 353 Havendale Blvd, Auburndale, FL"),
 ("8628f0137ffffff", "Oregon City Shopping Center", "1900 McLoughlin Blvd, Oregon City, OR 97045", 45.3688138, -122.6000253, "mall_shopping_center", "West Linn Central Village, 21900 Willamette Dr, West Linn, OR"),
 ("8626aa467ffffff", "Corner Mart Exxon (US-85, small host - parcel check needed)", "501 Railroad St, Gilcrest, CO 80623", 40.280857, -104.7793504, "fuel_travel_center", "Safeway, 4548 Centerplace Dr, Greeley, CO (8.8 km)"),
 ("8648b3847ffffff", "H-E-B (N 10th St)", "3200 N 10th St, McAllen, TX 78501", 26.232592, -98.223591, "grocery_bigbox", "Las Tiendas S.C., 724 E Expressway 83, McAllen, TX"),
 ("8629a970fffffff", "Shell (I-5 / Mercey Springs Rd, 24h)", "24729 Mercey Springs Rd, Los Banos, CA 93635", 36.929788, -120.841611, "fuel_travel_center", "Petro Travel Center, 28991 Gonzaga Rd, Santa Nella, CA"),
 # v2 (learned metro screen) new cells
 ("8629a0c47ffffff", "Boulder Crossing (new retail center, SR-79 / Domenigoni Pkwy)", "32873 Domenigoni Pkwy, Winchester, CA 92596", 33.6963129, -117.0875296, "mall_shopping_center", "Winchester Ranch Markets (gas), 28340 Winchester Rd, Winchester, CA"),
 ("862ab2ce7ffffff", "Merchant Row Commons (Livernois Ave)", "Livernois Ave at W Davison, Detroit, MI 48238", 42.3928529, -83.1405607, "mall_shopping_center", "Greenfield Center, 15601 Grand River Ave, Detroit, MI"),
 ("862a94d07ffffff", "Kroger (Glendale Ave)", "2555 Glendale Ave, Toledo, OH 43614", 41.6114115, -83.6063726, "grocery_bigbox", "Marathon, 1049 Western Ave, Toledo, OH (1.1 km from cell)"),
 ("86298614fffffff", "Pilot Travel Center (I-15 / Craig Rd, 24h)", "3812 E Craig Rd, North Las Vegas, NV 89031", 36.2409528, -115.0978424, "fuel_travel_center", "Nellis Trails Mall, 765 N Nellis Blvd, Las Vegas, NV"),
]


def hav(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, (a1, o1, a2, o2))
    return 2 * 6371 * np.arcsin(np.sqrt(np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2))


if __name__ == "__main__":
    sel = pd.read_csv(OUT / "selected_10_cells.csv")
    sc = pd.read_csv(PROC / "superchargers.csv")
    live = sc[sc.status.isin(["OPEN", "EXPANDING", "CONSTRUCTION", "PERMIT", "PLAN", "CLOSED_TEMP"])]
    h = pd.DataFrame(HOSTS, columns=["cell6", "host", "address", "host_lat", "host_lng", "host_type", "alternate_host"])
    m = sel.merge(h, on="cell6", how="left")
    m["host_to_cell_centre_km"] = hav(m.lat, m.lng, m.host_lat, m.host_lng)
    d = hav(m.host_lat.values[:, None], m.host_lng.values[:, None], live.lat.values[None], live.lng.values[None])
    m["host_to_nearest_tesla_km"] = d.min(axis=1)
    m["nearest_tesla_status"] = live.status.values[d.argmin(axis=1)]
    m.to_csv(OUT / "selected_10_sites_with_hosts.csv", index=False)
    print(m[["state", "host", "host_to_cell_centre_km", "host_to_nearest_tesla_km", "nearest_tesla_status", "stalls"]].round(2).to_string())
