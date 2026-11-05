"""Central parameters and assumptions for the Supercharger siting model.

Everything here that is NOT read from data is an explicit assumption. They are grouped so
they can be changed in one place and re-run (python run_all.py).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
for p in (PROC, OUT):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- H3 spatial base
H3_NODE_RES = 8        # road-graph nodes (~0.5 km edge): snaps road vertices into one network
H3_CAND_RES = 6        # candidate sites / path arcs (~3.7 km edge, ~36 km2)
H3_OD_RES = 4          # OD aggregation of county population (~22 km edge)
H3_CV_RES = 3          # spatial cross-validation blocks (~60 km edge)
COMP_GRID_DEG = 0.3    # competitor DCFC data arrives aggregated on a 0.3 degree grid

# ---------------------------------------------------------------- vehicle / range (assumptions)
RANGE_NOMINAL_KM = 400             # typical EPA range of the fleet
RANGE_USABLE_KM = 320              # 10%->90% usable window: a gap longer than this = route not drivable
RANGE_COMFORT_KM = 200             # comfort spacing (winter, highway speed, towing); calibrated in backtest
DETOUR_RING = 1                    # stations within 1 ring (~4-7 km) of the route count as "on route"
TRIP_MIN_KM, TRIP_MAX_KM = 150, 1100
KWH_PER_KM = 0.19

# ---------------------------------------------------------------- local demand (assumptions)
LOCAL_RADIUS_RING = 9              # Huff catchment ~ 9 rings of res-6 cells (~35 km)
PUBLIC_DCFC_KWH_PER_EV_YEAR = 420  # avg public fast-charging energy per EV per year (around home metro)
CORRIDOR_TRIPS_PER_EV_YEAR = 4     # long trips (>150 km) per EV per year that need en-route charging
DETOUR_FACTOR = 1.25               # straight-line -> road distance
HUFF_BETA = 2.0                    # distance decay exponent; calibrated in backtest
HUFF_GAMMA_STALLS = 0.6            # MCI elasticity on stall count
ATTR_SAT_STALLS = 16               # attractiveness saturates above this size
OUTSIDE_OPTION = 1.5               # attractiveness of "charge at home / elsewhere / don't charge"
COMP_ATTR_MULT = 0.55              # a competitor effective port vs a Tesla stall (reliability, NACS adapters)
FACILITY_MULT = {                  # MCI commercial-signal multipliers by host type (prior, not fitted)
    "fuel_travel_center": 1.10, "grocery_bigbox": 1.00, "mall_shopping_center": 1.00,
    "restaurant": 1.05, "hotel_casino": 0.85, "other": 0.90, "unknown": 0.95,
}
STALL_KWH_CAP_DAY = 650            # practical kWh/stall/day before queues build (V3/V4 mix)
CONGESTION_KAPPA = 1.2             # attractiveness penalty per unit of over-utilisation

# EV stock index by year (2023 = 1.0, AFDC registrations). Shapes total demand over the backtest.
EV_INDEX = {2015: 0.11, 2016: 0.15, 2017: 0.20, 2018: 0.30, 2019: 0.40, 2020: 0.50, 2021: 0.65,
            2022: 0.85, 2023: 1.00, 2024: 1.25, 2025: 1.50, 2026: 1.75, 2027: 2.0}

# ---------------------------------------------------------------- backtest
BACKTEST_YEARS = list(range(2016, 2026))
CALIB_YEARS = [y for y in BACKTEST_YEARS if y <= 2021]
TEST_YEARS = [y for y in BACKTEST_YEARS if y >= 2022]
HIT_RADIUS_KM = 15
TOPK = 50
NMS_KM = 25                         # non-max suppression between picks in a ranked list

# ---------------------------------------------------------------- economics (assumptions, USD)
PRICE_PER_KWH = 0.45                # blended Supercharger retail price
ENERGY_COST_PER_KWH = 0.14          # energy charge
DEMAND_CHARGE_PER_KW_MONTH = 16.0
OTHER_OPEX_PER_STALL_YEAR = 2500
CAPEX_PER_STALL = 45_000            # V4 cabinet + post, installed
SITE_FIXED_CAPEX = 250_000          # civil works, transformer pad, design
MEGAPACK_UNIT_KWH, MEGAPACK_UNIT_KW, MEGAPACK_UNIT_COST = 3900, 1900, 1_000_000
L2_UNIT_COST = 6_000
DISCOUNT_RATE, LIFE_YEARS = 0.09, 12
STALL_KW_DESIGN = 250               # design draw per active stall (V4 cabinet sharing)
COINCIDENCE = 0.6                   # site peak / (stalls * stall kW): V4 cabinets share power

# ---------------------------------------------------------------- grid / flexible interconnection (assumptions)
# Feeder headroom by area type (kW) and lead times (months). No national hosting-capacity data is public,
# so these are scenario parameters, not observations; replace with utility ICA maps where available.
AREA_HEADROOM_KW = {"urban": 2500, "suburban": 3500, "rural": 1500}
FIRM_LEADTIME_MONTHS = {"urban": 24, "suburban": 18, "rural": 30}
FLEX_LEADTIME_MONTHS = 8
FIRM_UPGRADE_COST_PER_KW = 350      # cost of upgrading above headroom (USD per kW of deficit)
INTERCONNECT_OPTIONS = ["firm", "static_flex", "dynamic_flex", "bridge"]
FLEX_FLOOR_SHARE = 0.4              # dynamic envelope never below 40% of request
STATES_WITH_FLEX_PROGRAMS = {"CA", "NY", "MA", "IL", "MN", "CO", "NJ", "MD", "TX"}  # assumption, verify per utility
LOST_SESSION_PENALTY = 0.25         # brand / experience cost per curtailed kWh on top of lost margin

# regional retail price / energy cost (USD/kWh) - assumption, replace with tariff data
REGION_ECON = {"CA": (0.56, 0.30), "NE": (0.52, 0.24), "default": (0.45, 0.14)}
NE_STATES = {"NY", "NJ", "MA", "CT", "RI", "NH", "VT", "ME"}

# ---------------------------------------------------------------- portfolio
N_SITES = 10
MIN_DIST_EXISTING_KM = 6            # recommendations must be genuinely new locations
MIN_DIST_BETWEEN_PICKS_KM = 40
SHORTLIST = 160
METRO_SCREEN = "learned"            # "learned" = rolling-validated logistic screen (v2); "pot" = demand potential (v1)
SHORTLIST_STATE_CAP = {"metro": 8, "corridor": 6}
MAX_SITES_PER_STATE = 3             # portfolio concentration limit (utility / regulatory risk)
STALL_OPTIONS = (8, 12, 16, 20, 24, 32, 40, 48)
PROFIT_WEIGHT = 2 / 3               # economic objective dominates; corridor flow gets 1/3 (Pareto pick rule)

# ---------------------------------------------------------------- due-diligence exclusions (step 6 feedback loop)
# Cells rejected after host search; the master problem is re-solved without them.
EXCLUDE_CELLS = {
    "86298614fffffff": "North Las Vegas: cell centre on/next to Nellis AFB, no commercial host in cell; "
                       "nearest viable host (Pilot, E Craig Rd) is 1.6 km from an existing Supercharger",
}
