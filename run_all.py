"""Run the whole pipeline end to end.

    python run_all.py            # reuse cached backtest components if present
    python run_all.py --fresh    # recompute everything from the raw data

Steps
  1 src/decode_supercharge.py  raw supercharge.info export -> data/processed/superchargers_conus.csv
  2 src/prepare.py             clean counties, competitors (AFDC), roads, H3 keys
  3 src/network.py             H3 road graph, gravity OD flows, 7,000 shortest paths, candidate cells
  4 src/backtest.py            2016-2025 rolling-origin backtest + calibration
  5 src/select_sites.py        2026 screening, configuration columns, MILP + epsilon-constraint, forward check
Requirements: python>=3.10, pandas, numpy, scipy, scikit-learn, networkx, h3>=4
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RAW = ROOT / "data" / "raw"


def run(args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable] + args, cwd=SRC, check=True)


if __name__ == "__main__":
    fresh = "--fresh" in sys.argv
    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    parts = [str(RAW / f"sc_part{i}.txt") for i in range(1, 5)]
    run(["decode_supercharge.py", *parts, str(proc / "superchargers_conus.csv")])
    run(["prepare.py"])
    if fresh or not (proc / "paths.pkl").exists():
        run(["network.py"])
    if fresh:
        (proc / "backtest_components.pkl").unlink(missing_ok=True)
    run(["backtest.py"])
    run(["select_sites.py"])
    run(["hosts.py"])          # host properties found in due diligence (see EXCLUDE_CELLS in config.py)
    run(["report.py"])
    run(["sensitivity.py"])
    run(["make_map.py"])
