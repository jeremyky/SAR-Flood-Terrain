"""Terrain ablation on the Sen1Floods11 -> GEOID-Flood transfer (pre-registered; PREREGISTRATIONS.md, 2026-11-21).

    python scripts/ablation_all_report.py --runs $S/runs/all --out $S/reports/ablation_all.csv

All 77/79 GEOID test events (none is in Sen1Floods11 training once EMSR388 is excluded), two-way CIs over
seeds and test events, event-mean and pooled IoU: flood-only and all-water labels, and IoU_flood under the
GEOID-Flood protocol (test and held-out 2026).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hand_gate import compare, load  # noqa: E402

PAIRS = [("sar_dem_slope", "sar_only"), ("sar_hand", "sar_only"), ("sar_hand", "sar_dem_slope"),
         ("sar_terrain", "sar_only"), ("sar_terrain", "sar_hand"),
         ("fm_sar_hand", "fm_sar"), ("fm_sar", "sar_only"), ("sar_hand", "fm_sar")]
NAMES = {"geoid_test_flood": "flooded only", "geoid_test": "all water", "geoid_proto_test": "protocol flood_diff",
         "geoid_proto_heldout": "protocol flood_diff, held-out 2026"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--boot", type=int, default=4000)
    p.add_argument("--out", type=Path)
    p.add_argument("--seeds", nargs="+", help="seed dirs to use (default: all); the registered tables use seed0 seed1 seed2")
    args = p.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for name, label in NAMES.items():
        df = load(args.runs, name, args.seeds)
        df = df[df.metric.isin(["label", "flood_diff"])]
        for x, y in PAIRS:
            if {x, y} <= set(df.variant):
                rows.append({"test": label, "comparison": f"{x} - {y}", **compare(df, x, y, rng, args.boot)})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"{args.runs}: all GEOID test events; two-way 95% CIs (seeds x events)")
    print(table.round(4).to_string(index=False))
    ev = table[table.test == "flooded only"].set_index("comparison").event_mean
    if {"sar_dem_slope - sar_only", "sar_hand - sar_only"} <= set(ev.index):
        print(f"\nshare of HAND's flood event-mean gain recovered by DEM + slope: "
              f"{ev['sar_dem_slope - sar_only'] / ev['sar_hand - sar_only']:.2f}")
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
