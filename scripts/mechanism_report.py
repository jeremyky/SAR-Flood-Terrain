"""Does the HAND gain depend on HAND being the right HAND for the place? (pre-registered, PREREGISTRATIONS.md #6)

Models trained on the 11 Sen1Floods11 events, scored on all 77/79 GEOID-Flood test events:
  test-time perturbations of the HAND input of the trained SAR + HAND models (external_<name>_h<perturb>):
    shift10 (100 m misregistration), shift50 (500 m), blur25 (Gaussian, sigma 250 m), swap (HAND of a tile
    from another event);
  a control trained and tested with swapped HAND (variant sar_hand_swap).
Each is compared with SAR only and with the unperturbed SAR + HAND model: event-mean and pooled IoU, two-way
95% CIs over seeds and events (hand_gate.compare).

    python scripts/mechanism_report.py --runs $S/runs/all --out $S/reports/mechanism.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hand_gate import compare, load  # noqa: E402

PERTURB = ["shift10", "shift50", "blur25", "swap"]
NAMES = {"geoid_test_flood": "flooded only", "geoid_test": "all water"}


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
        base = load(args.runs, name, args.seeds)
        base = base[base.variant.isin(["sar_only", "sar_hand", "sar_hand_swap"])]
        frames = [base]
        for k in PERTURB:
            d = load(args.runs, f"{name}_h{k}", args.seeds)
            d = d[d.variant == "sar_hand"].assign(variant=f"sar_hand@{k}")
            frames.append(d)
        df = pd.concat(frames)
        df = df[df.metric == "label"]
        pairs = [("sar_hand", "sar_only")]
        pairs += [(f"sar_hand@{k}", "sar_only") for k in PERTURB] + [(f"sar_hand@{k}", "sar_hand") for k in PERTURB]
        pairs += [("sar_hand_swap", "sar_only"), ("sar_hand_swap", "sar_hand")]
        for x, y in pairs:
            if {x, y} <= set(df.variant):
                rows.append({"test": label, "comparison": f"{x} - {y}", **compare(df, x, y, rng, args.boot)})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"{args.runs}: all GEOID test events; two-way 95% CIs (seeds x events)")
    print(table.round(4).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
