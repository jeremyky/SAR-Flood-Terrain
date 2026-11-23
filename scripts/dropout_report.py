"""Pre-registered outcomes for HAND modality dropout (PREREGISTRATIONS.md, 2026-11-21): sar_hand_md vs sar_hand vs sar_only.

    python scripts/dropout_report.py --runs $S/runs/geoid_lc/e20 --train-list $D/geoid_trainval/splits/train.txt \
        --out $S/reports/dropout_e20.csv

(1) mean: sar_hand_md - sar_hand (and each vs sar_only), event-mean and pooled IoU, two-way CIs, for
    flooded-only and all-water labels (external eval) and flood_diff under the GEOID-Flood protocol;
(2) tail: events where a HAND model is worse than SAR-only by > 0.02 event IoU (draw-averaged);
(3) EMSR280 (Sweden) all-water IoU per variant.
Unseen test activations only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hand_gate import compare, load  # noqa: E402
from learning_curve_report import training_events  # noqa: E402

VARIANTS = ("sar_only", "sar_hand", "sar_hand_md")
NAMES = {"geoid_test_flood": "flooded only", "geoid_test": "all water", "geoid_proto_test": "protocol flood_diff",
         "geoid_proto_heldout": "protocol flood_diff, held-out 2026"}


def per_event(d: pd.DataFrame) -> pd.DataFrame:
    ev = d.groupby(["variant", "seed", "event"])[["tp", "fp", "fn"]].sum()
    pos = (ev.tp + ev.fn).groupby("event").max() > 0
    ev = ev[ev.index.get_level_values("event").isin(pos[pos].index)]
    return (ev.tp / (ev.tp + ev.fp + ev.fn)).groupby(["variant", "event"]).mean().unstack("variant")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--boot", type=int, default=4000)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    seen = training_events(args.train_list)
    rng = np.random.default_rng(0)
    rows = []
    pd.set_option("display.width", 250)
    for name, label in NAMES.items():
        try:
            df = load(args.runs, name)
        except SystemExit:
            continue
        df = df[df.variant.isin(VARIANTS) & ~df.event.isin(seen) & df.metric.isin(["label", "flood_diff"])]
        if set(df.variant) != set(VARIANTS):
            print(f"{name}: missing variants {set(VARIANTS) - set(df.variant)}")
            continue
        for x, y in (("sar_hand_md", "sar_hand"), ("sar_hand", "sar_only"), ("sar_hand_md", "sar_only")):
            rows.append({"test": label, "comparison": f"{x} - {y}", **compare(df, x, y, rng, args.boot)})
        ev = per_event(df)
        for v in ("sar_hand", "sar_hand_md"):
            d = ev[v] - ev["sar_only"]
            rows.append({"test": label, "comparison": f"{v}: events worse than SAR only by > {args.margin}",
                         "events": int(d.notna().sum()), "improved": None, "worse": int((d < -args.margin).sum()),
                         "event_mean": float(d[d < -args.margin].sum()), "tail_events": " ".join(sorted(d[d < -args.margin].index))})
        if name == "geoid_test" and "EMSR280" in ev.index:
            print("EMSR280 (Sweden) all-water IoU:", ev.loc["EMSR280"].round(3).to_dict())
    table = pd.DataFrame(rows)
    print(f"{args.runs}: unseen activations; two-way 95% CIs (draws/seeds x events); tail rows: 'worse' = event count, "
          "'event_mean' = summed loss over those events")
    print(table.drop(columns=["tail_events"], errors="ignore").round(4).to_string(index=False))
    for r in rows:
        if r.get("tail_events"):
            print(f"  {r['test']} / {r['comparison']}: {r['tail_events']}")
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
