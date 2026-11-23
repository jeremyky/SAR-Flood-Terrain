"""Collect metrics.json files into one table: mean ± std over seeds per (run, split, post-processing).

    python scripts/summarize.py /scratch/$USER/floodseg/runs [--csv out.csv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def rows(runs_root: Path):
    for path in runs_root.rglob("metrics.json"):
        run = path.parent.parent.relative_to(runs_root).as_posix()
        seed = path.parent.name
        m = json.loads(path.read_text())
        for split, variants in m["splits"].items():
            for post, s in variants.items():
                yield {
                    "run": run, "seed": seed, "split": split, "post": post,
                    "iou": s["total"]["iou"], "f1": s["total"]["f1"],
                    "precision": s["total"]["precision"], "recall": s["total"]["recall"],
                    "chip_iou": s["chip_mean_iou"],
                    "dry_fp": s["dry_chip_false_positive_rate"],
                    **{f"phys_{k}": v for k, v in s["physics"].items()},
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    df = pd.DataFrame(rows(args.runs_root))
    if df.empty:
        raise SystemExit("no metrics.json found")
    if args.csv:
        df.to_csv(args.csv, index=False)
    # leave-one-event-out runs are named loeo/<Event>/<variant>: pool them per variant
    loeo = df["run"].str.startswith("loeo/")
    df.loc[loeo, "event"] = df.loc[loeo, "run"].str.split("/").str[1]
    df.loc[loeo, "run"] = "loeo/" + df.loc[loeo, "run"].str.split("/").str[2]
    metrics = ["iou", "chip_iou", "f1", "dry_fp", "phys_high_hand_flood_fraction", "phys_monotonicity_violation_rate"]
    table = df.groupby(["split", "run", "post"])[metrics].agg(["mean", "std"]).round(3)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(table)
    if loeo.any():
        per_event = df[loeo & (df["post"] == "raw")].pivot_table(index="event", columns="run", values="iou", aggfunc="mean").round(3)
        print("\nLeave-one-event-out test IoU per held-out event (mean over seeds):")
        print(per_event)


if __name__ == "__main__":
    main()
