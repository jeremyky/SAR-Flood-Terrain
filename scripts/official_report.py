"""Sen1Floods11 official split (test tiles share events with training) and its Bolivia hold-out (an event
absent from training), for comparability with published results. Pooled IoU over the split's chips (the
usual Sen1Floods11 metric), per seed, and the paired difference to SAR only; mean ± sd over seeds.

    python scripts/official_report.py --runs $S/runs --out $S/reports/official_split.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

VARIANTS = {"sar_only": "SAR only", "sar_hand": "+ HAND", "sar_terrain": "+ HAND, DEM, slope"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    rows = []
    for v in VARIANTS:
        for m in sorted((args.runs / v).glob("seed*/metrics.json")):
            s = json.loads(m.read_text())["splits"]
            for split in ("test", "bolivia"):
                rows.append({"variant": v, "seed": m.parent.name, "split": split, "iou": s[split]["raw"]["total"]["iou"]})
    d = pd.DataFrame(rows).pivot_table(index=["split", "seed"], columns="variant", values="iou")
    out = []
    for split, g in d.groupby(level="split"):
        for v in VARIANTS:
            r = {"split": split, "variant": v, "seeds": g[v].notna().sum(), "iou_mean": g[v].mean(), "iou_sd": g[v].std()}
            if v != "sar_only":
                delta = g[v] - g["sar_only"]
                r.update(delta_mean=delta.mean(), delta_sd=delta.std(), delta_by_seed=" ".join(f"{x:+.3f}" for x in delta))
            out.append(r)
    table = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print("Sen1Floods11 official split: pooled IoU (raw predictions), mean ± sd over seeds; delta paired by seed")
    print(table.round(4).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
