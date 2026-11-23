"""Per-event comparison of variants on an external test set (output of scripts/eval_external.py).

Pools confusion counts per event (summed over the event's chips), averages over seeds, and
compares each variant with its baseline across events: mean delta, bootstrap 95% CI, Wilcoxon
signed-rank, and the share of events improved. Events without any labelled flood are dropped.

    python scripts/external_report.py --runs /scratch/$USER/floodseg/runs/all --name geoid_test \
        --pairs sar_hand:sar_only r34_sar_hand:r34_sar_only r34_sar_only:sar_only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def load(runs: Path, name: str) -> pd.DataFrame:
    frames = []
    for csv in runs.glob(f"*/seed*/external_{name}/chips.csv"):
        df = pd.read_csv(csv)
        df[["tp", "fp", "fn", "tn"]] = np.array([json.loads(c) for c in df.cm])
        df["variant"], df["seed"] = csv.parent.parent.parent.name, csv.parent.parent.name
        frames.append(df)
    return pd.concat(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pairs", nargs="+", default=["sar_hand:sar_only", "r34_sar_hand:r34_sar_only", "r34_sar_only:sar_only"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    pd.set_option("display.width", 250, "display.max_rows", 200)

    chips = load(args.runs, args.name)
    print("seeds per variant:", chips.groupby("variant").seed.nunique().to_dict())
    ev = chips.groupby(["variant", "seed", "event"])[["tp", "fp", "fn", "tn"]].sum().reset_index()
    ev = ev[(ev.tp + ev.fn) > 0]
    ev["iou"] = ev.tp / (ev.tp + ev.fp + ev.fn)
    ev["recall"] = ev.tp / (ev.tp + ev.fn)
    ev["precision"] = ev.tp / (ev.tp + ev.fp).replace(0, np.nan)
    per_event = ev.groupby(["event", "variant"])[["iou", "recall", "precision"]].mean().reset_index()
    t = per_event.pivot(index="event", columns="variant", values="iou")
    if args.out:
        per_event.to_csv(args.out, index=False)

    pooled = chips.groupby(["variant", "seed"])[["tp", "fp", "fn"]].sum()
    pooled["iou"] = pooled.tp / (pooled.tp + pooled.fp + pooled.fn)
    print(f"\n{t.shape[0]} events with flood. Mean over events (event IoU, seeds averaged) and all-pixel pooled IoU:")
    summary = pd.DataFrame({
        "event_mean_iou": t.mean(),
        "event_median_iou": t.median(),
        "pooled_iou": pooled.groupby("variant").iou.mean(),
        "pooled_iou_seed_std": pooled.groupby("variant").iou.std(),
        "recall": per_event.pivot(index="event", columns="variant", values="recall").mean(),
        "precision": per_event.pivot(index="event", columns="variant", values="precision").mean(),
    })
    print(summary.round(3).to_string())

    rng = np.random.default_rng(0)
    rows = []
    for pair in args.pairs:
        var, base = pair.split(":")
        if var not in t or base not in t:
            continue
        d = (t[var] - t[base]).dropna().to_numpy()
        boot = rng.choice(d, (10000, len(d))).mean(axis=1)
        rows.append({
            "variant": var, "baseline": base, "n_events": len(d), "delta_mean": d.mean(), "delta_median": np.median(d),
            "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
            "wilcoxon_p": wilcoxon(d).pvalue, "improved": f"{(d > 0).sum()}/{len(d)}",
        })
    print("\nPaired over events (bootstrap 95% CI of mean event-IoU delta, Wilcoxon signed-rank):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    if "sar_hand" in t and "sar_only" in t:
        d = (t["sar_hand"] - t["sar_only"]).sort_values()
        print("\nLargest sar_hand losses / gains by event:")
        print(pd.concat([d.head(8), d.tail(8)]).round(3).to_string())


if __name__ == "__main__":
    main()
