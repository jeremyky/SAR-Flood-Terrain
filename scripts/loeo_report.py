"""Leave-one-event-out comparison: per-event IoU table and paired stats vs a baseline.

    python scripts/loeo_report.py --results results.csv [--pairs sar_hand:sar_only r34_sar_hand:r34_sar_only]
    python scripts/loeo_report.py --strata hand_strata.csv

`results.csv` comes from scripts/summarize.py --csv; `hand_strata.csv` from scripts/hand_strata.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def loeo_table(results: Path) -> pd.DataFrame:
    df = pd.read_csv(results)
    df = df[df.run.str.startswith("loeo/") & (df.split == "test") & (df.post == "raw")].copy()
    parts = df.run.str.split("/")
    df["event"], df["variant"] = parts.str[1], parts.str[2]
    return df


def paired(t: pd.DataFrame, variant: str, base: str, rng: np.random.Generator) -> dict:
    both = t[[variant, base]].dropna()
    d = (both[variant] - both[base]).to_numpy()
    boot = rng.choice(d, (10000, len(d))).mean(axis=1)
    return {
        "variant": variant, "baseline": base, "n_events": len(d),
        "mean_iou": both[variant].mean(), "base_iou": both[base].mean(), "delta": d.mean(),
        "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
        "wilcoxon_p": wilcoxon(d).pvalue if np.any(d) else 1.0, "improved": f"{(d > 0).sum()}/{len(d)}",
    }


def report_results(results: Path, pairs: list[str], seeds: list[str]) -> None:
    df = loeo_table(results)
    if seeds:
        df = df[df.seed.isin(seeds)]
    seeds = df.groupby("variant").seed.nunique()
    t = df.pivot_table(index="event", columns="variant", values="iou", aggfunc="mean")
    print("Seeds per variant:", seeds.to_dict())
    print("\nLOEO test IoU per held-out event (mean over seeds):")
    print(pd.concat([t, t.mean().to_frame("MEAN").T]).round(3).to_string())
    print("\nSeed std per event:")
    print(df.pivot_table(index="event", columns="variant", values="iou", aggfunc="std").round(3).to_string())

    rng = np.random.default_rng(0)
    if not pairs:
        pairs = [f"{v}:sar_only" for v in t.columns if not v.startswith("r34") and v != "sar_only"]
        pairs += [f"{v}:r34_sar_only" for v in t.columns if v.startswith("r34") and v != "r34_sar_only"]
        if "r34_sar_only" in t.columns:
            pairs.append("r34_sar_only:sar_only")
    rows = [paired(t, *p.split(":"), rng) for p in pairs if all(x in t.columns for x in p.split(":"))]
    print("\nPaired over events (bootstrap 95% CI of mean IoU delta, Wilcoxon signed-rank):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    for metric in ("recall", "precision", "dry_fp"):
        if metric in df:
            means = df.pivot_table(index="event", columns="variant", values=metric, aggfunc="mean").mean()
            print(f"\nMean over events, {metric}:", means.round(3).to_dict())


def report_strata(strata: Path) -> None:
    df = pd.read_csv(strata)
    order = ["0-1", "1-2", "2-5", "5-10", "10-20", ">20"]
    g = df.groupby(["variant", "event", "hand_bin"])[["tp", "fp", "fn", "tn"]].mean().reset_index()
    print("Seeds per variant:", df.groupby("variant").seed.nunique().to_dict())
    variants = sorted(g.variant.unique())
    if len(variants) != 2:
        print(g.to_string())
        return
    base, var = ("sar_only", "sar_hand") if {"sar_only", "sar_hand"} == set(variants) else variants
    w = g.pivot_table(index=["event", "hand_bin"], columns="variant", values=["tp", "fp", "fn"])
    flood = (w[("tp", base)] + w[("fn", base)]).groupby("event").sum()
    out = pd.DataFrame({
        "d_tp_pct": 100 * (w[("tp", var)] - w[("tp", base)]),
        "d_fp_pct": 100 * (w[("fp", var)] - w[("fp", base)]),
        "flood_share_pct": 100 * (w[("tp", base)] + w[("fn", base)]),
    })
    out = out.div(flood, level="event", axis=0).reset_index()
    out["hand_bin"] = pd.Categorical(out.hand_bin, order, ordered=True)
    print(f"\n{var} minus {base}, per HAND bin, as % of the event's flood pixels")
    print("(flood_share_pct = where the event's true flood sits)")
    for col in ("flood_share_pct", "d_tp_pct", "d_fp_pct"):
        print(f"\n{col}:")
        print(out.pivot_table(index="event", columns="hand_bin", values=col, observed=False).round(1).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path)
    parser.add_argument("--pairs", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="*", default=[], help="restrict to these seeds, e.g. seed0 seed1 seed2")
    parser.add_argument("--strata", type=Path)
    args = parser.parse_args()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    if args.results:
        report_results(args.results, args.pairs, args.seeds)
    if args.strata:
        report_strata(args.strata)


if __name__ == "__main__":
    main()
