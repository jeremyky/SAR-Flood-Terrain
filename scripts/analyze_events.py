"""Where does a variant help or hurt relative to a baseline under leave-one-event-out?

Joins per-chip LOEO test results (chips_test.csv, averaged over seeds) with
per-chip terrain/label features computed from the rasters, then reports
per-event features next to the IoU change, and chip-level rank correlations.

    python scripts/analyze_events.py --runs /scratch/$USER/floodseg/runs \
        --data /scratch/$USER/floodseg/data/sen1floods11 --base sar_only --variant sar_hand
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import spearmanr

HAND_DIR = Path("data/flood_events/HandLabeled")


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def chip_features(root: Path, chip: str) -> dict:
    label = read(root / HAND_DIR / "LabelHand" / f"{chip}_LabelHand.tif")[0]
    s1 = read(root / HAND_DIR / "S1Hand" / f"{chip}_S1Hand.tif")
    jrc_path = root / HAND_DIR / "JRCWaterHand" / f"{chip}_JRCWaterHand.tif"  # absent for external datasets
    jrc = read(jrc_path)[0] if jrc_path.exists() else None
    hand = read(root / "terrain" / f"{chip}_HAND.tif")[0]
    slope = read(root / "terrain" / f"{chip}_Slope.tif")[0]
    dem = read(root / "terrain" / f"{chip}_DEM.tif")[0]
    valid = label != -1
    flood = label == 1
    return {
        "chip": chip,
        "flood_frac": float(flood[valid].mean()) if valid.any() else np.nan,
        "perm_water_frac": float((jrc == 1)[valid].mean()) if jrc is not None and valid.any() else np.nan,
        "hand_median": float(np.nanmedian(hand)),
        "low_hand_frac": float(np.nanmean(hand < 5)),  # share of chip within 5 m of drainage
        "hand_of_flood": float(np.nanmedian(hand[flood])) if flood.any() else np.nan,
        "slope_median": float(np.nanmedian(slope)),
        "relief": float(np.nanpercentile(dem, 95) - np.nanpercentile(dem, 5)),
        "vv_median": float(np.nanmedian(s1[0])),
        # how separable flood is in SAR alone: dB gap between dry and flooded VV
        "vv_contrast": float(np.nanmedian(s1[0][valid & ~flood]) - np.nanmedian(s1[0][flood])) if flood.any() else np.nan,
    }


def load_variant(runs: Path, variant: str) -> pd.DataFrame:
    frames = []
    for csv in runs.glob(f"loeo/*/{variant}/seed*/chips_test.csv"):
        df = pd.read_csv(csv)
        cm = np.array([json.loads(c) for c in df["cm"]])
        df[["tp", "fp", "fn", "tn"]] = cm
        df["seed"] = csv.parent.name
        frames.append(df[["chip", "event", "seed", "tp", "fp", "fn", "iou", "recall", "precision"]])
    df = pd.concat(frames)
    n_seeds = df.groupby("chip").seed.nunique()
    return df.groupby(["chip", "event"])[["tp", "fp", "fn", "iou", "recall", "precision"]].mean().reset_index(), int(n_seeds.min())


def pooled_iou(g: pd.DataFrame, suffix: str) -> float:
    tp, fp, fn = g[f"tp{suffix}"].sum(), g[f"fp{suffix}"].sum(), g[f"fn{suffix}"].sum()
    return tp / (tp + fp + fn) if tp + fp + fn else np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", default="sar_only")
    parser.add_argument("--variant", default="sar_hand")
    parser.add_argument("--features-cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="write the joined per-chip table here")
    args = parser.parse_args()

    base, n_base = load_variant(args.runs, args.base)
    var, n_var = load_variant(args.runs, args.variant)
    print(f"seeds per chip: {args.base}={n_base}, {args.variant}={n_var}")
    df = base.merge(var, on=["chip", "event"], suffixes=("_b", "_v"))

    cache = args.features_cache
    if cache and cache.exists():
        feats = pd.read_csv(cache)
    else:
        feats = pd.DataFrame(chip_features(args.data, c) for c in sorted(df.chip))
        if cache:
            feats.to_csv(cache, index=False)
    df = df.merge(feats, on="chip")
    landcover = args.data / "landcover_fractions.csv"  # from scripts/build_landcover.py
    lc_cols = []
    if landcover.exists():
        lc = pd.read_csv(landcover)
        lc_cols = [c for c in ("lc_tree", "lc_shrub", "lc_grass", "lc_crop", "lc_built", "lc_bare", "lc_water", "lc_wetland") if lc[c].max() > 0]
        df = df.merge(lc[["chip", *lc_cols]], on="chip")
    for k in ("tp", "fp", "fn"):
        df[f"d_{k}"] = df[f"{k}_v"] - df[f"{k}_b"]
    if args.out:
        df.to_csv(args.out, index=False)

    feature_cols = ["flood_frac", "perm_water_frac", "low_hand_frac", "hand_median", "hand_of_flood", "slope_median", "relief", "vv_contrast", *lc_cols]
    rows = []
    for event, g in df.groupby("event"):
        flood_px = g.tp_b.sum() + g.fn_b.sum()
        rows.append({
            "event": event, "n_chips": len(g),
            f"iou_{args.base}": pooled_iou(g, "_b"), f"iou_{args.variant}": pooled_iou(g, "_v"),
            "d_iou": pooled_iou(g, "_v") - pooled_iou(g, "_b"),
            # change in found flood / false alarms, as a share of the event's flood pixels
            "d_tp_pct": 100 * g.d_tp.sum() / flood_px, "d_fp_pct": 100 * g.d_fp.sum() / flood_px,
            **{c: g[c].median() for c in feature_cols},
        })
    events = pd.DataFrame(rows).sort_values("d_iou", ascending=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\nPer event (terrain/label features are chip medians):")
    print(events.round(3).to_string(index=False))

    print("\nEvent-level Spearman correlation of d_iou with features (n = %d events):" % len(events))
    for c in feature_cols:
        rho, p = spearmanr(events[c], events["d_iou"], nan_policy="omit")
        print(f"  {c:16s} rho={rho:+.2f} p={p:.3f}")

    flooded = df[(df.tp_b + df.fn_b) > 100].copy()  # chips with real flood; IoU is noise on near-empty chips
    flooded["d_iou_chip"] = flooded.iou_v - flooded.iou_b
    print("\nChip-level Spearman correlation of per-chip IoU change with features (n = %d flooded chips):" % len(flooded))
    for c in feature_cols:
        rho, p = spearmanr(flooded[c], flooded["d_iou_chip"], nan_policy="omit")
        print(f"  {c:16s} rho={rho:+.2f} p={p:.3f}")


if __name__ == "__main__":
    main()
