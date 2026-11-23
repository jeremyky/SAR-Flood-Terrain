"""Which external events does HAND help? Correlate per-event HAND gain with event features.

Inputs: per-event IoU tables from scripts/external_report.py --out (all-water and flood-only),
the converted dataset roots (all-water labels and flood-only labels), WorldCover fractions
(scripts/build_landcover.py) and the GEOID-Flood tile catalog (location, country, year).

Features are event medians of per-chip values. Spearman correlations are reported with
Benjamini-Hochberg adjusted p-values across all features for each target.

    python scripts/external_explain.py --water-root .../geoid_test --flood-root .../geoid_test_flood \
        --water-events external_geoid_test_events.csv --flood-events external_geoid_test_flood_events.csv \
        --catalog .../tile_catalog.parquet --features-cache geoid_test_chip_features.csv
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from analyze_events import HAND_DIR, chip_features  # noqa: E402


def flood_fraction(root: Path, chip: str) -> float:
    with rasterio.open(root / HAND_DIR / "LabelHand" / f"{chip}_LabelHand.tif") as src:
        y = src.read(1)
    valid = y != -1
    return float((y[valid] == 1).mean()) if valid.any() else np.nan


def features(args) -> pd.DataFrame:
    if args.features_cache and args.features_cache.exists():
        return pd.read_csv(args.features_cache)
    chips = sorted(p.name.removesuffix("_S1Hand.tif") for p in (args.water_root / HAND_DIR / "S1Hand").glob("*_S1Hand.tif"))
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(chip_features, [args.water_root] * len(chips), chips, chunksize=16))
        flood = list(pool.map(flood_fraction, [args.flood_root] * len(chips), chips, chunksize=16))
    df = pd.DataFrame(rows).rename(columns={"flood_frac": "water_frac"})
    df["flood_only_frac"] = flood
    df["perm_water_frac"] = df.water_frac - df.flood_only_frac  # labelled permanent water
    lc = args.water_root / "landcover_fractions.csv"
    if lc.exists():
        df = df.merge(pd.read_csv(lc).drop(columns=["lc_nodata"]), on="chip", how="left")
    df["event"] = df.chip.str.rsplit("_", n=1).str[0]
    if args.features_cache:
        df.to_csv(args.features_cache, index=False)
    return df


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--water-root", type=Path, required=True)
    parser.add_argument("--flood-root", type=Path, required=True)
    parser.add_argument("--water-events", type=Path, required=True)
    parser.add_argument("--flood-events", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--features-cache", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    pd.set_option("display.width", 250, "display.max_rows", 200)

    chip = features(args)
    feat_cols = [c for c in chip.columns if c not in ("chip", "event")]
    ev = chip.groupby("event")[feat_cols].median()

    from shapely import wkb

    cat = pd.read_parquet(args.catalog)
    cat = cat[cat.event_id.isin(ev.index)]
    pts = cat.geometry.apply(lambda g: wkb.loads(g).centroid)
    cat = cat.assign(lat=pts.apply(lambda p: p.y), continent=cat.macro_areas_set.apply(lambda a: a[0] if len(a) else "unknown"))
    meta = cat.groupby("event_id").agg(
        abs_lat=("lat", lambda v: np.abs(v).median()),
        year=("event_time", lambda v: pd.to_datetime(v).dt.year.iloc[0]),
        country=("countries", "first"), continent=("continent", "first"), n_tiles=("tile_id", "count"),
    )
    meta["europe"] = (meta.continent == "Europe").astype(float)
    ev = ev.join(meta)
    num_cols = [c for c in ev.columns if pd.api.types.is_numeric_dtype(ev[c])]

    targets = {}
    for label, path in (("water", args.water_events), ("flood", args.flood_events)):
        t = pd.read_csv(path).pivot(index="event", columns="variant", values="iou")
        targets[f"d_{label}_unet"] = t.sar_hand - t.sar_only
        targets[f"d_{label}_r34"] = t.r34_sar_hand - t.r34_sar_only
        targets[f"base_{label}_unet"] = t.sar_only
    tg = pd.DataFrame(targets)
    joined = ev.join(tg, how="inner")
    if args.out:
        joined.to_csv(args.out)
    print(f"{len(joined)} events joined")

    print("\nConsistency of per-event HAND gain across backbones (Spearman, over events):")
    for label in ("water", "flood"):
        a, b = joined[f"d_{label}_unet"], joined[f"d_{label}_r34"]
        m = a.notna() & b.notna()
        rho, p = spearmanr(a[m], b[m])
        print(f"  {label}: rho={rho:+.2f} p={p:.2g} (n={m.sum()})")

    for target in ("d_flood_unet", "d_flood_r34", "d_water_unet"):
        rows = []
        for c in num_cols:
            m = joined[c].notna() & joined[target].notna()
            if m.sum() < 10 or joined.loc[m, c].nunique() < 3:
                continue
            rho, p = spearmanr(joined.loc[m, c], joined.loc[m, target])
            rows.append({"feature": c, "rho": rho, "p": p, "n": int(m.sum())})
        r = pd.DataFrame(rows)
        r["p_bh"] = bh(r.p)
        print(f"\nSpearman of {target} with event features (BH-adjusted over {len(r)} features):")
        print(r.sort_values("p").round(3).to_string(index=False))

    print("\nEvents per continent:", joined.continent.value_counts().to_dict())
    print("Median HAND gain on flooded water by continent:")
    print(joined.groupby("continent")[["d_flood_unet", "d_flood_r34"]].median().round(3).to_string())
    print("\nHAND gain on flooded water, Europe vs rest (Mann-Whitney):")
    for target in ("d_flood_unet", "d_flood_r34"):
        eu, rest = joined.loc[joined.europe == 1, target].dropna(), joined.loc[joined.europe == 0, target].dropna()
        print(f"  {target}: Europe median {eu.median():+.3f} (n={len(eu)}), rest {rest.median():+.3f} (n={len(rest)}), "
              f"p={mannwhitneyu(eu, rest).pvalue:.3f}")


if __name__ == "__main__":
    main()
