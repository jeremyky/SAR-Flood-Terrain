"""Do the reference labels look HAND-masked? (PREREGISTRATIONS.md #13; descriptive)

If a labelling workflow removed water above a HAND threshold T, labelled water would stop at T. Per event and label
set, from per-pixel histograms of HAND (0.25-m bins) over valid labelled pixels and over labelled water:
  share of valid area and of labelled water with HAND > T, for T in 5, 10, 15, 20 m;
  a cut-off test at each T: water fraction of the valid pixels in [T-1, T) and in [T, T+1). An event is flagged at T
  when the lower band holds >= 1,000 water pixels, the upper band >= 1,000 valid pixels, and the upper band no water.
A mask computed from another HAND product would not cut sharply in ours, so a flag is a strong sign and its absence a
weak one; label water well above T in an event rules out a mask at T for that event.

    python scripts/label_hand_check.py --root $S/data/geoid_test --root $S/data/geoid_test_flood \
        --root $S/data/sen1floods11 --out $S/reports/label_hand_check.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

THRESHOLDS = (5, 10, 15, 20)
EDGES = np.arange(0, 200.25, 0.25)


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def histograms(root: Path) -> pd.DataFrame:
    labels = sorted((root / "data/flood_events/HandLabeled/LabelHand").glob("*_LabelHand.tif"))
    acc: dict[str, list[np.ndarray]] = {}
    for path in labels:
        chip = path.name.removesuffix("_LabelHand.tif")
        hand_path = root / "terrain" / f"{chip}_HAND.tif"
        if not hand_path.exists():
            continue
        y, h = read(path), read(hand_path)
        assert y.shape == h.shape, chip
        valid = (y >= 0) & np.isfinite(h)
        hv = np.clip(h[valid], 0, EDGES[-1] - 1e-3)
        a = np.histogram(hv, EDGES)[0]
        w = np.histogram(np.clip(h[valid & (y == 1)], 0, EDGES[-1] - 1e-3), EDGES)[0]
        event = chip.rsplit("_", 1)[0] if root.name.startswith("sen1") else chip.split("_")[0]
        if event in acc:
            acc[event][0] += a
            acc[event][1] += w
        else:
            acc[event] = [a, w]
    rows = []
    for event, (a, w) in sorted(acc.items()):
        r = {"labels": root.name, "event": event, "valid_px": int(a.sum()), "water_px": int(w.sum())}
        for t in THRESHOLDS:
            above = EDGES[:-1] >= t
            lo = (EDGES[:-1] >= t - 1) & (EDGES[:-1] < t)
            hi = (EDGES[:-1] >= t) & (EDGES[:-1] < t + 1)
            r[f"area_above_{t}"] = a[above].sum() / max(a.sum(), 1)
            r[f"water_above_{t}"] = w[above].sum() / max(w.sum(), 1)
            r[f"wfrac_below_{t}"] = w[lo].sum() / max(a[lo].sum(), 1)
            r[f"wfrac_above_{t}"] = w[hi].sum() / max(a[hi].sum(), 1)
            r[f"flag_{t}"] = bool(w[lo].sum() >= 1000 and a[hi].sum() >= 1000 and w[hi].sum() == 0)
        cum = np.cumsum(w) / max(w.sum(), 1)
        r["water_hand_p99"] = EDGES[1:][np.searchsorted(cum, 0.99)] if w.sum() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    df = pd.concat([histograms(r) for r in args.root])
    df = df[df.water_px > 0]
    pd.set_option("display.width", 250)
    for labels, g in df.groupby("labels", sort=False):
        print(f"\n== {labels}: {len(g)} events with labelled water")
        for t in THRESHOLDS:
            some_area = g[f"area_above_{t}"] >= 0.01
            print(f"  T = {t:2d} m: events with >= 1% of area above T: {some_area.sum():3d}; of these, with no labelled "
                  f"water above T: {(some_area & (g[f'water_above_{t}'] == 0)).sum():3d}; "
                  f"median share of water above T {g[f'water_above_{t}'].median():.4f}; cut-off flags: {g[f'flag_{t}'].sum()}")
        print(f"  99th percentile of labelled-water HAND per event: median {g.water_hand_p99.median():.1f} m, "
              f"range {g.water_hand_p99.min():.1f}-{g.water_hand_p99.max():.1f} m")
        flagged = g[g[[f"flag_{t}" for t in THRESHOLDS]].any(axis=1)]
        if len(flagged):
            print("  flagged events:", ", ".join(f"{e} ({', '.join(str(t) for t in THRESHOLDS if r[f'flag_{t}'])} m)"
                                               for e, (_, r) in zip(flagged.event, flagged.iterrows())))
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
