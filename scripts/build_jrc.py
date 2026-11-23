"""JRC Global Surface Water (Pekel et al. 2016, v1.4 2021) permanent-water layer aligned to each chip.

Sen1Floods11 ships a binary JRC layer (`JRCWaterHand`); GEOID-Flood does not. This builds the same kind of
layer for any root in Sen1Floods11 layout from the public GSW occurrence tiles (10 x 10 degree, 30 m,
read remotely through /vsicurl/), so `channels: [vv, vh, jrc]` works on GEOID-Flood too.

    # choose the occurrence threshold that best reproduces Sen1Floods11's own layer
    python scripts/build_jrc.py calibrate --root $D/sen1floods11
    # write <root>/data/flood_events/HandLabeled/JRCWaterHand/<chip>_JRCWaterHand.tif (uint8 0/1)
    python scripts/build_jrc.py build --root $D/geoid_trainval --threshold 50
"""
from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

GSW_URL = "/vsicurl/https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{name}v1_4_2021.tif"
HAND_DIR = Path("data/flood_events/HandLabeled")
GSW_NODATA = 255


def gsw_tiles(west: float, south: float, east: float, north: float) -> list[str]:
    """GSW tiles are named by their top-left corner, e.g. 70W_10S covers 70-60 W, 10-20 S."""
    names = []
    for top in range(math.ceil(north / 10) * 10, math.floor(south / 10) * 10, -10):
        for left in range(math.floor(west / 10) * 10, math.ceil(east / 10) * 10, 10):
            names.append(f"{abs(left)}{'W' if left < 0 else 'E'}_{abs(top)}{'S' if top < 0 else 'N'}")
    return names


def occurrence(s1_path: Path) -> tuple[np.ndarray, dict]:
    """GSW occurrence (0-100 %, NaN where GSW has no data) on the chip grid."""
    with rasterio.open(s1_path) as s1:
        profile = s1.profile.copy()
        bounds = transform_bounds(s1.crs, "EPSG:4326", *s1.bounds)
    out = np.full((profile["height"], profile["width"]), np.nan, np.float32)
    for name in gsw_tiles(*bounds):
        try:
            src = rasterio.open(GSW_URL.format(name=name))
        except rasterio.errors.RasterioIOError:
            continue  # no tile (open ocean / no land)
        with src:
            part = np.full_like(out, np.nan)
            reproject(source=rasterio.band(src, 1), destination=part, src_nodata=GSW_NODATA,
                      dst_transform=profile["transform"], dst_crs=profile["crs"], dst_nodata=np.nan,
                      resampling=Resampling.nearest)
        out = np.where(np.isnan(out), part, out)
    return out, profile


def chips(root: Path) -> list[Path]:
    return sorted((root / HAND_DIR / "S1Hand").glob("*_S1Hand.tif"))


def calibrate_one(s1_path: Path) -> np.ndarray:
    """Confusion counts vs the shipped JRC layer for every integer threshold 1..100 (occurrence >= t)."""
    occ, _ = occurrence(s1_path)
    chip = s1_path.name.removesuffix("_S1Hand.tif")
    with rasterio.open(s1_path.parent.parent / "JRCWaterHand" / f"{chip}_JRCWaterHand.tif") as src:
        ref = src.read(1) == 1
    occ = np.nan_to_num(occ, nan=0.0)
    t = np.arange(1, 101)[:, None]
    pred = occ.ravel()[None, :] >= t
    r = ref.ravel()[None, :]
    return np.stack([(pred & r).sum(1), (pred & ~r).sum(1), (~pred & r).sum(1)], 1)  # (100, 3): tp fp fn


def build_one(s1_path: Path, threshold: float, overwrite: bool) -> str:
    chip = s1_path.name.removesuffix("_S1Hand.tif")
    out = s1_path.parent.parent / "JRCWaterHand" / f"{chip}_JRCWaterHand.tif"
    if out.exists() and not overwrite:
        return f"{chip}: exists"
    occ, profile = occurrence(s1_path)
    water = (np.nan_to_num(occ, nan=0.0) >= threshold).astype(np.uint8)
    profile.update(count=1, dtype="uint8", nodata=None, compress="deflate", predictor=2)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(water, 1)
    return f"{chip}: water {water.mean():.3f} gsw-nodata {np.isnan(occ).mean():.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "build"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--workers", type=int, default=16)
        p.add_argument("--limit", type=int, default=None)
    sub.choices["build"].add_argument("--threshold", type=float, required=True, help="occurrence %% >= threshold is water")
    sub.choices["build"].add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = chips(args.root)[: args.limit]
    with ProcessPoolExecutor(args.workers) as pool:
        if args.cmd == "calibrate":
            cm = sum(pool.map(calibrate_one, paths))
            iou = cm[:, 0] / np.maximum(cm.sum(1), 1)
            best = int(iou.argmax())
            for t in sorted({1, 10, 25, 50, 75, 90, 99, best + 1}):
                print(f"occurrence >= {t:3d}%: IoU vs shipped JRC layer {iou[t - 1]:.3f}")
            print(f"best threshold {best + 1}% (IoU {iou[best]:.3f}) over {len(paths)} chips")
        else:
            (args.root / HAND_DIR / "JRCWaterHand").mkdir(exist_ok=True)
            for msg in pool.map(build_one, paths, [args.threshold] * len(paths), [args.overwrite] * len(paths)):
                print(msg, flush=True)


if __name__ == "__main__":
    main()
