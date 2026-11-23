"""ESA WorldCover 2021 (10 m, s3://esa-worldcover, public) resampled onto each S1 chip grid.

Writes <root>/terrain/<chip>_WorldCover.tif (uint8 class codes, 0 = no data)
and <root>/landcover_fractions.csv with per-chip class fractions.
"""
from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")

URL = "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{name}_Map.tif"
CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 70: "snow",
           80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}


def tile_names(west: float, south: float, east: float, north: float) -> list[str]:
    """WorldCover tiles are 3x3 degrees, named by their lower-left corner."""
    names = []
    for lat in range(math.floor(south / 3) * 3, math.floor(north / 3) * 3 + 1, 3):
        for lon in range(math.floor(west / 3) * 3, math.floor(east / 3) * 3 + 1, 3):
            ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
            names.append(ns + ew)
    return names


def build_one(s1_path: Path, out_dir: Path) -> dict:
    chip = s1_path.name.removesuffix("_S1Hand.tif")
    out = out_dir / f"{chip}_WorldCover.tif"
    with rasterio.open(s1_path) as s1:
        profile = s1.profile.copy()
        bounds = transform_bounds(s1.crs, "EPSG:4326", *s1.bounds)
    if out.exists():
        with rasterio.open(out) as src:
            lc = src.read(1)
    else:
        lc = np.zeros((profile["height"], profile["width"]), dtype=np.uint8)
        for name in tile_names(*bounds):
            try:
                src = rasterio.open(URL.format(name=name))
            except rasterio.errors.RasterioIOError:
                continue
            with src:
                part = np.zeros_like(lc)
                reproject(rasterio.band(src, 1), part, src_nodata=0, dst_transform=profile["transform"],
                          dst_crs=profile["crs"], dst_nodata=0, resampling=Resampling.nearest)
            lc = np.where(lc == 0, part, lc)
        profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(lc, 1)
    known = lc > 0
    row = {"chip": chip, "lc_nodata": float((~known).mean())}
    for code, name in CLASSES.items():
        row[f"lc_{name}"] = float((lc[known] == code).mean()) if known.any() else np.nan
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    s1_paths = sorted((args.root / "data/flood_events/HandLabeled/S1Hand").glob("*_S1Hand.tif"))
    out_dir = args.root / "terrain"
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(build_one, s1_paths, [out_dir] * len(s1_paths)))
    df = pd.DataFrame(rows)
    df.to_csv(args.root / "landcover_fractions.csv", index=False)
    print(df.describe().round(3).to_string())


if __name__ == "__main__":
    main()
