"""Build DEM, HAND and slope rasters aligned to each Sen1Floods11 S1 chip.

DEM:  Copernicus GLO-30 (s3://copernicus-dem-30m, public)
HAND: ASF global HAND derived from GLO-30 (s3://glo-30-hand, public)

Tiles are read remotely as COGs through GDAL's /vsicurl/ and bilinearly
resampled onto the chip grid. Slope is computed from the resampled DEM.
Outputs go to <root>/terrain/<chip>_{DEM,HAND,Slope}.tif (float32, NaN nodata).
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

DEM_URL = "/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{name}_DEM/{name}_DEM.tif"
HAND_URL = "/vsicurl/https://glo-30-hand.s3.amazonaws.com/v1/2021/{name}_HAND.tif"
EARTH_RADIUS = 6371008.8


def tile_name(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00"


def tiles_for_bounds(west: float, south: float, east: float, north: float) -> list[str]:
    return [
        tile_name(lat, lon)
        for lat in range(math.floor(south), math.floor(north) + 1)
        for lon in range(math.floor(west), math.floor(east) + 1)
    ]


def warp_tiles(url_template: str, tiles: list[str], dst_profile: dict) -> np.ndarray:
    out = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype=np.float32)
    for name in tiles:
        try:
            src = rasterio.open(url_template.format(name=name))
        except rasterio.errors.RasterioIOError:
            continue  # tile absent (open ocean); stays NaN
        with src:
            part = np.full_like(out, np.nan)
            reproject(
                source=rasterio.band(src, 1),
                destination=part,
                src_nodata=src.nodata,
                dst_transform=dst_profile["transform"],
                dst_crs=dst_profile["crs"],
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        out = np.where(np.isnan(out), part, out)
    return out


def slope_degrees(dem: np.ndarray, transform, crs) -> np.ndarray:
    """Slope from a DEM on a geographic (degree) or projected (metre) grid."""
    dx, dy = abs(transform.a), abs(transform.e)
    if crs.is_geographic:
        rows = np.arange(dem.shape[0]) + 0.5
        lat = np.deg2rad(transform.f + transform.e * rows)
        dy_m = np.deg2rad(dy) * EARTH_RADIUS
        dx_m = (np.deg2rad(dx) * EARTH_RADIUS * np.cos(lat))[:, None]
    else:
        dx_m, dy_m = dx, dy
    gy, gx = np.gradient(dem)
    return np.degrees(np.arctan(np.hypot(gx / dx_m, gy / dy_m))).astype(np.float32)


def build_one(s1_path: Path, out_dir: Path, overwrite: bool) -> str:
    chip = s1_path.name.removesuffix("_S1Hand.tif")
    outputs = {k: out_dir / f"{chip}_{k}.tif" for k in ("DEM", "HAND", "Slope")}
    if not overwrite and all(p.exists() for p in outputs.values()):
        return f"{chip}: exists"
    with rasterio.open(s1_path) as s1:
        profile = s1.profile.copy()
        bounds = transform_bounds(s1.crs, "EPSG:4326", *s1.bounds)
    tiles = tiles_for_bounds(*bounds)
    dem = warp_tiles(DEM_URL, tiles, profile)
    hand = warp_tiles(HAND_URL, tiles, profile)
    slope = slope_degrees(dem, profile["transform"], profile["crs"])
    profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3)
    for key, arr in (("DEM", dem), ("HAND", hand), ("Slope", slope)):
        with rasterio.open(outputs[key], "w", **profile) as dst:
            dst.write(arr, 1)
    nan_frac = float(np.isnan(dem).mean()), float(np.isnan(hand).mean())
    return f"{chip}: tiles={len(tiles)} nan(dem,hand)={nan_frac[0]:.3f},{nan_frac[1]:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Sen1Floods11 root (contains data/flood_events)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    s1_paths = sorted((args.root / "data/flood_events/HandLabeled/S1Hand").glob("*_S1Hand.tif"))[: args.limit]
    out_dir = args.root / "terrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for msg in pool.map(build_one, s1_paths, [out_dir] * len(s1_paths), [args.overwrite] * len(s1_paths)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
