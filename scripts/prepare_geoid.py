"""Convert GEOID-Flood tiles (Bosco et al., links-ads/geoid-flood, CC BY 4.0) to the Sen1Floods11 layout.

Uses the post-event Sentinel-1 GRD image only (single-image setting, like Sen1Floods11):
linear sigma0 -> dB. Labels: 0 background -> 0, 1 permanent water and 2 flooded water -> 1,
255 unmapped -> -1. Tiles follow the paper's selection (is_valid and invalid_pixel_frac <= 0.95).

Chip ids are <event>_<rest of tile id> (e.g. EMSR712_10-0) so floodseg.data.Chip.event is the
CEMS activation. Afterwards run scripts/build_terrain.py --root <out> for DEM/HAND/slope.

    python scripts/prepare_geoid.py --geoid /scratch/$USER/floodseg/geoid_data/geoid-flood \
        --catalog tile_catalog.parquet --split test --out /scratch/$USER/floodseg/data/geoid_test \
        --exclude EMSR388
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

HAND_DIR = Path("data/flood_events/HandLabeled")


def convert(tile_id: str, event: str, src_root: Path, out_root: Path, label_mode: str = "water", image: str = "post") -> str:
    # files are <event>-<aoi>-<bbox>_<layer>.tif inside <event>-<aoi>/<layer>/
    event_dir = src_root / tile_id.rsplit("-", 1)[0]
    posts = sorted((event_dir / "s1grd").glob(f"{tile_id}_s1grd_post_*.tif"))
    label_path = event_dir / "label" / f"{tile_id}_label.tif"
    if not posts or not label_path.exists():
        return f"{tile_id}: missing"
    if image == "pre":  # latest pre-event acquisition (closest to the event); file names sort by date
        pres = sorted((event_dir / "s1grd").glob(f"{tile_id}_s1grd_pre_*.tif"))
        if not pres:
            return f"{tile_id}: missing"
        posts = pres[-1:]
    chip = f"{event}_{tile_id.removeprefix(event + '-')}"
    with rasterio.open(posts[0]) as src:
        s1, profile = src.read().astype(np.float32), src.profile.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = np.where(s1 > 0, 10 * np.log10(s1), np.nan).astype(np.float32)
    with rasterio.open(label_path) as src:
        raw = src.read(1)
    label = np.full(raw.shape, -1, dtype=np.int16)
    label[raw == 0] = 0
    if label_mode == "water":  # any water, like Sen1Floods11
        label[(raw == 1) | (raw == 2)] = 1
    elif label_mode == "flood":  # flooded water only; permanent water is ignored
        label[raw == 2] = 1
    elif label_mode == "flood_vs_rest":  # flooded water vs everything else; permanent water is a negative (GEOID protocol)
        label[raw == 1] = 0
        label[raw == 2] = 1
    else:  # "raw": keep the three classes (1 permanent water, 2 flooded water) for protocol-matched scoring
        label[raw == 1] = 1
        label[raw == 2] = 2

    profile.update(driver="GTiff", dtype="float32", count=2, nodata=np.nan, compress="deflate", predictor=3)
    s1_out = out_root / HAND_DIR / "S1Hand" / f"{chip}_S1Hand.tif"
    if not s1_out.exists():  # S1 is independent of the label mode; skip when already written or linked
        with rasterio.open(s1_out, "w", **profile) as dst:
            dst.write(s1)
    profile.update(dtype="int16", count=1, nodata=-1, predictor=2)
    with rasterio.open(out_root / HAND_DIR / "LabelHand" / f"{chip}_LabelHand.tif", "w", **profile) as dst:
        dst.write(label, 1)
    return f"{tile_id}: ok {chip}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geoid", type=Path, required=True, help="unpacked tree, e.g. geoid_data/geoid-flood")
    parser.add_argument("--catalog", type=Path, required=True, help="tile_catalog.parquet for that tree")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", default=[], help="event ids to drop (overlap with Sen1Floods11)")
    parser.add_argument("--label-mode", choices=["water", "flood", "flood_vs_rest", "raw"], default="water",
                        help="water: permanent + flooded water = 1 (Sen1Floods11 convention); flood: flooded only, permanent "
                             "ignored; raw: 0 background, 1 permanent, 2 flooded (scripts/eval_geoid_protocol.py)")
    parser.add_argument("--image", choices=["post", "pre"], default="post",
                        help="which S1 acquisition to write (pre: latest pre-event image, for pre/post differencing)")
    parser.add_argument("--split-name", default=None, help="also write chip ids to <out>/splits/<name>.txt (e.g. train, valid)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cat = pd.read_parquet(args.catalog)
    cat = cat[(cat.split == args.split) & cat.is_valid & (cat.invalid_pixel_frac <= 0.95) & ~cat.event_id.isin(args.exclude)]
    for sub in ("S1Hand", "LabelHand"):
        (args.out / HAND_DIR / sub).mkdir(parents=True, exist_ok=True)
    print(f"{len(cat)} tiles, {cat.event_id.nunique()} events")
    with ProcessPoolExecutor(args.workers) as pool:
        names = cat.event_id + "-" + cat.tile_id.astype(str) + "-" + cat.bbox_id.astype(str)
        msgs = list(pool.map(convert, names, cat.event_id, [args.geoid] * len(cat), [args.out] * len(cat), [args.label_mode] * len(cat),
                                [args.image] * len(cat)))
    missing = [m for m in msgs if m.endswith("missing")]
    if args.split_name:
        (args.out / "splits").mkdir(exist_ok=True)
        chips = [m.split(" ok ")[1] for m in msgs if " ok " in m]
        (args.out / "splits" / f"{args.split_name}.txt").write_text("\n".join(chips) + "\n")
    print(f"converted {len(msgs) - len(missing)}, missing {len(missing)}")
    for m in missing[:10]:
        print(" ", m)


if __name__ == "__main__":
    main()
