"""Qualitative figure: SAR, HAND, label and the errors of a SAR-only and a SAR + HAND model on chosen tiles.

    python scripts/example_tiles.py --data-root /scratch/$USER/floodseg/data/geoid_test \
        --base-run .../runs/all/sar_only/seed0 --hand-run .../runs/all/sar_hand/seed0 \
        --chips EMSR848_16-21 EMSR501_1-18 EMSR280_1-0 --out figures/fig_example_tiles.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np  # noqa: F401  (before torch: conda libstdc++ must load first)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from floodseg import data as D  # noqa: E402
from floodseg.model import build_model  # noqa: E402

# categorical slots 1-3 of the reference palette (validated all-pairs): hit, false alarm, miss
HIT, FALSE_ALARM, MISS = "#2a78d6", "#eb6834", "#1baf7a"
DRY, IGNORED, SURFACE, INK = "#f1f0ec", "#bdbcb6", "#fcfcfb", "#0b0b0b"


def predict(run: Path, root: Path, chip: D.Chip) -> tuple[np.ndarray, dict]:
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    channels = tuple(cfg["data"]["channels"])
    model = build_model(len(channels), cfg.get("model", {}))
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    model.eval()
    item = D.Sen1Floods11(root, [chip], channels)[0]
    with torch.no_grad():
        prob = torch.sigmoid(model(item["x"][None])).numpy()[0, 0]
    return prob >= cfg["eval"].get("threshold", 0.5), item


def error_map(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """0 dry/correct, 1 hit, 2 false alarm, 3 miss, 4 ignored."""
    out = np.zeros(y.shape, dtype=np.uint8)
    out[pred & (y == 1)] = 1
    out[pred & (y == 0)] = 2
    out[~pred & (y == 1)] = 3
    out[y == D.IGNORE] = 4
    return out


def view_window(y: np.ndarray) -> tuple[slice, slice]:
    """Crop to the labelled area; if that is a long strip, take the square window inside it that holds the
    most labelled water (a zoom), so unlabelled pixels do not dominate the panel."""
    v = y != D.IGNORE
    r, c = np.where(v)
    r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
    h, w = r1 - r0, c1 - c0
    if max(h, w) <= 1.5 * min(h, w):
        return slice(r0, r1), slice(c0, c1)
    s = min(h, w)
    water = (y == 1).astype(np.int64)
    if w > h:
        cols = water[r0:r1].sum(0)[c0:c1]
        best = int(np.argmax(np.convolve(cols, np.ones(s, dtype=np.int64), mode="valid")))
        return slice(r0, r1), slice(c0 + best, c0 + best + s)
    rows = water[:, c0:c1].sum(1)[r0:r1]
    best = int(np.argmax(np.convolve(rows, np.ones(s, dtype=np.int64), mode="valid")))
    return slice(r0 + best, r0 + best + s), slice(c0, c1)


def iou(pred: np.ndarray, y: np.ndarray) -> float:
    v = y != D.IGNORE
    tp, fp, fn = (pred & (y == 1) & v).sum(), (pred & (y == 0) & v).sum(), (~pred & (y == 1) & v).sum()
    return tp / max(tp + fp + fn, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--hand-run", type=Path, required=True)
    parser.add_argument("--chips", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    errors = ListedColormap([DRY, HIT, FALSE_ALARM, MISS, IGNORED])
    images = {}
    cols = ["SAR VV (dB)", "HAND (m)", "Label", "SAR only", "SAR + HAND"]
    fig, axes = plt.subplots(len(args.chips), len(cols), figsize=(2.1 * len(cols), 2.25 * len(args.chips) + 0.6))
    axes = np.atleast_2d(axes)
    for r, chip_id in enumerate(args.chips):
        chip = D.Chip(chip_id)
        pred_s, item = predict(args.base_run, args.data_root, chip)
        pred_h, _ = predict(args.hand_run, args.data_root, chip)
        y = item["y"][0].numpy()
        hand = item["hand"][0].numpy()
        with __import__("rasterio").open(args.data_root / D.HAND_DIR / "S1Hand" / f"{chip_id}_S1Hand.tif") as src:
            vv = src.read(1)
        label = np.where(y == D.IGNORE, 4, np.where(y == 1, 1, 0))
        rs, cs = view_window(y)
        zoom = (rs.stop - rs.start) * (cs.stop - cs.start) < y.size
        panels = [
            (vv, dict(cmap="gray", vmin=-25, vmax=0)),
            (np.clip(hand, 0, 30), dict(cmap="Blues_r", vmin=0, vmax=30)),  # sequential, one hue: dark = near drainage
            (label, dict(cmap=errors, vmin=-0.5, vmax=4.5)),
            (error_map(pred_s, y), dict(cmap=errors, vmin=-0.5, vmax=4.5)),
            (error_map(pred_h, y), dict(cmap=errors, vmin=-0.5, vmax=4.5)),
        ]
        for c, (img, kw) in enumerate(panels):
            ax = axes[r, c]
            images[c] = ax.imshow(img[rs, cs], interpolation="nearest", **kw)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
        km = (cs.stop - cs.start) * 10 / 1000
        axes[r, 0].set_ylabel(f"{chip.event}\n{chip_id.split('_', 1)[1]}" + (f"\n({km:.1f} km window)" if zoom else ""), fontsize=8)
        axes[r, 3].set_xlabel(f"IoU {iou(pred_s, y):.2f}", fontsize=8)
        axes[r, 4].set_xlabel(f"IoU {iou(pred_h, y):.2f}", fontsize=8)
    handles = [Patch(color=HIT, label="water, detected"), Patch(color=FALSE_ALARM, label="false alarm"),
               Patch(color=MISS, label="water, missed"), Patch(facecolor=DRY, edgecolor="#8a8984", lw=0.5, label="dry, correct"),
               Patch(color=IGNORED, label="not labelled (not scored)")]
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    # colour bars under the SAR and HAND columns, legend under the label / error columns
    for c, (label, ticks) in {0: ("VV backscatter (dB)", [-25, -15, -5, 0]), 1: ("HAND (m; ≥30 shown as 30)", [0, 10, 20, 30])}.items():
        pos = axes[-1, c].get_position()
        cax = fig.add_axes([pos.x0 + 0.1 * pos.width, 0.055, 0.8 * pos.width, 0.012])
        cb = fig.colorbar(images[c], cax=cax, orientation="horizontal", ticks=ticks)
        cb.ax.tick_params(labelsize=7, length=2)
        cb.set_label(label, fontsize=7, labelpad=1)
    x0 = axes[-1, 2].get_position().x0
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(x0, 0.0), ncol=3, frameon=False, fontsize=7.5,
               handlelength=1.2, columnspacing=1.0)
    fig.savefig(args.out, dpi=150)
    fig.savefig(args.out.with_suffix(".pdf"), dpi=400)  # image panels rasterized at 400 dpi (JSTARS colour)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
