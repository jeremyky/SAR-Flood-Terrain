"""Where does the HAND gain occur? Confusion counts per HAND band, slope band and ESA WorldCover class on
GEOID-Flood test, for models trained on the 11 Sen1Floods11 events (descriptive; PREREGISTRATIONS.md #7).

    # 1. counts (GPU): one row per (variant, seed, event, stratum type, stratum)
    python scripts/strata_external.py eval --data-root $D/geoid_test_flood --out $S/reports/strata_geoid_test_flood.csv \
        $S/runs/all/{sar_only,sar_hand}/seed*
    # 2. report: pooled and event-mean IoU gain per stratum (two-way CIs) and false-alarm rate
    python scripts/strata_external.py report --counts $S/reports/strata_geoid_test_flood.csv --out ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (before torch: the conda libstdc++ must load first)
import pandas as pd
import rasterio
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from floodseg import data as D  # noqa: E402
from floodseg.model import build_model  # noqa: E402

HAND_BINS, HAND_LABELS = [1, 2, 5, 10, 20], ["0-1 m", "1-2 m", "2-5 m", "5-10 m", "10-20 m", ">20 m"]
SLOPE_BINS, SLOPE_LABELS = [2, 5, 10, 20], ["0-2°", "2-5°", "5-10°", "10-20°", ">20°"]
WORLDCOVER = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up", 60: "bare",
              70: "snow and ice", 80: "permanent water", 90: "herbaceous wetland", 95: "mangroves", 100: "moss and lichen"}


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chips = D.all_chips(args.data_root)
    terrain = args.data_root / "terrain"
    rows = []
    for run in args.runs:
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        channels = tuple(cfg["data"]["channels"])
        model = build_model(len(channels), cfg.get("model", {})).to(device)
        model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        model.eval()
        thr = cfg["eval"].get("threshold", 0.5)
        variant, seed = run.parent.name, run.name
        loader = DataLoader(D.Sen1Floods11(args.data_root, chips, channels), batch_size=2, num_workers=args.workers)
        acc: dict[tuple, np.ndarray] = {}
        with torch.no_grad():
            for batch in loader:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    probs = torch.sigmoid(model(batch["x"].to(device)).float()).cpu().numpy()[:, 0]
                for i, cid in enumerate(batch["chip"]):
                    y, hand = batch["y"][i, 0].numpy(), batch["hand"][i, 0].numpy()
                    valid = y != D.IGNORE
                    pred, truth = probs[i] >= thr, y == 1
                    slope = _read(terrain / f"{cid}_Slope.tif")
                    lc = _read(terrain / f"{cid}_WorldCover.tif")
                    strata = {
                        "hand": np.where(np.isfinite(hand), np.digitize(hand, HAND_BINS), -1),
                        "slope": np.where(np.isfinite(slope), np.digitize(slope, SLOPE_BINS), -1),
                        "landcover": np.where(np.isfinite(lc), lc, -1).astype(int),
                    }
                    for kind, s in strata.items():
                        for v in np.unique(s[valid]):
                            if v < 0:
                                continue
                            m = valid & (s == v)
                            c = np.array([(pred & truth & m).sum(), (pred & ~truth & m).sum(),
                                          (~pred & truth & m).sum(), (~pred & ~truth & m).sum()], dtype=np.int64)
                            key = (batch["event"][i], kind, int(v))
                            acc[key] = acc.get(key, 0) + c
        for (event, kind, v), c in acc.items():
            label = {"hand": HAND_LABELS, "slope": SLOPE_LABELS}.get(kind)
            name = label[v] if label else WORLDCOVER.get(v, str(v))
            rows.append({"variant": variant, "seed": seed, "event": event, "stratum_type": kind, "stratum": name,
                         "tp": c[0], "fp": c[1], "fn": c[2], "tn": c[3]})
        print(run, flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


def report(args: argparse.Namespace) -> None:
    from hand_gate import compare
    d = pd.read_csv(args.counts)
    rng = np.random.default_rng(0)
    rows = []
    for (kind, stratum), g in d.groupby(["stratum_type", "stratum"], sort=False):
        tot = g.groupby("variant")[["tp", "fp", "fn", "tn"]].sum()
        npix = tot.loc["sar_only"].sum() / g[g.variant == "sar_only"].seed.nunique()
        water = (tot.loc["sar_only", "tp"] + tot.loc["sar_only", "fn"]) / g[g.variant == "sar_only"].seed.nunique()
        r = {"stratum_type": kind, "stratum": stratum, "pixels_M": npix / 1e6, "water_pixels_M": water / 1e6}
        for v in ("sar_only", "sar_hand"):
            t = tot.loc[v]
            r[f"iou_{v}"] = t.tp / (t.tp + t.fp + t.fn) if (t.tp + t.fp + t.fn) else np.nan
            r[f"false_alarm_{v}"] = t.fp / (t.fp + t.tn) if (t.fp + t.tn) else np.nan
        if water > 0:
            c = compare(g.assign(metric="label"), "sar_hand", "sar_only", rng, args.boot)
            r.update({k: c[k] for k in ("events", "event_mean", "ev_lo", "ev_hi", "pooled", "pool_lo", "pool_hi")})
        rows.append(r)
    table = pd.DataFrame(rows)
    order = {"hand": 0, "slope": 1, "landcover": 2}
    table = table.sort_values(["stratum_type"], key=lambda s: s.map(order), kind="stable")
    pd.set_option("display.width", 250)
    print(f"{args.counts}: SAR + HAND vs SAR only per stratum; pooled IoU over events and seeds; "
          "two-way 95% CIs (seeds x events with water in the stratum); false alarm = FP / (FP + TN)")
    print(table.round(4).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("runs", nargs="+", type=Path)
    e.add_argument("--data-root", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--workers", type=int, default=6)
    r = sub.add_parser("report")
    r.add_argument("--counts", type=Path, required=True)
    r.add_argument("--boot", type=int, default=2000)
    r.add_argument("--out", type=Path)
    args = p.parse_args()
    evaluate(args) if args.cmd == "eval" else report(args)


if __name__ == "__main__":
    main()
