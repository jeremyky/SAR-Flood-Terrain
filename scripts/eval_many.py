"""Score several runs in one pass over the data. Writes exactly what scripts/eval_external.py (`external`),
`scripts/eval_geoid_protocol.py eval` (`protocol`) and `scripts/eval_prob_hist.py eval` (`probhist`) write, but
loads every batch once for all runs instead of once per run. Useful where data loading, not the GPU, is the bottleneck (few CPUs). Single-image runs that share their
input channels only; no HAND perturbation or post-mask.

    python scripts/eval_many.py external --data-root $D/geoid_test --name geoid_test runs/all/sar_mono/seed*
    python scripts/eval_many.py protocol --post $D/geoid_test_raw --pre $D/geoid_test_pre --name geoid_proto_test \
        runs/all/sar_mono/seed*
    python scripts/eval_many.py probhist --data-root $D/geoid_test --name geoid_test runs/all/sar_hand/seed{3,4}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (import before torch: the conda libstdc++ must load first)
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from eval_geoid_protocol import counts  # noqa: E402
from eval_prob_hist import BINS, average_precision  # noqa: E402
from floodseg import data as D  # noqa: E402
from floodseg.metrics import Tracker  # noqa: E402
from floodseg.model import build_model  # noqa: E402


def load_models(runs: list[Path], name: str, device, done: str = "chips.csv") -> tuple[list[dict], tuple[str, ...]]:
    models, channels = [], set()
    for run in runs:
        out = run / f"external_{name}"
        if (out / done).exists():
            print("skip", run)
            continue
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        ch = tuple(cfg["data"]["channels"])
        if cfg["data"].get("hand_perturb") or any(c.endswith("_pre") for c in ch):
            raise SystemExit(f"{run}: HAND-perturbed and change-detection runs are not supported")
        model = build_model(len(ch), cfg.get("model", {})).to(device).eval()
        model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        models.append({"run": run, "out": out, "model": model, "thr": cfg["eval"].get("threshold", 0.5),
                       "hand_mask": cfg["eval"].get("hand_mask")})
        channels.add(ch)
    if len(channels) > 1:
        raise SystemExit(f"runs differ in input channels: {channels}")
    return models, (channels.pop() if channels else ())


def water(model: dict, batch: dict, device) -> np.ndarray:
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return torch.sigmoid(model["model"](batch["x"].to(device)).float()).cpu().numpy()[:, 0]


def external(args, device) -> None:
    """As scripts/eval_external.py with floodseg.train.evaluate (raw and the run's own HAND mask)."""
    models, channels = load_models(args.runs, args.name, device)
    if not models:
        return
    chips = [c for c in D.all_chips(args.data_root) if c.event not in args.exclude_events]
    print(f"{len(chips)} chips, {len({c.event for c in chips})} events, {len(models)} runs", flush=True)
    loader = DataLoader(D.Sen1Floods11(args.data_root, chips, channels), batch_size=args.batch_size,
                        num_workers=args.workers)
    trackers = [{"raw": Tracker(), "hand_mask": Tracker()} for _ in models]
    for batch in loader:
        for m, tr in zip(models, trackers):
            probs = water(m, batch, device)
            for i in range(len(probs)):
                pred = probs[i] >= m["thr"]
                y, hand = batch["y"][i, 0].numpy(), batch["hand"][i, 0].numpy()
                tr["raw"].add(batch["chip"][i], batch["event"][i], pred, y, hand)
                if m["hand_mask"] is not None:
                    masked = pred & ~(np.nan_to_num(hand, nan=0.0) > m["hand_mask"])
                    tr["hand_mask"].add(batch["chip"][i], batch["event"][i], masked, y, hand)
    for m, tr in zip(models, trackers):
        m["out"].mkdir(exist_ok=True)
        (m["out"] / "metrics.json").write_text(json.dumps({k: t.summary() for k, t in tr.items() if t.rows}, indent=1))
        rows = tr["raw"].per_chip()
        with open(m["out"] / "chips.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(m["run"], json.dumps(tr["raw"].summary()["total"]), flush=True)


def protocol(args, device) -> None:
    """As `scripts/eval_geoid_protocol.py eval` for single-image runs without --post-mask."""
    models, channels = load_models(args.runs, args.name, device)
    if not models:
        return
    pre_ids = {c.chip_id for c in D.all_chips(args.pre)}
    chips = [c for c in D.all_chips(args.post) if c.chip_id in pre_ids]
    print(f"{len(chips)} chips with a pre-event image, {len({c.event for c in chips})} events, {len(models)} runs",
          flush=True)
    loaders = [DataLoader(D.Sen1Floods11(root, chips, channels), batch_size=args.batch_size, num_workers=args.workers)
               for root in (args.post, args.pre)]
    rows = [[] for _ in models]
    for post, pre in zip(*loaders):
        assert list(post["chip"]) == list(pre["chip"])
        y_post, y_pre = post["y"][:, 0].numpy(), pre["y"][:, 0].numpy()
        for m, r in zip(models, rows):
            w_post_all, w_pre_all = (water(m, b, device) >= m["thr"] for b in (post, pre))
            for i, chip in enumerate(post["chip"]):
                valid = (y_post[i] >= 0) & (y_pre[i] >= 0)
                flood, perm_or_flood = y_post[i] == 2, y_post[i] >= 1
                w_post, w_pre = w_post_all[i], w_pre_all[i]
                for metric, pred, gt in (("flood_diff", w_post & ~w_pre, flood), ("flood_post", w_post, flood),
                                         ("water_post", w_post, perm_or_flood)):
                    r.append([chip, post["event"][i], metric, *counts(pred, gt, valid)])
    for m, r in zip(models, rows):
        m["out"].mkdir(exist_ok=True)
        df = pd.DataFrame(r, columns=["chip", "event", "metric", "tp", "fp", "fn", "tn"])
        df.to_csv(m["out"] / "chips.csv", index=False)
        pooled = df.groupby("metric")[["tp", "fp", "fn"]].sum()
        print(m["run"], (pooled.tp / (pooled.tp + pooled.fp + pooled.fn)).round(4).to_dict(), flush=True)


def probhist(args, device) -> None:
    """As `scripts/eval_prob_hist.py eval`: per event, 200-bin histograms of the water probability."""
    models, channels = load_models(args.runs, args.name, device, done="prob_hist.npz")
    if not models:
        return
    chips = [c for c in D.all_chips(args.data_root) if c.event not in args.exclude_events]
    events = sorted({c.event for c in chips})
    index = {e: i for i, e in enumerate(events)}
    print(f"{len(chips)} chips, {len(events)} events, {len(models)} runs", flush=True)
    loader = DataLoader(D.Sen1Floods11(args.data_root, chips, channels), batch_size=args.batch_size,
                        num_workers=args.workers)
    hists = [(np.zeros((len(events), BINS), np.int64), np.zeros((len(events), BINS), np.int64)) for _ in models]
    for batch in loader:
        y = batch["y"][:, 0].numpy()
        for m, (pos, neg) in zip(models, hists):
            b = np.minimum((water(m, batch, device) * BINS).astype(np.int64), BINS - 1)
            for i, ev in enumerate(batch["event"]):
                k = index[ev]
                pos[k] += np.bincount(b[i][y[i] == 1], minlength=BINS)
                neg[k] += np.bincount(b[i][y[i] == 0], minlength=BINS)
    for m, (pos, neg) in zip(models, hists):
        m["out"].mkdir(exist_ok=True)
        np.savez_compressed(m["out"] / "prob_hist.npz", events=np.array(events), pos=pos, neg=neg)
        print(m["run"], "pooled AP %.4f" % average_precision(pos.sum(0), neg.sum(0)), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["external", "protocol", "probhist"])
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--data-root", type=Path, help="external: dataset root")
    p.add_argument("--exclude-events", nargs="*", default=[])
    p.add_argument("--post", type=Path, help="protocol: post-event root (three-class labels)")
    p.add_argument("--pre", type=Path, help="protocol: pre-event root")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    {"external": external, "protocol": protocol, "probhist": probhist}[args.mode](args, device)


if __name__ == "__main__":
    main()
