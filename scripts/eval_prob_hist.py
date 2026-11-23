"""Threshold-free check: is the HAND gain more than a shift in calibration?

`eval` stores, per run and test event, histograms of the predicted water probability (200 bins) for
labelled-positive and labelled-negative pixels (<run>/external_<name>/prob_hist.npz). `report` turns
them into average precision (AUPRC) per event and pooled over events, compares variant pairs per event
(paired over seeds) with a bootstrap over test events, and also reports IoU at the best single threshold
(chosen on the test set itself, so an upper bound for any calibration fix applied equally to both models).

    python scripts/eval_prob_hist.py eval --data-root $D/geoid_test_flood --name geoid_test_flood runs/geoid/*/seed*
    python scripts/eval_prob_hist.py report --runs runs/geoid --name geoid_test_flood --train-list .../train.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: F401  (import before torch: the conda libstdc++ must load first)
import pandas as pd

BINS = 200


def evaluate_runs(args: argparse.Namespace) -> None:
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from floodseg import data as D
    from floodseg.model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chips = [c for c in D.all_chips(args.data_root) if c.event not in args.exclude_events]
    events = sorted({c.event for c in chips})
    index = {e: i for i, e in enumerate(events)}
    for run in args.runs:
        out = run / f"external_{args.name}" / "prob_hist.npz"
        if out.exists():
            print("skip", run)
            continue
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        channels = tuple(cfg["data"]["channels"])
        model = build_model(len(channels), cfg.get("model", {})).to(device).eval()
        model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        loader = DataLoader(D.Sen1Floods11(args.data_root, chips, channels), batch_size=args.batch_size, num_workers=args.workers)
        pos = np.zeros((len(events), BINS), np.int64)
        neg = np.zeros((len(events), BINS), np.int64)
        with torch.no_grad():
            for batch in loader:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    prob = torch.sigmoid(model(batch["x"].to(device)).float()).cpu().numpy()[:, 0]
                y = batch["y"][:, 0].numpy()
                b = np.minimum((prob * BINS).astype(np.int64), BINS - 1)
                for i, ev in enumerate(batch["event"]):
                    k = index[ev]
                    pos[k] += np.bincount(b[i][y[i] == 1], minlength=BINS)
                    neg[k] += np.bincount(b[i][y[i] == 0], minlength=BINS)
        out.parent.mkdir(exist_ok=True)
        np.savez_compressed(out, events=np.array(events), pos=pos, neg=neg)
        print(run, "pooled AP %.4f" % average_precision(pos.sum(0), neg.sum(0)), flush=True)


def average_precision(pos: np.ndarray, neg: np.ndarray) -> float:
    """AP from binned scores, thresholds descending (step-wise sum of precision x recall increments)."""
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    if tp[-1] == 0:
        return np.nan
    precision = tp / np.maximum(tp + fp, 1)
    recall_step = pos[::-1] / tp[-1]
    return float((precision * recall_step).sum())


def best_iou(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    fn = pos.sum() - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    k = int(iou.argmax())
    return float(iou[k]), 1 - (k + 1) / BINS  # threshold = lower edge of the last included bin


def load(runs: Path, name: str) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    for f in sorted(runs.glob(f"*/seed*/external_{name}/prob_hist.npz")):
        z = np.load(f)
        out[(f.parents[2].name, f.parents[1].name)] = (z["events"], z["pos"], z["neg"])
    if not out:
        raise SystemExit(f"no prob_hist.npz for {name} under {runs}")
    return out


def report(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from learning_curve_report import training_events

    pd.set_option("display.width", 250)
    data = load(args.runs, args.name)
    events = next(iter(data.values()))[0]
    keep = np.ones(len(events), bool)
    if args.train_list:
        keep &= ~np.isin(events, list(training_events(args.train_list)))
    rng = np.random.default_rng(0)
    rows = []
    for pair in args.pairs:
        a, b = pair.split(":")
        seeds = sorted({s for v, s in data if v == a} & {s for v, s in data if v == b})
        if not seeds:
            continue
        # per (seed, event) AP; per seed pooled AP and best-threshold IoU
        H = {v: np.stack([np.stack(data[(v, s)][1:]) for s in seeds])[:, :, keep] for v in (a, b)}  # seed, pos/neg, event, bin
        has_pos = H[a][0, 0].sum(-1) > 0
        H = {v: h[:, :, has_pos] for v, h in H.items()}
        n_ev = H[a].shape[2]
        ap = {v: np.array([[average_precision(h[s, 0, e], h[s, 1, e]) for e in range(n_ev)] for s in range(len(seeds))]) for v, h in H.items()}
        d = (ap[a] - ap[b]).mean(0)  # paired over seeds, per event
        boot = rng.choice(d, (10000, n_ev)).mean(1)
        rows.append({"pair": pair, "measure": "event-mean AP", "events": n_ev, "seeds": len(seeds),
                     b: ap[b].mean(), a: ap[a].mean(), "delta": d.mean(),
                     "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5), "improved": f"{(d > 0).sum()}/{n_ev}"})
        idx = rng.integers(0, n_ev, (args.boot, n_ev))
        for measure, fn in (("pooled AP", lambda p, q: average_precision(p, q)), ("pooled IoU at best threshold", lambda p, q: best_iou(p, q)[0])):
            val = {v: np.mean([fn(h[s, 0].sum(0), h[s, 1].sum(0)) for s in range(len(seeds))]) for v, h in H.items()}
            bd = np.array([np.mean([fn(H[a][s, 0, i].sum(0), H[a][s, 1, i].sum(0)) - fn(H[b][s, 0, i].sum(0), H[b][s, 1, i].sum(0))
                                    for s in range(len(seeds))]) for i in idx])
            rows.append({"pair": pair, "measure": measure, "events": n_ev, "seeds": len(seeds), b: val[b], a: val[a],
                         "delta": val[a] - val[b], "ci_lo": np.percentile(bd, 2.5), "ci_hi": np.percentile(bd, 97.5)})
        thr = {v: [best_iou(h[s, 0].sum(0), h[s, 1].sum(0))[1] for s in range(len(seeds))] for v, h in H.items()}
        print(f"{pair}: best pooled-IoU thresholds by seed:", {v: [round(t, 3) for t in ts] for v, ts in thr.items()})
    table = pd.DataFrame(rows)
    print(f"{args.runs} / {args.name}" + (" (unseen activations only)" if args.train_list else ""))
    print(table.round(4).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("runs", nargs="+", type=Path)
    e.add_argument("--data-root", type=Path, required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--exclude-events", nargs="*", default=[])
    e.add_argument("--batch-size", type=int, default=2)
    e.add_argument("--workers", type=int, default=6)
    r = sub.add_parser("report")
    r.add_argument("--runs", type=Path, required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--train-list", type=Path)
    r.add_argument("--pairs", nargs="+", default=["sar_hand:sar_only"])
    r.add_argument("--boot", type=int, default=1000)
    r.add_argument("--out", type=Path)
    args = parser.parse_args()
    evaluate_runs(args) if args.cmd == "eval" else report(args)


if __name__ == "__main__":
    main()
