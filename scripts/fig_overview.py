"""Overview figure (manuscript Fig. 1): the paper's two headline results for models trained on the 11
Sen1Floods11 events.

(a) Where the HAND gain shows up: Sen1Floods11's official test split (tiles from the training events), its
    Bolivia hold-out (an unseen event) and 79 external GEOID-Flood events. Pooled water IoU, SAR + HAND
    minus SAR only: per-seed dots and mean for Sen1Floods11, mean and two-way 95% CI for GEOID-Flood.
(b) What helps on 77 new floods: event-mean flood IoU gain over the SAR-only U-Net for terrain inputs
    and for TerraMind with and without HAND (two-way 95% CIs over seeds and events).

    python scripts/fig_overview.py --scratch /scratch/$USER/floodseg --out figures/fig_overview.png

Reads reports/official_split.csv (scripts/official_report.py) and reports/ablation_all.csv
(scripts/ablation_all_report.py); the two TerraMind-vs-U-Net rows are computed here with the same
bootstrap (hand_gate.compare).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hand_gate import compare, load  # noqa: E402
from make_figures import COLORS, INK, INK2, SURFACE, plt, save  # noqa: E402

GREY = "#8a8984"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("figures/fig_overview.png"))
    args = p.parse_args()
    rep = args.scratch / "reports"

    off = pd.read_csv(rep / "official_split.csv").set_index(["split", "variant"])
    abl = pd.read_csv(rep / "ablation_all.csv").set_index(["test", "comparison"])
    flood = load(args.scratch / "runs" / "all", "geoid_test_flood")
    rng = np.random.default_rng(0)
    tm = {x: compare(flood, x, "sar_only", rng, 4000) for x in ("fm_sar", "fm_sar_hand")}

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.6), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.9})

    # (a) same-event test tiles vs unseen floods
    rows = [("Sen1Floods11 test\n(training events)", "test"),
            ("Bolivia\n(unseen event)", "bolivia")]
    for i, (label, split) in enumerate(rows):
        r = off.loc[(split, "sar_hand")]
        seeds = [float(s) for s in r.delta_by_seed.split()]
        a.plot(seeds, [i] * len(seeds), "o", ms=4, color=COLORS["sar_hand"], alpha=0.45, mec="none")
        a.plot(r.delta_mean, i, "D", ms=7, color=COLORS["sar_hand"], mec=SURFACE, mew=1.2)
        a.annotate(f"{r.delta_mean:+.3f}".replace("-", "−"), (max(seeds), i), xytext=(6, 0),
                   textcoords="offset points", va="center", fontsize=7.5, color=INK2)
    g = abl.loc[("all water", "sar_hand - sar_only")]
    a.plot([g.pool_lo, g.pool_hi], [2, 2], color=COLORS["sar_hand"], lw=2, solid_capstyle="round")
    a.plot(g.pooled, 2, "D", ms=7, color=COLORS["sar_hand"], mec=SURFACE, mew=1.2)
    a.annotate(f"{g.pooled:+.3f}", (g.pool_hi, 2), xytext=(6, 0), textcoords="offset points", va="center",
               fontsize=7.5, color=INK2)
    a.set_yticks(range(3), [r[0] for r in rows] + ["79 GEOID-Flood\nevents (unseen)"])
    a.set_xlabel("HAND gain\n(pooled water IoU)")
    a.set_title("(a) Where the gain shows up", loc="left")

    # (b) what helps on new floods
    items = [("+ DEM + slope", abl.loc[("flooded only", "sar_dem_slope - sar_only")], GREY),
             ("+ HAND", abl.loc[("flooded only", "sar_hand - sar_only")], COLORS["sar_hand"]),
             ("+ HAND, DEM, slope", abl.loc[("flooded only", "sar_terrain - sar_only")], COLORS["sar_hand"]),
             ("TerraMind (101 M)", pd.Series(tm["fm_sar"]), COLORS["sar_only"]),
             ("TerraMind + HAND", pd.Series(tm["fm_sar_hand"]), COLORS["sar_only"])]
    for i, (label, r, c) in enumerate(items):
        b.plot([r.ev_lo, r.ev_hi], [i, i], color=c, lw=2, solid_capstyle="round")
        b.plot(r.event_mean, i, "o", ms=7, color=c, mec=SURFACE, mew=1.2)
        b.annotate(f"{r.event_mean:+.3f}".replace("-", "−"), (r.ev_hi, i), xytext=(6, 0),
                   textcoords="offset points", va="center", fontsize=7.5, color=INK2)
    b.set_yticks(range(len(items)), [it[0] for it in items])
    b.set_xlabel("Gain over SAR-only U-Net (7.8 M)\n(event-mean flood IoU)")
    b.set_title("(b) Gains on 77 new floods", loc="left")

    for ax in (a, b):
        ax.axvline(0, color=INK2, lw=0.8)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    a.set_xlim(-0.04, 0.24)
    b.set_xlim(-0.06, 0.10)
    fig.suptitle("Trained on the 11 Sen1Floods11 events: terrain helps on new floods, not on familiar ones",
                 x=0.01, ha="left", fontsize=10)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.82, bottom=0.25)
    save(fig, args.out)
    print({k: {m: round(v[m], 4) for m in ("event_mean", "ev_lo", "ev_hi")} for k, v in tm.items()})


if __name__ == "__main__":
    main()
