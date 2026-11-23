# Pre-registered analyses

Five analyses were fixed in writing, with their analysis code, before their outcomes were computed. The
texts below are copied from the study's lab notebook as written at the time (lightly formatted). The
commit hashes refer to the development repository in which they were recorded; its timestamped history
is the evidence that each plan predates its result and is available from the authors on request.

Outcomes are reported in the paper whatever they showed. Four of the five were not supported or not
borne out.

## 1. Does the HAND gain decline with the number of training events at fixed data?

Fixed 2026-11-12 (commit c5616fa; script committed in 323a2a9).

> The post-hoc finding that the HAND gain shrinks with more events at fixed data. New, independent
> equal-tiles draws r3–r5 at 500 tiles from 20, 40 and 159 events. Test, on flood-only GEOID test, using
> only the new draws: per test event, mean paired gain over the e20 and e40 draws minus mean paired gain
> over the e159 draws; one-sided Wilcoxon over test events, α = 0.05. Same for all water as a secondary.
> Supported if > 0. Script: `scripts/confirm_gain_vs_events.py`.
>
> **Amendment (2026-11-12, before any confirmation run had finished):** 41 of the 80 GEOID test
> activations also have AoIs in GEOID train (the GEOID split is by AoI, not by activation). Primary
> analysis is therefore restricted to the test events whose activation is not in
> `geoid_trainval/splits/train.txt` (`--unseen-only`, 38 flood events); the all-events version is reported
> as secondary.

Disclosed deviation: while adding pooled IoU to `learning_curve_report.py` (2026-11-16), the per-draw
flood gains of the new e40 and e159 draws were seen before the last e20 draw finished (a failed job had to
be resubmitted). The test, events and draws had been committed beforehand and were not changed.

**Outcome: not supported.** Contrast +0.0025 [−0.0072, +0.0130], one-sided Wilcoxon p = 0.367 (flood,
38 unseen events); all water p = 0.368. Paper Section V-E, Table S4.

## 2. Relief gate: ignore HAND on flat ground

Fixed 2026-11-21 (commit 8b9a359), before any gated outcome was computed.

> A chip is flat when the 90th percentile of its valid HAND is < 2 m; flat chips take the SAR-only
> model's prediction (same seed / draw), all others the HAND model's. No retraining, no threshold tuning
> on test. Primary: gated − ungated HAND model, flood_diff under the GEOID-Flood protocol, unseen
> activations, 20 and 159 events, U-Net. Secondary: external-eval label modes, all events (includes
> EMSR419, which motivated the rule, so in-sample), held-out 2026 events, CD and TerraMind pairs. A
> threshold sweep is exploratory only. Script: `scripts/hand_gate.py`.

**Outcome: not supported.** Gated − ungated, protocol, unseen activations: 20 events −0.016
[−0.047, +0.001]; 159 events −0.003 [−0.007, +0.001]. Paper Section V-G, Table S6.

## 3. HAND modality dropout

Fixed 2026-11-21 before training (commit 6eb1e69); amendment before any result (commit b08389c).

> `sar_hand_md` = `sar_hand` with the HAND channel replaced by −1 (outside the normalized range) on a
> random 50% of training chips; HAND always present at test; everything else identical (159 events seeds
> 0–2, 20 events draws 0–4). Hypothesis: fewer large per-event HAND losses without losing the mean gain.
> Outcomes, unseen activations, two-way CIs: (1) mean: `sar_hand_md − sar_hand`, event-mean IoU
> (flooded-only and all-water labels) and pooled IoU_flood under the protocol; (2) tail: events where the
> variant is worse than SAR-only by > 0.02 event IoU (draw-averaged); (3) descriptive: EMSR280 (Sweden)
> all-water IoU. No other p, fill value or variant will be tried on test.
>
> **Amendment (before any `sar_hand_md` result):** the Sweden failure (EMSR280) is in the models trained on
> Sen1Floods11's 11 events, so also train `sar_hand_md` on all 11 Sen1Floods11 events (seeds 0–2) and score
> it on GEOID test (79 events) with the same outcomes.

**Outcome: not supported.** Dropout raises no mean gain by more than +0.001; with 11 training events it
removes part of HAND's gain (flood event-mean −0.008 [−0.016, −0.001]). Paper Section V-G, Table S7.

## 4. Terrain ablation on the Sen1Floods11 → GEOID-Flood transfer

Fixed 2026-11-21 before training (commit 58cc5ad).

> Train on all 11 Sen1Floods11 events, seeds 0–2: `sar_dem_slope` ([vv,vh,dem,slope]) and `sar_terrain`
> ([vv,vh,dem,hand,slope]), everything else as `sar_hand`. Score on geoid_test_flood, geoid_test and the
> GEOID-Flood protocol (test + held-out). Outcomes (`scripts/ablation_all_report.py`, two-way CIs over
> seeds × events, all 77/79 events): pairs sar_dem_slope − sar_only, sar_hand − sar_dem_slope,
> sar_terrain − sar_only, sar_terrain − sar_hand; event-mean IoU (flood, all water) and protocol pooled
> IoU_flood. Primary descriptive quantity: share of HAND's flood event-mean gain recovered by DEM + slope.
> Expectation from the GEOID 20-event ablation: about two-thirds. Reported whatever the result; no other
> variant or test set will be tried.

**Outcome: expectation not borne out.** DEM + slope recovers 26% of HAND's flood event-mean gain; HAND
beats DEM + slope (flood +0.024 [+0.005, +0.048], all water +0.047 [+0.010, +0.099]). Paper Section V-C,
Table III.

## 5. TerraMind on the Sen1Floods11 → GEOID-Flood transfer

Fixed 2026-11-21 before training (commit 6f7aacf).

> Train `fm_sar`, `fm_sar_hand` on all 11 Sen1Floods11 events, seeds 0–2, base.yaml schedule (100 epochs,
> 256 crops, batch 16), LR 5e-5 taken unchanged from the GEOID pilot (no tuning here). Same test sets and
> report as the ablation above; pairs fm_sar_hand − fm_sar (primary: is the HAND gain on TerraMind as large
> as on the U-Net?), fm_sar − sar_only, sar_hand − fm_sar. Reported whatever the result.

**Outcome:** HAND gain on TerraMind, flood event-mean +0.031 [−0.000, +0.066] (U-Net +0.032), all water
+0.040 [+0.004, +0.081] (U-Net +0.061); the U-Net with HAND beats TerraMind without it (flood +0.045
[+0.022, +0.075]). Paper Section V-D, Table IV.
