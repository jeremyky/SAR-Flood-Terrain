# Pre-registered analyses

Fifteen analyses were fixed in writing, with their analysis code, before their outcomes were computed. The
texts below are copied from the study's lab notebook as written at the time (lightly formatted). The
commit hashes refer to the development repository in which they were recorded; its timestamped history
is the evidence that each plan predates its result and is available from the authors on request.

Outcomes are reported in the paper whatever they showed. Of the first five, four were not supported or
not borne out; the mechanism tests (#6) were supported; #7 and #8 are descriptive.

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
38 unseen events); all water p = 0.368. Paper Section V-F, Table S4.

## 2. Relief gate: ignore HAND on flat ground

Fixed 2026-11-21 (commit 8b9a359), before any gated outcome was computed.

> A chip is flat when the 90th percentile of its valid HAND is < 2 m; flat chips take the SAR-only
> model's prediction (same seed / draw), all others the HAND model's. No retraining, no threshold tuning
> on test. Primary: gated − ungated HAND model, flood_diff under the GEOID-Flood protocol, unseen
> activations, 20 and 159 events, U-Net. Secondary: external-eval label modes, all events (includes
> EMSR419, which motivated the rule, so in-sample), held-out 2026 events, CD and TerraMind pairs. A
> threshold sweep is exploratory only. Script: `scripts/hand_gate.py`.

**Outcome: not supported.** Gated − ungated, protocol, unseen activations: 20 events −0.016
[−0.047, +0.001]; 159 events −0.003 [−0.007, +0.001]. Paper Section V-H, Table S6.

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
removes part of HAND's gain (flood event-mean −0.008 [−0.016, −0.001]). Paper Section V-H, Table S7.

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
Table IV.

## 5. TerraMind on the Sen1Floods11 → GEOID-Flood transfer

Fixed 2026-11-21 before training (commit 6f7aacf).

> Train `fm_sar`, `fm_sar_hand` on all 11 Sen1Floods11 events, seeds 0–2, base.yaml schedule (100 epochs,
> 256 crops, batch 16), LR 5e-5 taken unchanged from the GEOID pilot (no tuning here). Same test sets and
> report as the ablation above; pairs fm_sar_hand − fm_sar (primary: is the HAND gain on TerraMind as large
> as on the U-Net?), fm_sar − sar_only, sar_hand − fm_sar. Reported whatever the result.

**Outcome:** HAND gain on TerraMind, flood event-mean +0.031 [−0.000, +0.066] (U-Net +0.032), all water
+0.040 [+0.004, +0.081] (U-Net +0.061); the U-Net with HAND beats TerraMind without it (flood +0.045
[+0.022, +0.075]). Paper Section V-E, Table VI.

## 6. Does the gain need the right HAND for the place? (mechanism tests)

Fixed 2026-11-22, before any of these runs or evaluations, in response to an internal review asking whether
the HAND gain reflects drainage information about the scene or a side effect (normalization, regularization,
an easy-to-learn extra channel).

Models trained on the 11 Sen1Floods11 events (3 seeds), scored on all 77/79 GEOID-Flood test events,
flooded-only and all-water labels (`scripts/mechanism_report.py`, two-way 95% CIs over seeds × events,
event-mean IoU primary, pooled secondary). The scored pixels are identical in every condition; only the
model's HAND input changes (`floodseg.data`, `hand_perturb`).

- **Test-time perturbations** of the trained SAR + HAND models: `shift:10` (100 m misregistration),
  `shift:50` (500 m), `blur:25` (Gaussian, σ = 250 m), `swap` (HAND of a tile from another event, fixed
  donor per tile). Each compared with SAR only and with the unperturbed SAR + HAND model.
- **Training-time control** `sar_hand_swap`: trained and tested with swapped HAND (a random tile of another
  event per training sample; a fixed donor per tile at test), so the model sees realistic HAND values that
  carry no information about the scene. Compared with SAR only and with SAR + HAND.

Hypotheses: (H1) with `swap` and `shift:50` the gain over SAR only largely disappears (falls by more than half
or its CI includes zero), i.e. the gain needs correctly located HAND; (H2) `sar_hand_swap` gains nothing over
SAR only (CI includes zero), i.e. the gain is not a side effect of having an extra terrain-like channel.
`shift:10` (robustness to misregistration) and `blur:25` (fine drainage detail) are descriptive. No other
perturbation, magnitude or test set will be tried; results are reported whatever they show.

## 7. Where does the gain occur? (descriptive)

Fixed 2026-11-22. Same models (SAR only, SAR + HAND, 11 events, 3 seeds), all GEOID-Flood test events, both
label modes: confusion counts per HAND band (0–1, 1–2, 2–5, 5–10, 10–20, > 20 m), slope band (0–2, 2–5, 5–10,
10–20, > 20°) and ESA WorldCover class (`scripts/strata_external.py`). Reported: pooled IoU per stratum,
HAND gain with two-way CIs, and false-alarm rate FP / (FP + TN), in particular in built-up areas. Descriptive;
no hypothesis test.

## 8. Geographic-cluster sensitivity of the headline result

Fixed 2026-11-22. The CI of the headline transfer difference (SAR + HAND − SAR only, event-mean IoU, U-Net and
ResNet-34, both label modes) recomputed resampling countries (35) and seeds instead of events and seeds
(`scripts/cluster_bootstrap.py`), to account for events from the same country not being independent.
Reported alongside the event-level CI whatever it shows.

**Deviation (2026-11-22, before any outcome of #6 was seen).** The first submission of #6 ran with the
`floodseg` package installed in the conda environment, which still pointed at the development repository's
code without HAND perturbations: the three `sar_hand_swap` training runs silently ignored
`data.hand_perturb` (they trained ordinary SAR + HAND models), and every evaluation job failed on start-up, so
no result of #6 was produced or seen. The runs were moved out of the results tree; all SLURM scripts now
import this repository's own `src/`, training rejects unknown data config keys and records the code path,
and #6 was resubmitted unchanged. The stratified evaluation (#7) ran under the same environment, but it uses
no perturbation and the two code versions are otherwise identical, so it is kept.

## Outcomes of 6–8 (2026-11-22)

**6. Supported (both hypotheses).** Event-mean Δ vs SAR only, flood / all water: correct HAND +0.032 / +0.061;
shifted 100 m +0.025 / +0.031; shifted 500 m +0.003 [−0.011, +0.018] / −0.003; blurred σ = 250 m −0.001 /
−0.034; another event's HAND at test −0.061 / −0.122; trained and tested with another event's HAND −0.005
[−0.015, +0.006] / −0.001 [−0.021, +0.015]. Paper Section V-D, Table V, Table S9.

**7.** Flood-only pooled gain is largest at 1–5 m HAND (≈ +0.09) and comes largely from fewer false alarms on
higher ground (false-alarm rate 0.77% → 0.19% at 5–10 m, 0.62% → 0.02% above 20 m); gains in tree cover,
grassland, cropland, built-up and bare ground, small or none in herbaceous wetland. Paper Section V-B,
Table S8.

**8.** Country-cluster CIs are essentially the event-level ones (U-Net flood +0.032 [+0.017, +0.051] vs
[+0.018, +0.050], 34 countries, as one country has no flood event; all water +0.061 [+0.039, +0.089] vs [+0.040, +0.087]). Paper Section V-B.

## 9-15. Third review (fixed 2026-11-23, before any of these runs or analyses)

Post hoc in the sense that they answer an external review of the preprint draft (tagged `preprint-draft-2026-11-23`)
rather than a hypothesis we held before seeing results; fixed here before any outcome was computed. Seen beforehand:
every result in the draft (including the leave-one-event-out gain, +0.025 event-mean with 5 seeds, and the
official-split gain, +0.007 pooled with 3 seeds); the overlap counts of #11 at activation level (15 of 80 test
activations have AoIs in GEOID-Flood validation, only EMSR586 of them none in train, so "unseen" shrinks from 38 to
37 flood and 39 to 38 water events); the epochs per learning-curve subset (#14). The new scripts were smoke-tested on
two tiles of one event (EMSR162) with one SAR-only model, and the matched report (#9) with SAR only paired against
itself, which shows no HAND outcome. Unless stated otherwise, CIs are two-way bootstraps over seeds and test events,
and results are reported whatever they show.

### 9. Same tiles, same metric: does HAND help more when the tile's event is held out?

The 90 tiles of Sen1Floods11's official test split come from 10 events. Each is scored by the official-split models
(trained on other tiles of the same 10 events) and by the leave-one-event-out model that held its event out; SAR only
and SAR + HAND, seeds 0-4 in both settings (official-split seeds 3-4 are new). Per event, IoU (all water, hand labels)
is pooled over its official test tiles; the gain is SAR + HAND minus SAR only from the same seed. **Primary:** event-mean
gain with the event held out minus event-mean gain with the event in training, two-way bootstrap (events shared by
both settings, seeds resampled within each). **Secondary:** the same difference in IoU pooled over the 90 tiles; the
number of events where the held-out gain is larger. Script: `scripts/matched_split_report.py`.
**Hypothesis:** the difference is positive (95% CI above zero). **Consequence fixed now:** if the CI includes zero,
the title, abstract and introduction stop saying that same-event splits hide the gain and report only the measured
contrast.

### 10. Physics losses and HAND post-masks on the Sen1Floods11 → GEOID-Flood transfer

Train `sar_mono`, `sar_ceil`, `sar_mono3_w1`, `sar_mono3_w10` (the leave-one-event-out loss variants, Table S1) on
all 11 Sen1Floods11 events, seeds 0-2. Score HAND post-masks at 5, 10, 15 and 20 m (water removed where HAND exceeds the
threshold; no retraining) on the SAR-only and SAR + HAND models, seeds 0-4. Outcomes as Table IV (all 77/79 GEOID test
events): event-mean IoU flooded only and all water, pooled IoU_flood under the GEOID-Flood protocol (masks applied to
both water maps before differencing). Pairs: each loss variant − SAR only; each SAR-only mask − SAR only; each SAR +
HAND mask − SAR + HAND. Script: `scripts/revision_transfer_report.py losses|masks`. **Consequence fixed now:** the
statement that HAND as a loss or mask does nothing is kept for the transfer only if no loss or mask improves SAR only
in flood or all-water event-mean IoU with a CI above zero; otherwise every such gain is reported next to the input
gain.

### 11. Overlap of GEOID-Flood test events with train and validation

Descriptive counts: test activations with AoIs in train and in validation; test tiles whose footprint equals that of a
train or validation tile, in any activation (`scripts/split_overlap_audit.py`). Sensitivity: every result of
GEOID-Flood-trained models reported on unseen activations (Tables VII-IX, S2, S4-S7 and the confirmation test #1) is
recomputed with unseen meaning unseen by train *and* validation (`--unseen-only`/`--train-list` set to train + valid;
`TRAIN=... sbatch slurm/paper_reports.sbatch` and the scripts' `--train-list`). The pre-registered primaries stay as
registered; the sensitivity results are reported next to them. The Sen1Floods11-trained transfer uses no GEOID-Flood
data for training or checkpoint selection, so it is unaffected.

### 12. TerraMind learning rate on the Sen1Floods11 → GEOID-Flood transfer

`fm_sar` and `fm_sar_hand` at learning rates 1e-5, 2e-5, 1e-4 and 2e-4 (5e-5 exists), seeds 0-2, otherwise as #5. For
each variant the learning rate is chosen by the mean over seeds of its best Sen1Floods11 validation IoU (the 15%
validation split of the 11 events), never by test results. **Primary:** Table VI's comparisons (TM + HAND − TM, TM −
U-Net, U-Net + HAND − TM) at the selected learning rates; all learning rates reported in the supplement. Script:
`scripts/revision_transfer_report.py fm-lr`. **Consequence fixed now:** if U-Net + HAND − TM no longer has a CI above zero
on flood event-mean IoU, the claim that the small U-Net with HAND beats TerraMind is dropped.

### 13. Do the reference labels look HAND-masked? (descriptive)

Per event, for GEOID-Flood test (all water; flooded only) and Sen1Floods11 hand labels: the share of valid area and
of labelled water with HAND above 5, 10, 15 and 20 m, the 99th percentile of labelled-water HAND, and a cut-off flag
(at least 1,000 water pixels within 1 m below the threshold, at least 1,000 valid pixels within 1 m above it, and no
water there). `scripts/label_hand_check.py`. Interpreted together with what the labelling procedures document.

### 14. Learning curve at a fixed number of epochs

The learning curve fixes the optimizer steps (32,670), so small subsets are passed over many more times (10 events:
about 360-1,020 epochs; 20: 150-400; 159: 30). Retrain SAR only and SAR + HAND on the 10-, 20- and 40-event subsets
(draws 0-4, seed = draw) for 30 epochs, as the 159-event models (`bash scripts/sweep_geoid.sh epochs`), and score on
GEOID test (flooded only, all water). Reported: the HAND gain per size (event-mean and pooled, unseen activations by
train, and by train and validation) next to the fixed-step gains; descriptive, no test.

### 15. Five seeds for the headline transfer

Seeds 3-4 for the U-Net and ResNet-34, SAR only and SAR + HAND, trained on all 11 Sen1Floods11 events and scored on
GEOID test (flooded only, all water) and under the protocol. Table III and Fig. 1 then report seeds 0-4 whatever they
show. Tables registered with 3 seeds (#4-#8) stay at seeds 0-2.

## Outcomes of 9–15 (from 2026-11-23)

Execution notes, decided without seeing any outcome: GPU-queue waits were long, so the official-split and
all-event seeds 3-4, the loss variants and the mask evaluations ran on A100 MIG slices (1g.24gb, 2 CPUs)
instead of full GPUs, with identical code and configs (`SBATCH_EXTRA` in `scripts/sweep_revision.sh`). The
report scripts glob every seed, so `scripts/ablation_all_report.py` and `scripts/mechanism_report.py` gained
`--seeds` to keep the tables registered with 3 seeds (#4-#8) at seeds 0-2. In Fig. 1b the + HAND row uses
seeds 0-4 (#15) and the other rows seeds 0-2 (Tables IV and VI). `slurm/paper_reports.sbatch` now also runs
the confirmation test, relief gate and dropout reports, so one call with `TRAIN=.../train_valid.txt` gives
every sensitivity result of #11.

**9. Not supported.** On the 90 official test tiles (10 events, 5 seeds per setting), event-mean HAND gain
with the event in training +0.001 [−0.021, +0.019], with it held out +0.017 [−0.004, +0.043]; held out minus
in training +0.016 [−0.009, +0.045] (primary). Pooled over the 90 tiles: +0.008 vs. +0.024, difference +0.016
[−0.013, +0.046]. The held-out gain is larger on 8 of 10 events; Paraguay contributes most. As fixed in
advance, the title, abstract and introduction no longer say that same-event splits hide the gain; they
report the contrast. With seeds 3-4 the official-split numbers also change: test +0.008 ± 0.007 (was
+0.007 ± 0.008 with 3 seeds), Bolivia +0.105 ± 0.074 (was +0.128 ± 0.072; per seed +0.160, +0.046, +0.179,
+0.133, +0.008). Paper Section V-A, Fig. 1a, Table S10; `reports/matched_split.csv`.

**11 (descriptive part).** 41 of the 80 test activations have AoIs in train and 15 in validation; only
EMSR586 is in validation but not in train, so unseen by train and validation leaves 37 flood / 38 water
events. No test tile has exactly the footprint of any train or validation tile (0 of 2,635). Paper Section
III; `reports/split_overlap_audit.txt`.

**13.** No event is flagged as a cut-off at 5, 10, 15 or 20 m in either dataset. Of the 72 flood events with at
least 1% of their area above 15 m HAND, 56 have labelled flood water there. Median share of labelled water above
15 m: 0.15% (GEOID flood), 0.09% (GEOID all water), 0.01% (Sen1Floods11 hand labels); 99th percentile of
labelled-water HAND, median over events: 6.5 m, 6.0 m, 5.0 m. Paper Section VI-D, Table S15;
`reports/label_hand_check.txt`.
