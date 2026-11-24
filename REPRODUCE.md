# Reproducing the results

Everything runs on a SLURM cluster with one GPU per job (we used UVA Rivanna/Afton, A40/A6000/A100).
Paths default to `/scratch/$USER/floodseg`; set `FLOODSEG_ROOT` to change that (configs expand
`${FLOODSEG_ROOT}`), `FLOODSEG_ENV` for the conda env, and `ACCOUNT` for the SLURM allocation.

## Provenance

- Every run directory holds the fully resolved `config.yaml` it was trained with, including `git_commit`;
  those hashes refer to the development repository (see `PREREGISTRATIONS.md`), from which this
  repository was copied without history. The code is unchanged in substance.
- `experiments/slurm_history.tsv`: every training / evaluation / data-prep submission (job id, time,
  state, exact command), exported from SLURM accounting. Failed jobs were resubmitted; the run used is
  the last COMPLETED one for that run name and seed.
- `experiments/slurm_history.tsv` is refreshed by `scripts/export_slurm_history.sh`.
- Zenodo deposit (built by `sbatch slurm/archive.sbatch`, metadata `paper/zenodo.json`): the code at the
  paper's commit, every run directory without weights (configs, histories, metrics, per-chip confusion
  counts) with all reports, and the checkpoints of every model behind a GEOID-Flood result; checksums in
  `SHA256SUMS`. **DOI: [to add after upload].**
- Training is seeded (`random`, `numpy`, `torch`) but cuDNN kernels are not forced deterministic, so a
  rerun gives slightly different numbers per run; the paired, multi-seed statistics are what we report.

## 1. Environment

```bash
conda env create -p ~/.conda/envs/floodseg -f env/environment.lock.yml   # exact versions we used
# or: conda create ... python=3.11; pip install -r env/requirements.lock.txt; pip install -e .
```
torch 2.11.0+cu128 (no V100 / sm_70 kernels; `slurm/train.sbatch` constrains the GPU type).
Import numpy before torch on hosts with an old system libstdc++.
TerraMind baseline only: a second env with terratorch, see `env/README-fm.md`.

## 2. Data (public)

```bash
python scripts/download_sen1floods11.py --root $FLOODSEG_ROOT/data/sen1floods11    # ~0.7 GB
python scripts/build_terrain.py        --root $FLOODSEG_ROOT/data/sen1floods11    # GLO-30 DEM + HAND, slope
python scripts/build_landcover.py      --root $FLOODSEG_ROOT/data/sen1floods11    # ESA WorldCover (analysis only)
bash scripts/download_geoid.sh                                                    # GEOID-Flood, ~190 GB
sbatch slurm/prepare_geoid.sbatch            # test split -> data/geoid_test (all water) and data/geoid_test_flood (EMSR388 excluded)
sbatch slurm/prepare_geoid_trainval.sbatch   # train/val -> data/geoid_trainval; held-out 2026 -> data/geoid_heldout{,_flood}
sbatch slurm/prepare_geoid_protocol.sbatch   # three-class labels + latest pre-event S1 -> data/geoid_{test,heldout}_{raw,pre}
sbatch slurm/build_jrc.sbatch                # JRC GSW >= 50% layer for the GEOID roots (calibrate: scripts/build_jrc.py)
sbatch slurm/prepare_geoid_cd.sbatch         # pre-event S1 for train/val, flood-vs-rest labels -> data/geoid_trainval_cd
```

## 3. Experiments → paper

Sections, tables and figures refer to `paper/jstars/main.pdf`; S-items to `supplement.pdf`. `$D` is the
data root, `runs/` and `reports/` live under `$FLOODSEG_ROOT`, and `train.txt` is
`$D/geoid_trainval/splits/train.txt`.

| paper | what | submit | report |
|---|---|---|---|
| III, Tables I–II | datasets and test-event selection | (section 2 above) | `scripts/prepare_geoid.py`; event counts from `scripts/external_report.py` and `learning_curve_report.py --unseen-only/--shared-only` |
| V-A, Table S1 | Sen1Floods11 leave-one-event-out, 6 variants × 3 seeds (+ seeds 3–4 for the main pair; 30-m-stride monotonicity variants) | `bash scripts/sweep.sh loeo "0 1 2"` | `scripts/loeo_report.py`, `scripts/hand_strata.py` |
| V-A, Fig. 1a | official split and Bolivia (seeds 0–4) | `bash scripts/sweep.sh official "0 1 2"`; seeds 3–4: `bash scripts/sweep_revision.sh seeds` | `scripts/official_report.py --runs runs --out reports/official_split.csv` |
| V-A, Fig. 1a, Table S10 | same tiles with their event in training or held out (pre-registered #9) | (official and LOEO runs above, seeds 0–4) | `scripts/matched_split_report.py --runs runs --out reports/matched_split.csv` |
| V-B, Table S11 | physics losses and HAND post-masks (5–20 m) on the transfer (pre-registered #10) | `bash scripts/sweep_revision.sh losses`, `... masks` (evaluations included; the loss-run evaluations were run with `scripts/eval_many.py external` on `geoid_test` and `geoid_test_flood` and `scripts/eval_many.py protocol` with `--name geoid_proto_test`, same outputs) | `scripts/revision_transfer_report.py losses\|masks --runs runs/all --out reports/transfer_{losses,masks}.csv`; the exploratory input-vs-mask rows: `... input-vs-mask --out reports/transfer_input_vs_mask_exploratory.csv` |
| V-B, Table III, Fig. 2, Fig. S2 | train on all 11 events, U-Net and ResNet-34, test on GEOID-Flood (seeds 0–4, #15) | `VARIANTS="sar_only sar_hand r34_sar_only r34_sar_hand" bash scripts/sweep.sh all "0 1 2"`, then `slurm/eval_external.sbatch` (geoid_test, geoid_test_flood); seeds 3–4: `bash scripts/sweep_revision.sh seeds` | `scripts/revision_transfer_report.py headline --runs runs/all --out reports/transfer_headline.csv` (Table III, Fig. 1 CIs); `scripts/external_report.py --out reports/headline5/external_<name>.csv` (IoU, precision, recall, Wilcoxon); `scripts/external_explain.py` on those (Fig. S2, correlations); `scripts/example_tiles.py` |
| V-B | threshold-free (AP, best-threshold IoU), seeds 0–4 | `sbatch slurm/eval_prob_hist.sbatch $D/<name> <name> runs/all/{sar_only,sar_hand,r34_sar_only,r34_sar_hand}/seed*` (seeds 3-4 were run with `scripts/eval_many.py probhist`, same output) | `scripts/eval_prob_hist.py report --runs runs/all --name <name> --pairs sar_hand:sar_only r34_sar_hand:r34_sar_only` → `reports/probhist5.txt` |
| V-B to V-G | GEOID-Flood pre/post protocol | `sbatch slurm/eval_geoid_protocol.sbatch <test\|heldout> <run dirs>` | `scripts/eval_geoid_protocol.py report --train-list train.txt` |
| V-C, Table IV, Fig. 1b | terrain ablation on the transfer (pre-registered) | `VARIANTS="sar_dem_slope sar_terrain" bash scripts/sweep.sh all "0 1 2"` + both evals | `scripts/ablation_all_report.py --runs runs/all --seeds seed0 seed1 seed2 --out reports/ablation_all.csv` |
| V-C | JRC permanent water on the transfer | `VARIANTS="sar_jrc sar_hand_jrc" bash scripts/sweep.sh all "0 1 2"` (`scripts/build_jrc.py`) | `scripts/external_report.py --pairs ...` |
| V-C, Table S2 | terrain and JRC ablation with GEOID-Flood training | `bash scripts/sweep_geoid.sh ablate`, `... jrc` | `slurm/paper_reports.sbatch` |
| V-B, Table S8 | errors by HAND band, slope band and land cover (pre-registered, descriptive) | (no training) | `scripts/strata_external.py eval --data-root $D/geoid_test{,_flood} --out reports/strata_geoid_test{,_flood}.csv runs/all/{sar_only,sar_hand}/seed{0,1,2}` (GPU; registered with seeds 0–2), then `... report` |
| V-B | country-cluster bootstrap (pre-registered) | (no training) | `scripts/cluster_bootstrap.py --runs runs/all --events external_explain_events.csv --seeds seed0 seed1 seed2` (registered with seeds 0–2) |
| V-D, Table V, Table S9 | HAND perturbations and swapped-HAND control (pre-registered) | `VARIANTS=sar_hand_swap bash scripts/sweep.sh all "0 1 2"` + both evals; `slurm/eval_external.sbatch $D/geoid_test{,_flood} <name>_h<k> --hand-perturb {shift:10,shift:50,blur:25,swap} runs/all/sar_hand/seed{0,1,2}` | `scripts/mechanism_report.py --runs runs/all --seeds seed0 seed1 seed2 --out reports/mechanism.csv` |
| V-E, Table VI, Fig. 1b, Table S12 | TerraMind-1.0-base ± HAND on the transfer (pre-registered #5) at learning rates 1e-5 to 2e-4, the rate chosen on Sen1Floods11 validation (pre-registered #12); 5e-5 is the earlier draft's | `FLOODSEG_ENV=~/.conda/envs/floodseg-fm VARIANTS="fm_sar fm_sar_hand" bash scripts/sweep.sh all "0 1 2"` + both evals (5e-5); `bash scripts/sweep_revision.sh fm-lr` (other rates) | `scripts/revision_transfer_report.py fm-lr --runs runs/all --out reports/transfer_fm_lr.csv` (Table VI); `... fm-lr-all --out reports/transfer_fm_lr_all_exploratory.csv` (Table S12); `scripts/fig_overview.py --fm fm_sar_lr1e-4 fm_sar_hand_lr1e-4 --fm-report transfer_fm_lr.csv` |
| V-E, Table S3, Fig. S3 | TerraMind ± HAND with GEOID-Flood training | LR pilot (validation only), then `FM_LR=5.0e-5 bash scripts/sweep_geoid.sh fm` | `slurm/paper_reports.sbatch`, `scripts/make_figures.py` |
| V-F, Table VII | train on GEOID-Flood (159 events) | `bash scripts/sweep_geoid.sh full` | `slurm/paper_reports.sbatch` |
| V-F, Table VIII, Fig. 3, Tables S4–S5, Fig. S1 | learning curve and equal-tiles control | `bash scripts/sweep_geoid.sh curve`, `... tiles` | `slurm/paper_reports.sbatch` (`learning_curve_report.py`, `volume_vs_diversity.py`), `slurm/figures.sbatch` |
| V-F | pre-registered confirmation test | (part of `tiles`) | `scripts/confirm_gain_vs_events.py` (`--metric ap` for the post-hoc AP version) |
| V-F, Table S13 | learning curve at a fixed 30 epochs (pre-registered #14) | `bash scripts/sweep_geoid.sh epochs` | `scripts/learning_curve_report.py --runs runs --curve-dir geoid_lc_ep30 --full-dir "" --name <geoid_test_flood\|geoid_test> --unseen-only <train.txt\|train_valid.txt>` |
| V-F, Table S16 | change detection (pre + post input) ± HAND | `bash scripts/sweep_geoid.sh cd` | `scripts/eval_geoid_protocol.py report` |
| V-G, Table S6 | relief gate (pre-registered, no training) | `scripts/hand_gate.py stats --data-root $D/geoid_{test,heldout}_flood --out reports/hand_stats_geoid_{test,heldout}.csv` | `scripts/hand_gate.py report --runs runs/{geoid_lc/e20,geoid} --stats ... --name geoid_proto_test --train-list train.txt` |
| V-G, Table S7 | HAND dropout (pre-registered) | `bash scripts/sweep_geoid.sh dropout`; `VARIANTS=sar_hand_md bash scripts/sweep.sh all "0 1 2"` + both evals | `scripts/dropout_report.py --runs runs/{all,geoid_lc/e20,geoid} --train-list train.txt` (`/dev/null` for `runs/all`) |
| III, Table S14 | overlap of GEOID-Flood test with train and validation; every GEOID-trained result with "unseen" = unseen by train and validation (pre-registered #11) | (no training) | `scripts/split_overlap_audit.py --test $D/geoid_test --trainval $D/geoid_trainval --write-list $D/geoid_trainval/splits/train_valid.txt`, then `TRAIN=$D/geoid_trainval/splits/train_valid.txt OUT=reports/paper_trainvalid sbatch slurm/paper_reports.sbatch`, then `scripts/trainvalid_sensitivity.py reports/paper_20261124 reports/paper_trainvalid --out reports/trainvalid_sensitivity.csv` |
| VI-D, Table S15 | labelled water above HAND thresholds (pre-registered #13, descriptive) | (no training) | `scripts/label_hand_check.py --root $D/geoid_test --root $D/geoid_test_flood --root $D/sen1floods11 --out reports/label_hand_check.csv` |
| all tables | every learning-curve, ablation, protocol, confirmation, gate and dropout table quoted | (after the runs above) | `sbatch slurm/paper_reports.sbatch` → `reports/paper_<date>/` (the paper quotes `paper_20261124`; rerunning reproduces the GEOID-trained tables to the last digit) |
| figures | | | `sbatch slurm/figures.sbatch` (PNG + PDF) |
| manuscript | | | `cd paper/jstars && tectonic -X compile main.tex` and `supplement.tex` (or pdflatex + bibtex); `bash scripts/make_arxiv.sh` for an arXiv bundle |

Compute: about 10 min per Sen1Floods11 run and 1.5–2.5 h per GEOID run on one A40; the full study is
roughly 470 training jobs, ~600 GPU-hours.
