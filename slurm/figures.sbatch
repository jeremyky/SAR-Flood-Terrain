#!/bin/bash
# All paper figures (PNG + PDF) into figures/.
#   sbatch slurm/figures.sbatch
#SBATCH --job-name=floodseg-figures
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/floodseg/logs/%x-%j.out
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FLOODSEG_ENV:-$HOME/.conda/envs/floodseg}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src
S=/scratch/$USER/floodseg
python scripts/make_figures.py --runs $S/runs --scratch $S --out figures \
  --train-list $S/data/geoid_trainval/splits/train.txt
python scripts/example_tiles.py --data-root $S/data/geoid_test \
  --base-run $S/runs/all/sar_only/seed0 --hand-run $S/runs/all/sar_hand/seed0 \
  --chips EMSR848_16-21 EMSR501_1-18 EMSR280_1-0 --out figures/fig_example_tiles.png
python scripts/fig_overview.py --scratch $S --out figures/fig_overview.png \
  --fm fm_sar_lr1e-4 fm_sar_hand_lr1e-4 --fm-report transfer_fm_lr.csv   # learning rate selected in #12; needs reports/official_split.csv, matched_split.csv, transfer_headline.csv, ablation_all.csv, transfer_fm_lr.csv
ls -la figures
