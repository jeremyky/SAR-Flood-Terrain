#!/bin/bash
# Usage: sbatch slurm/train.sbatch configs/base.yaml [key=value ...]
#SBATCH --job-name=floodseg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000|b200|rtxpro6000   # V100 (sm_70) is not supported by the cu128 torch build
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/floodseg/logs/%x-%j.out
# Add --account=<allocation> on the command line if your default allocation is not set.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FLOODSEG_ENV:-$HOME/.conda/envs/floodseg}"
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}   # TerraMind weights (model.arch=terramind)
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this checkout's floodseg
python -m floodseg.train --config "$@"
