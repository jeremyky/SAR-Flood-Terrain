#!/bin/bash
# Usage: sbatch --account=<alloc> slurm/eval_external.sbatch <data_root> <name> <run dirs...>
#SBATCH --job-name=floodseg-external
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000|b200|rtxpro6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/floodseg/logs/%x-%j.out

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FLOODSEG_ENV:-$HOME/.conda/envs/floodseg}"
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}   # TerraMind weights (model.arch=terramind)
cd "$SLURM_SUBMIT_DIR"
DATA=$1; NAME=$2; shift 2
python scripts/eval_external.py --data-root "$DATA" --name "$NAME" "$@"
echo "EXTERNAL DONE"
