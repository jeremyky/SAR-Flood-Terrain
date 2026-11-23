#!/bin/bash
# Usage: sbatch --account=<alloc> slurm/eval_geoid_protocol.sbatch <test|heldout> <run dirs...>
#SBATCH --job-name=floodseg-geoid-proto
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|a6000|b200|rtxpro6000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/floodseg/logs/%x-%j.out

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FLOODSEG_ENV:-$HOME/.conda/envs/floodseg}"
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}   # TerraMind weights (model.arch=terramind)
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this checkout's floodseg
SET=$1; shift
D=/scratch/$USER/floodseg/data
python scripts/eval_geoid_protocol.py eval --post $D/geoid_${SET}_raw --pre $D/geoid_${SET}_pre --name geoid_proto_$SET "$@"
echo "PROTO EVAL DONE"
