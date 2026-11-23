#!/bin/bash
# Convert GEOID-Flood train/val tiles (all-water labels) and the held-out tree, then build terrain.
#SBATCH --job-name=floodseg-geoid-prep2
#SBATCH --partition=standard
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/floodseg/logs/%x-%j.out

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FLOODSEG_ENV:-$HOME/.conda/envs/floodseg}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this checkout's floodseg
S=/scratch/$USER/floodseg
TV=$S/data/geoid_trainval
for split in train val; do
  name=$split; [ $split = val ] && name=valid
  python scripts/prepare_geoid.py --geoid $S/geoid_data/geoid-flood --catalog $S/geoid_hf/geoid-flood/tile_catalog.parquet \
    --split $split --out $TV --split-name $name --workers 32
done
python scripts/build_terrain.py --root $TV --workers 32 > $S/logs/geoid_trainval_terrain.log
HO=$S/data/geoid_heldout
python scripts/prepare_geoid.py --geoid $S/geoid_data/geoid-flood-heldout --catalog $S/geoid_hf/geoid-flood-heldout/tile_catalog.parquet \
  --split test --out $HO --workers 32
python scripts/build_terrain.py --root $HO --workers 32 > $S/logs/geoid_heldout_terrain.log
# flood-only labels for the held-out events, sharing S1 and terrain
HF=$S/data/geoid_heldout_flood
mkdir -p $HF/data/flood_events/HandLabeled
ln -sfn $HO/data/flood_events/HandLabeled/S1Hand $HF/data/flood_events/HandLabeled/S1Hand
ln -sfn $HO/terrain $HF/terrain
python scripts/prepare_geoid.py --geoid $S/geoid_data/geoid-flood-heldout --catalog $S/geoid_hf/geoid-flood-heldout/tile_catalog.parquet \
  --split test --out $HF --label-mode flood --workers 32
echo "PREP2 DONE"
