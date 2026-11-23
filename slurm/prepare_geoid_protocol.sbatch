#!/bin/bash
# Inputs for protocol-matched GEOID scoring (scripts/eval_geoid_protocol.py): three-class labels on the
# post-event image (S1 and terrain shared with geoid_test) and the latest pre-event image for the same tiles.
# Same tiles as geoid_test (EMSR388 excluded) and geoid_heldout. Needs prepare_geoid*.sbatch first.
#SBATCH --job-name=floodseg-geoid-proto-prep
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/floodseg/logs/%x-%j.out

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FLOODSEG_ENV:-$HOME/.conda/envs/floodseg}"
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # this checkout's floodseg
S=/scratch/$USER/floodseg
HL=data/flood_events/HandLabeled
for set in test heldout; do
  if [ $set = test ]; then TREE=geoid-flood; EXCL="--exclude EMSR388"; else TREE=geoid-flood-heldout; EXCL=""; fi
  BASE=$S/data/geoid_$set
  ARGS="--geoid $S/geoid_data/$TREE --catalog $S/geoid_hf/$TREE/tile_catalog.parquet --split test $EXCL --label-mode raw --workers 16"
  RAW=$S/data/geoid_${set}_raw; PRE=$S/data/geoid_${set}_pre
  mkdir -p $RAW/$HL
  ln -sfn $BASE/$HL/S1Hand $RAW/$HL/S1Hand
  for d in $RAW $PRE; do ln -sfn $BASE/terrain $d/terrain 2>/dev/null || { mkdir -p $d; ln -sfn $BASE/terrain $d/terrain; }; done
  python scripts/prepare_geoid.py $ARGS --out $RAW
  python scripts/prepare_geoid.py $ARGS --out $PRE --image pre
done
echo "PROTO PREP DONE"
