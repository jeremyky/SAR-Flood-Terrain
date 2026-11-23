#!/bin/bash
# Build the Zenodo deposit: code snapshot, small run outputs + reports, and checkpoints, with SHA-256 sums.
# Nothing is uploaded; the files land in $OUT for a manual upload (metadata: paper/zenodo.json).
#   sbatch slurm/archive.sbatch
#SBATCH --job-name=floodseg-archive
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/floodseg/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
S=/scratch/$USER/floodseg
TAG=$(date +%Y%m%d)
OUT=${OUT:-$S/archive_$TAG}
mkdir -p "$OUT"
commit=$(git rev-parse --short HEAD)
git diff --quiet HEAD || { echo "uncommitted changes; commit first" >&2; exit 1; }

# 1. code at the commit the paper numbers come from
git archive --format=tar.gz --prefix=sar-flood-terrain/ -o "$OUT/code_$commit.tar.gz" HEAD

# 2. every run directory without weights (resolved configs incl. git commit, training history, metrics,
#    per-chip confusion counts for every evaluation), all reports; failed attempts excluded
tar -C "$S" --exclude='*.pt' --exclude='*.failed_*' -czf "$OUT/run_outputs_$TAG.tar.gz" runs reports

# 3. checkpoints of every model behind a GEOID-Flood result (§5.2-5.8); LOEO (§5.1) models retrain in minutes
(cd "$S" && find runs/all runs/geoid runs/geoid_lc runs/geoid_tm -name best.pt -not -path '*.failed_*' | sort) \
  > "$OUT/checkpoints_$TAG.list"
tar -C "$S" -cf "$OUT/checkpoints_$TAG.tar" -T "$OUT/checkpoints_$TAG.list"

(cd "$OUT" && sha256sum code_*.tar.gz run_outputs_*.tar.gz checkpoints_*.tar checkpoints_*.list > SHA256SUMS)
du -sh "$OUT"/*
echo "wrote $OUT (commit $commit)"
