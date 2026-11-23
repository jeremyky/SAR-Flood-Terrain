#!/bin/bash
# Export every SLURM submission of this project (training, evaluation, data prep, reports) to
# experiments/slurm_history.tsv: job id, submit time, final state, exact submit line.
# A job belongs to the project if it ran one of the slurm/*.sbatch files here, or its name starts with floodseg-.
# sacct limits the date range per query, so it is queried month by month.
set -euo pipefail
cd "$(dirname "$0")/.."
files=$(ls slurm/*.sbatch | sed 's/[.]/[.]/g' | paste -sd'|')
start=${SINCE:-2026-09-01}
tmp=$(mktemp)
printf 'job_id\tsubmitted\tstate\tsubmit_line\n' > "$tmp"
while [[ "$start" < "$(date -d tomorrow +%F)" ]]; do
  end=$(date -d "$start +1 month" +%F)
  sacct -u "$USER" -S "$start" -E "$end" -X -n -P -o JobID,Submit,State,JobName,SubmitLine |
    awk -F'|' -v re="($files)" 'BEGIN{OFS="\t"} $5 ~ re || $4 ~ /^floodseg-/ {print $1, $2, $3, $5}' >> "$tmp"
  start=$end
done
awk '!seen[$1]++' "$tmp" > experiments/slurm_history.tsv && rm -f "$tmp"
echo "$(($(wc -l < experiments/slurm_history.tsv) - 1)) jobs -> experiments/slurm_history.tsv"
