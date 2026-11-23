#!/bin/bash
# Self-contained arXiv source bundle for the manuscript and its supplement:
# .tex sources, the compiled bibliography (.bbl; arXiv does not run BibTeX reliably) and the figure PDFs
# in one flat folder, then a test build of the bundle itself.
#   bash scripts/make_arxiv.sh            # -> paper/arxiv/ and paper/arxiv.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
TECTONIC=${TECTONIC:-$HOME/.conda/envs/tex/bin/tectonic}
SRC=paper/jstars
OUT=paper/arxiv
rm -rf "$OUT" && mkdir -p "$OUT/figures"
( cd "$SRC" && "$TECTONIC" -X compile --keep-intermediates main.tex >/dev/null 2>&1 )
cp "$SRC"/main.tex "$SRC"/sec_*.tex "$SRC"/supplement.tex "$SRC"/main.bbl "$SRC"/refs.bib "$OUT"/
for f in $(grep -ho 'includegraphics\[[^]]*\]{[^}]*}' "$SRC"/*.tex | sed 's/.*{\(.*\)}/\1/' | sort -u); do
  cp "figures/$f" "$OUT/figures/"
done
sed -i 's#\\graphicspath{{../../figures/}}#\\graphicspath{{figures/}}#' "$OUT"/main.tex "$OUT"/supplement.tex
rm -f "$SRC"/main.bbl "$SRC"/main.aux "$SRC"/main.blg "$SRC"/main.log "$SRC"/main.out
# test: the bundle must build on its own
( cd "$OUT" && "$TECTONIC" -X compile main.tex >/dev/null 2>&1 && "$TECTONIC" -X compile supplement.tex >/dev/null 2>&1 ) || { echo "bundle test build FAILED"; exit 1; }
rm -f "$OUT"/*.pdf
tar -C paper -czf paper/arxiv.tar.gz arxiv
echo "bundle: $(ls "$OUT" | wc -l) files + $(ls "$OUT/figures" | wc -l) figures -> paper/arxiv.tar.gz ($(du -h paper/arxiv.tar.gz | cut -f1))"
