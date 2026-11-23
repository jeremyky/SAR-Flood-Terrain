#!/bin/bash
# Self-contained arXiv source bundle: one document with the paper followed by the supplementary material,
# the compiled bibliography (.bbl; arXiv does not run BibTeX reliably), refs.bib and the figure PDFs in one
# flat folder, then a test build of the bundle itself.
#   bash scripts/make_arxiv.sh            # -> paper/arxiv/, paper/arxiv.tar.gz, paper/arxiv_preview.pdf (+ paper/preprint.pdf)
set -euo pipefail
cd "$(dirname "$0")/.."
TECTONIC=${TECTONIC:-$HOME/.conda/envs/tex/bin/tectonic}
SRC=paper/jstars
OUT=paper/arxiv
rm -rf "$OUT" && mkdir -p "$OUT/figures"
( cd "$SRC" && "$TECTONIC" -X compile --keep-intermediates main.tex >/dev/null 2>&1 )
cp "$SRC"/main.tex "$SRC"/sec_*.tex "$SRC"/main.bbl "$SRC"/refs.bib "$OUT"/
for f in $(grep -ho 'includegraphics\[[^]]*\]{[^}]*}' "$SRC"/*.tex | sed 's/.*{\(.*\)}/\1/' | sort -u); do
  cp "figures/$f" "$OUT/figures/"
done
rm -f "$SRC"/main.bbl "$SRC"/main.aux "$SRC"/main.blg "$SRC"/main.log "$SRC"/main.out

# supplement body (between \sloppy and \end{document}), appended after the references as one document
sed -n '/^\\sloppy$/,/^\\end{document}$/p' "$SRC"/supplement.tex | sed '1d;$d' \
  | sed 's/^\\section\*{/\\subsection*{/' > "$OUT"/sup_body.tex
python3 - "$OUT/main.tex" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]); t = p.read_text()
t = t.replace("\\graphicspath{{../../figures/}}", "\\graphicspath{{figures/}}")
tail = r"""
\clearpage
\onecolumn
\section*{Supplementary Material}
\setcounter{table}{0}\renewcommand{\thetable}{S\arabic{table}}
\setcounter{figure}{0}\renewcommand{\thefigure}{S\arabic{figure}}
\sloppy
\input{sup_body}

\end{document}"""
t = t.rstrip()
assert t.endswith("\\end{document}")
p.write_text(t[: -len("\\end{document}")] + tail.lstrip("\n") + "\n")
PY

# test: the bundle must build on its own
( cd "$OUT" && "$TECTONIC" -X compile main.tex >/dev/null 2>&1 ) || { echo "bundle test build FAILED"; exit 1; }
pages=$(pdfinfo "$OUT/main.pdf" 2>/dev/null | awk '/Pages/ {print $2}')
mv "$OUT/main.pdf" paper/arxiv_preview.pdf
cp paper/arxiv_preview.pdf paper/preprint.pdf  # committed copy linked from the README
tar -C paper -czf paper/arxiv.tar.gz arxiv
echo "bundle: $(ls "$OUT" | wc -l) files + $(ls "$OUT/figures" | wc -l) figures -> paper/arxiv.tar.gz; preview paper/arxiv_preview.pdf (${pages:-?} pages)"
