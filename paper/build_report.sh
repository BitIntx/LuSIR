#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$paper_dir"

command -v pandoc >/dev/null || {
  echo "pandoc is required to build the technical report." >&2
  exit 1
}
command -v pdflatex >/dev/null || {
  echo "pdflatex is required to build the technical report." >&2
  exit 1
}

pandoc TECHNICAL_REPORT.md \
  --from=gfm \
  --standalone \
  --toc \
  --number-sections \
  --shift-heading-level-by=-1 \
  --resource-path=".:.." \
  --pdf-engine=pdflatex \
  --variable=geometry:margin=0.8in \
  --variable=fontsize:10pt \
  --variable=colorlinks:true \
  --variable=linkcolor:blue \
  --variable=urlcolor:blue \
  --output=main.tex

pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
mv main.pdf sr_diffusion_report.pdf
rm -f main.aux main.log main.out main.toc
