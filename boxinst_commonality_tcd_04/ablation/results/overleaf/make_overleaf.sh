#!/bin/sh
# Package the CURRENT paper.tex (MDPI Remote Sensing) as a FLAT Overleaf upload for an existing project (2026-11-27):
#   ../paper.tex -> main.tex, lace.bib, every figure PDF main.tex includes, and the Definitions/ files (mdpi.cls, .bst,
#   journalnames.tex, logos), all at one level -> overleaf/upload/ (no subfolders, no zip).
# mdpi.cls hard-codes "Definitions/" paths, so the copied class and main.tex have that prefix stripped; the repo's own
# Definitions/ and paper.tex are untouched. Overleaf: Upload > select all files in overleaf/upload/ (overwrite).
# Output verified pixel-identical to paper.pdf under tectonic (XeLaTeX); set Overleaf's compiler to XeLaTeX to match exactly.
# Manual only (no on-save hook). Re-run after every paper.tex / figure / bib change.
set -e
cd "$(dirname "$0")"
here=$(pwd); cd ..
out=$(mktemp -d)/upload
mkdir -p "$out"
sed -e 's#\(includegraphics\[[^]]*\]{\)figs/#\1#g' -e 's#{Definitions/mdpi}#{mdpi}#' paper.tex > "$out/main.tex"
cp lace.bib "$out/"
for f in Definitions/*; do
  case "$f" in *.eps) continue ;; esac  # class is patched to load the .pdf logos; .eps unused
  cp "$f" "$out/"
done
sed -i '' 's#Definitions/##g' "$out/mdpi.cls"
grep -v '^[[:space:]]*%' paper.tex | grep -o 'includegraphics\[[^]]*\]{figs/[^}]*}' | sed 's/.*{\(.*\)}/\1/' | sort -u | while read f; do
  cp "$f" "$out/$(basename "$f")"
done
rm -rf "$here/upload" "$here/paper_overleaf.zip" && cp -R "$out" "$here/upload"
ls -1 "$here/upload"
