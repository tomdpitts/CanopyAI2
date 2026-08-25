# Join hard-wrapped LaTeX paragraphs into single lines.
# Breaks kept at: blank lines, lines ending in \\, and structural commands.
function flush() { if (buf != "") { print buf; buf = "" } }
BEGIN { buf = "" }
{
  line = $0
  sub(/[ \t]+$/, "", line)
  if (line ~ /^[ \t]*$/)                        { flush(); print ""; next }
  if (line ~ /^[ \t]*%/)                        { flush(); print line; next }
  if (line ~ /^[ \t]*\\(begin|end|section|subsection|subsubsection|paragraph|label|caption|item|newcolumntype|toprule|midrule|bottomrule|hline|documentclass|usepackage|maketitle|bibliography)/) \
                                                { flush(); print line; next }
  if (buf == "") buf = line
  else           buf = buf " " line
  if (line ~ /\\\\[ \t]*$/ || line ~ /\\\\ ?\[[^]]*\][ \t]*$/) flush()
}
END { flush() }
