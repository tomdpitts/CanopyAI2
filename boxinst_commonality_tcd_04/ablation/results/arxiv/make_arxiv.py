"""Derive a plain article-class arXiv source (paper_arxiv.tex) from ../paper.tex.
Nothing from the MDPI class survives: front matter is re-expressed with \title/\author/abstract,
the body is carried over verbatim (comments stripped, since arXiv publishes source), the MDPI
back-matter commands are replaced by plain starred sections, appendices get real titles, and the
bibliography uses natbib/unsrtnat (numeric, order of citation, as in the MDPI draft)."""
import re, pathlib
here = pathlib.Path(__file__).resolve().parent
src = here.parent / "paper.tex"
dst = here / "paper_arxiv.tex"
t = src.read_text()

# 0. strip comments (full-comment lines dropped; trailing comments truncated, keeping a bare % guard)
out = []
for line in t.split("\n"):
    m = re.search(r"(?<!\\)%", line)
    if m:
        head = line[:m.start()]
        if head.strip() == "":
            continue
        line = head.rstrip() + "%"
    out.append(line)
t = "\n".join(out)
t = re.sub(r"\n{3,}", "\n\n", t)

def braced(text, cmd):
    """Return the brace-balanced argument of \\cmd{...}."""
    i = text.index(cmd + "{") + len(cmd) + 1
    depth, j = 1, i
    while depth:
        c = text[j]
        depth += (c == "{") - (c == "}")
        j += 1
    return text[i:j-1].strip()

title    = braced(t, r"\Title")
abstract = braced(t, r"\abstract")
keywords = braced(t, r"\keyword")

# 1. body: from \begin{document} to the MDPI back matter
body = t[t.index(r"\begin{document}") + len(r"\begin{document}"): t.index(r"\authorcontributions{")]
body = body.replace("prompts such as points/boxes/masks", "prompts such as points, boxes or masks")
body = body.replace("\\begin{tabular}{lcccc}\n\\toprule\nMethod & P & R & $F_1$ & max\\,R", "\\small\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{lcccc}\n\\toprule\nMethod & P & R & $F_1$ & max\\,R")
body = body.replace("von Mises--Fisher components (cite)", r"von Mises--Fisher components~\cite{banerjee2005vmf}")

# 2. appendices: from \appendix to the reference block, with real section titles
app = t[t.index("\\appendix\n") + len("\\appendix\n"): t.index(r"\begin{adjustwidth}")]
assert r"\appendixstart" not in app and r"\appendixtitles" not in app
app_titles = ["Supplementary Analyses", "Scoring Protocol and Baseline Configurations"]
for title_i in app_titles:
    app = app.replace(r"\section[\appendixname~\thesection]{}", r"\section{%s}" % title_i, 1)
assert r"\appendixname" not in app

abbreviations = r"""\section*{Abbreviations}
\noindent AP, average precision; CHM, canopy height model; CNN, convolutional neural network; DINO, self-distillation with no labels; DSM, digital surface model; EM, expectation-maximisation; FAO, Food and Agriculture Organization of the United Nations; FPN, feature pyramid network; GSD, ground sample distance; GT, ground truth; IoU, intersection over union; ITC, individual tree crown; NEON, National Ecological Observatory Network; NMS, non-maximum suppression; OAM, OpenAerialMap; PCA, principal component analysis; RPN, region proposal network; RS, remote sensing; SAM, Segment Anything Model; SOTA, state of the art; SSL, self-supervised learning; TCD, tree crown detection; TCIS, tree crown instance segmentation; TCSS, tree crown semantic segmentation; UAV, unmanned aerial vehicle; ViT, vision transformer; WWF, World Wide Fund for Nature.
"""

preamble = r"""\ifdefined\XeTeXversion\else\pdfoutput=1\fi
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=25mm]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs,array,tabularx,multirow,threeparttable}
\usepackage[table]{xcolor}
\usepackage{algorithm,algpseudocode}
\usepackage{float}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{microtype}
\usepackage{url}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[pdfusetitle,colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\setlength{\parskip}{3pt}
\setlength{\emergencystretch}{2em}
\providecommand{\cmark}{$\bullet$}
\providecommand{\omark}{$\circ$}
\providecommand{\xmark}{$\times$}

\title{%s}
\author{Thomas Pitts\thanks{Corresponding author: \texttt{thomas.pitts@uts.edu.au}}, \quad Kunqi Li, \quad Bin Liang\\[6pt]
\normalsize Department of Data Science, University of Technology Sydney}
\date{}
\hypersetup{pdfauthor={Thomas Pitts, Kunqi Li, Bin Liang}}
""" % title

front = r"""\begin{document}
\maketitle

\begin{abstract}
%s
\end{abstract}

\noindent\textbf{Keywords:} %s
""" % (abstract, keywords)

contributions = braced(t, r"\authorcontributions")
funding = braced(t, r"\funding")
dataavail = braced(t, r"\dataavailability")
back = r"""
\section*{Author Contributions}
%s

\section*{Funding}
%s

\section*{Data Availability}
%s

\section*{Conflicts of Interest}
The authors declare no conflicts of interest.

%s
\bibliographystyle{unsrtnat}
\bibliography{lace}

\appendix
%s
\end{document}
""" % (contributions, funding, dataavail, abbreviations, app)

final = preamble + front + body.rstrip() + "\n" + back
final = final.replace(r"\texttt{edge\_band\_buffer\_percentage}", r"\texttt{edge\_band\_}\allowbreak\texttt{buffer\_}\allowbreak\texttt{percentage}")
final = final.replace(r"\begin{table}[H]", r"\begin{table}[!htbp]")  # let text fill the page instead of [H] gaps
final = re.sub(r"\n{3,}", "\n\n", final)
dst.write_text(final)
# bib: copy alongside, minus the %% header comments that mention the MDPI style
bib = (here.parent / "lace.bib").read_text()
(here / "lace.bib").write_text("\n".join(l for l in bib.split("\n") if not l.startswith("%%")).lstrip("\n"))

leak = [w for w in ("mdpi", "MDPI", "Definitions/", r"\hreflink", r"\pubvolume", r"\PublishersNote", "adjustwidth",
                    r"\extralength", r"\reftitle", r"\appendixtitles", r"\Title", r"\Author", r"\corres", r"\address",
                    r"\keyword", r"\abstract{", r"\orcid", r"\isPreprints", "Licensee") if w in final]
print("wrote", dst, len(final.split("\n")), "lines; leaks:", leak or "none")

# plain-text abstract for the arXiv metadata form (TeX math kept, LaTeX macros stripped)
a = abstract.replace("``", '"').replace("''", '"').replace("~", " ").replace(r"$\sim$", "~").replace(r"\%", "%").replace(r"\&", "&")
a = re.sub(r"\\cite\{[^}]*\}", "", a)
a = re.sub(r"\\(emph|textbf|textit)\{([^}]*)\}", r"\2", a)
a = re.sub(r"\s+", " ", a).strip()
(here / "abstract_for_form.txt").write_text(a + "\n")
