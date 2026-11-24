"""arXiv v2: the CVPR two-column layout, de-branded, with authors and back matter, derived from ../paper.tex.

    python3 make_arxiv_v2.py

Reuses ../cvpr/make_cvpr.py (executed in-process with its output directory redirected here, so the CVPR build
products in ../cvpr are never touched), then:
  - camera-ready mode (\\usepackage[pagenumbers]{...}): no line-number ruler, no review banner, no anonymous byline
  - style file copied from the CVPR kit as lacepreprint.sty, comments stripped and every 'cvpr'/'CVPR' token renamed
  - \\confName/\\confYear removed; real authors/affiliation/contact; no pagebackref
  - back matter as in arXiv v1 (contributions, funding, data availability, conflicts, AI-assistance statement)
  - the appendix runs on continuously (no page reset, no repeated title)
  - content = the CVPR version (CVPR-only rewrites in, omit blocks out)
Outputs here: paper_v2.tex, paper_v2.pdf, paper_v2.bbl, lace.bib, lacepreprint.sty, ieeenat_fullname.bst, figs/,
abstract_for_form.txt, lace_arxiv_v2.tar.gz (upload this), build_report.txt.
Guard: build fails if 'cvpr' (any case) appears in any file of the tarball or in the PDF text.
"""
import re, sys, pathlib, shutil, subprocess, tarfile

here = pathlib.Path(__file__).resolve().parent
results = here.parent
cvpr_dir = results / "cvpr"

# --- 1. generate the CVPR source into this directory -------------------------------------------
code = (cvpr_dir / "make_cvpr.py").read_text()
subs = [
    ('here = pathlib.Path(__file__).resolve().parent', f'here = pathlib.Path({str(here)!r})'),
    ('root = here.parent ', f'root = pathlib.Path({str(results)!r}) '),
    ('here / "kit" / f', f'pathlib.Path({str(cvpr_dir / "kit")!r}) / f'),
    ('dst = here / "cvpr_full.tex"', 'dst = here / "_cvpr_generated.tex"'),
]
for a, b in subs:
    assert code.count(a) == 1, f"make_cvpr.py changed: cannot redirect {a!r}"
    code = code.replace(a, b)
argv = sys.argv
sys.argv = ["make_cvpr.py", "--tex-only"]
try:
    exec(compile(code, str(cvpr_dir / "make_cvpr.py"), "exec"), {"__name__": "make_cvpr", "__file__": str(cvpr_dir / "make_cvpr.py")})
except SystemExit as e:
    assert not e.code, f"make_cvpr.py exited {e.code}"
sys.argv = argv
gen = here / "_cvpr_generated.tex"
doc = gen.read_text()
gen.unlink()

# --- 2. back matter from paper.tex -------------------------------------------------------------
P = (results / "paper.tex").read_text()
def braced(text, cmd):
    i = text.index(cmd + "{") + len(cmd) + 1
    depth, j = 1, i
    while depth:
        c = text[j]; depth += (c == "{") - (c == "}"); j += 1
    return text[i:j-1].strip()
contrib, funding, data = braced(P, r"\authorcontributions"), braced(P, r"\funding"), braced(P, r"\dataavailability")
coi = braced(P, r"\conflictsofinterest")

# --- 3. de-brand -------------------------------------------------------------------------------
def rep(a, b, n=1):
    global doc
    assert doc.count(a) == n, f"expected {n}x {a!r}, found {doc.count(a)}"
    doc = doc.replace(a, b)
rep(r"\usepackage[review]{cvpr}", r"\usepackage[pagenumbers]{lacepreprint}")
rep("\\def\\confName{CVPR}\n", "")
rep("\\def\\confYear{2027}\n", "")
doc = doc.replace("cvprblue", "linkblue")
rep(r"\usepackage[pagebackref,breaklinks,colorlinks,allcolors=linkblue]{hyperref}",
    r"\usepackage[breaklinks,colorlinks,allcolors=linkblue,pdfusetitle]{hyperref}")
rep(r"\author{Anonymous CVPR submission}",
    r"""\author{Thomas Pitts\thanks{Corresponding author. ORCID \href{https://orcid.org/0009-0000-8884-2381}{0009-0000-8884-2381}.}\qquad Kunqi Li\qquad Bin Liang\\
Department of Data Science, University of Technology Sydney\\
{\tt\small thomas.pitts@uts.edu.au}}
\hypersetup{pdfauthor={Thomas Pitts, Kunqi Li, Bin Liang}}""")
rep("\\label{cvpr:body-end}\n", "")
# make_cvpr.py emits the statement BEFORE the references since 2026-11-13; lift it (and its CVPR comment line) out
m = re.search(r"(?:% CVPR[^\n]*\n)?\\section\*\{LLM Usage Statement\}\n(.*?)\n\n(?=\{\n\\small\n\\bibliographystyle)", doc, flags=re.S)
assert m, "LLM Usage Statement block not found"
ai = m.group(1).strip()
doc = doc[:m.start()] + doc[m.end():]
back = (f"\\section*{{Author Contributions}}\n{contrib}\n\n\\section*{{Funding}}\n{funding}\n\n"
        f"\\section*{{Data Availability}}\n{data}\n\n\\section*{{Conflicts of Interest}}\n{coi}\n\n"
        f"\\section*{{Use of AI Assistance}}\n{ai}\n\n")
rep("{\n\\small\n\\bibliographystyle{ieeenat_fullname}", back + "{\n\\small\n\\bibliographystyle{ieeenat_fullname}")
rep("\\clearpage\n\\setcounter{page}{1}\n\\newpage\\twocolumn[\\centering\\Large\\textbf{\\thetitle}\\\\\\vspace{0.5em}Appendix\\\\\\vspace{1.0em}]\n\\appendix\n",
    "\\clearpage\n\\appendix\n")
assert "Supplementary" not in doc, "stray Supplementary wording: the paper says Appendix throughout"
doc = re.sub(r"\n{3,}", "\n\n", doc)
bad = sorted({c for c in doc if ord(c) > 127})
assert not bad, f"non-ASCII would break pdflatex on arXiv: {bad}"
(here / "paper_v2.tex").write_text(doc)

# style: kit file verbatim, renamed
sty = (cvpr_dir / "kit" / "cvpr.sty").read_text()   # verbatim (stripping its comments breaks it); tokens renamed below
sty = sty.replace(r"\ProvidesPackage{cvpr}[2027 LaTeX class for IEEE CVPR]", r"\ProvidesPackage{lacepreprint}[2026/11/10 two-column preprint style]")
sty = sty.replace("CVPR 8.5 x 11-Inch Proceedings Style `cvpr.sty'.", "two-column preprint style.")
sty = sty.replace("CVPR requires", "this style expects").replace("CVPR uses", "this style uses")
sty = sty.replace("cvpr", "lpp").replace("CVPR", "")
(here / "lacepreprint.sty").write_text(sty)
(here / "cvpr.sty").unlink(missing_ok=True)   # copied by make_cvpr's step 4

# --- 4. build ----------------------------------------------------------------------------------
r = subprocess.run(["tectonic", "--keep-logs", "--keep-intermediates", "paper_v2.tex"], cwd=here, capture_output=True, text=True)
if r.returncode:
    (here / "build_report.txt").write_text("BUILD FAILED\n" + r.stdout[-4000:] + r.stderr[-4000:])
    print(r.stderr[-3000:]); sys.exit(1)

# --- 5. tarball (tex, bbl, bib, sty, bst, figures actually used) + guards -----------------------
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", doc)
members = ["paper_v2.tex", "paper_v2.bbl", "lace.bib", "lacepreprint.sty", "ieeenat_fullname.bst"]
for g in figs:
    s = results / g
    if not s.exists():
        cands = list(s.parent.glob(s.name + ".*")); assert cands, f"figure missing: {g}"; s = cands[0]
    rel = s.relative_to(results)
    (here / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy(s, here / rel)
    members.append(str(rel))
members = sorted(set(members))
tex_hits = [f for f in members if not f.endswith((".pdf", ".bib", ".bbl")) and re.search(r"cvpr", (here / f).read_text(errors="replace"), re.I)]
pdftext = subprocess.run(["pdftotext", "paper_v2.pdf", "-"], cwd=here, capture_output=True, text=True).stdout
pdf_hits = sorted(h for h in set(re.findall(r".{0,40}cvpr.{0,40}", pdftext, re.I))
                  if not re.search(r"Recognition \(CVPR\)|\(CVPR\),? pages", h))   # cited papers' venues are fine
bbl_hits = sorted(set(re.findall(r"[^\n]*cvpr[^\n]*", (here / "paper_v2.bbl").read_text(), re.I)))
fig_hits = [f for f in members if f.endswith(".pdf") and re.search(rb"cvpr", (here / f).read_bytes(), re.I)]
meta = subprocess.run(["pdfinfo", "paper_v2.pdf"], cwd=here, capture_output=True, text=True).stdout
leaks = [w for w in ("mdpi", "anonymous", "review copy", "confidential") if w in pdftext.lower()]
with tarfile.open(here / "lace_arxiv_v2.tar.gz", "w:gz") as tf:
    for f in members:
        tf.add(here / f, arcname=f)

log = (here / "paper_v2.log").read_text(errors="replace")
undef = sorted(set(re.findall(r"(?:Reference|Citation) `([^']+)' on page \d+ undefined", log)))
overfull = [float(w) for w in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log) if float(w) > 2]
pages = re.search(r"Pages:\s+(\d+)", meta).group(1)
report = [
    f"pages: {pages}",
    f"'cvpr' in source files: {tex_hits or 'none'}; in PDF text: {pdf_hits or 'none'}; in figure PDFs: {fig_hits or 'none'}",
    f"other leaks in PDF text: {leaks or 'none'}",
    f"bbl lines with cvpr (should be cited venues only): {len(bbl_hits)}" + "".join("\n  " + h.strip() for h in bbl_hits),
    f"undefined refs/cites: {undef or 'none'}; overfull hboxes >2pt: {len(overfull)}",
    f"tarball: lace_arxiv_v2.tar.gz ({len(members)} files): " + ", ".join(members),
]
(here / "build_report.txt").write_text("\n".join(report) + "\n")
print("\n".join(report))
sys.exit(1 if (tex_hits or pdf_hits or fig_hits or undef) else 0)
