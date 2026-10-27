"""Put the canonical Table 1 front and centre in the planning docs, and banner every other doc that quotes 439 numbers.

Idempotent: blocks sit between HTML-comment markers and are replaced on re-run (so re-run after
build_table1_current.py whenever TABLE1_CURRENT.md changes). Inserts only; never deletes existing text.

  .venv/bin/python -m boxinst_commonality_tcd_04.results_439.mark_superseded
"""
import os, re

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(H)
CANON = os.path.join(H, "results_439", "TABLE1_CURRENT.md")
B0, B1 = "<!-- TABLE1_CURRENT:begin (auto, results_439/mark_superseded.py) -->", "<!-- TABLE1_CURRENT:end -->"

FRONT = ["README.md", "box_supervised_baselines/PLAN.md", "native_raster/NATIVE_RASTER.md", "PROTOCOL_439.md"]


def tables():
    t = open(CANON).read()
    a = t.index("## Table 1"); b = t.index("## Sources")
    return t[a:b].strip().replace("## ", "#### ")


def front_block(fp):
    rel = os.path.relpath(CANON, os.path.dirname(fp))
    return (f"{B0}\n> **CURRENT HEADLINE NUMBERS (2026-10-27)** — single source of truth: [`{os.path.basename(CANON)}`]({rel}) "
            f"(generated + asserted by `results_439/build_table1_current.py`). **Any 439 / sparse236 test number elsewhere in this "
            f"file that disagrees with this box is SUPERSEDED** (512 raster, pre-seam-fix baselines, or the 8 px masker).\n\n"
            f"{tables()}\n{B1}\n")


def banner(fp):
    rel = os.path.relpath(CANON, os.path.dirname(fp))
    return (f"{B0}\n> ⚠️ **SUPERSEDED 439 / sparse236 NUMBERS (marked 2026-10-27).** Current headline table: [`{os.path.basename(CANON)}`]({rel}). "
            f"Test-set numbers in this file are a historical experiment record (512 raster, pre-seam-fix baselines, 8 px masker or "
            f"earlier models) and must not be quoted unless they match that table.\n{B1}\n")


def put(fp, block):
    s = open(fp).read()
    if B0 in s:
        s = re.sub(re.escape(B0) + r".*?" + re.escape(B1) + r"\n", lambda _: block, s, count=1, flags=re.S)
    else:
        m = re.search(r"^# .*\n", s, flags=re.M)
        i = m.end() if m else 0
        s = s[:i] + "\n" + block + s[i:]
    open(fp, "w").write(s)


def main():
    docs = []
    for root in (H, os.path.join(REPO, "tcd04_semseg")):
        for d, dn, fn in os.walk(root):
            dn[:] = [x for x in dn if x not in ("_sam3_repo", "__pycache__", "node_modules") and not x.startswith(".venv")]
            docs += [os.path.join(d, f) for f in fn if f.endswith(".md") and "439" in open(os.path.join(d, f), errors="ignore").read()]
    front = {os.path.join(H, f) for f in FRONT}
    for fp in sorted(docs):
        if fp == CANON:
            continue
        put(fp, front_block(fp) if fp in front else banner(fp))
        print(("FRONT  " if fp in front else "BANNER ") + os.path.relpath(fp, REPO), flush=True)


if __name__ == "__main__":
    main()
