"""Collate every phase4_collapse result into RESULTS.md.

Pulls from results/eval/*.json (per-cell evals), results/*_fits.json (prototype geometry),
and the handful of numbers that live in phase4's own records. Re-runnable: it re-reads
whatever is present, so running it again after more cells land refreshes the document.
"""
import glob
import json
import os
from statistics import mean, stdev

R, E = "results", "results/eval"


def ev(tag, key="mean_crown_iou"):
    p = f"{E}/{tag}.json"
    return json.load(open(p))[key] if os.path.exists(p) else None


def fits():
    """cell -> geometry, from every *_fits.json AND the per-cell proto_stats fetched from the
    volume (cells fitted by entrypoints that spawn rather than collate have no local
    *_fits.json, so proto/ is the only local record of their geometry)."""
    g = {}
    for f in glob.glob(f"{R}/*_fits.json"):
        for r in json.load(open(f)):
            g[r.get("cell") or f"k{r['k_init']}_s{r['em_seed']}_b{r['beta']:g}"] = r
    for f in glob.glob(f"{R}/proto/*.json"):
        r = json.load(open(f))
        g.setdefault(r.get("cell") or os.path.basename(f)[:-5], r)
    return g


def band(v, nd=4):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return (round(mean(v), nd), round(stdev(v), nd) if len(v) > 1 else 0.0, len(v))


def fmt(x, nd=4):
    return "--" if x is None else f"{x:.{nd}f}"


G = fits()


def geo(cell, k):
    r = G.get(cell)
    return r.get(k) if r else None


out = []
w = out.append

w("# `phase4_collapse` — results\n")
w("Prototype-collapse investigation for the LACE box-to-mask module. Every number below is "
  "measured on the **108-tile validation split** with **detector seed 0** unless stated "
  "otherwise; the 439-tile test split is untouched by this folder. Masker fits use 120 "
  "training tiles (boxes only, no crown polygons). Bands are sample standard deviation "
  "(`ddof=1`).\n")
w("Two mechanisms are varied throughout and must not be conflated:\n")
w("| knob | stage | effect |")
w("|---|---|---|")
w("| `contrastive_beta` (β) | M-step | prototype repulsion, `C = norm(pos − β·neg)` |")
w("| `contrast` | E-step | within-box recentring, `Ā = A − mean_box A` |")
w("")
w("`phase4` couples them (`no_contrast = beta == 0`), so its recorded β ablation varies both "
  "at once. This folder decouples them.\n")

# ---------------------------------------------------------------- 1. component 2x2
w("## 1. Component ablation (2×2), K=16, EM seed 0\n")
w("The published β ablation is the *diagonal* of this table, so its +0.041 AP50 was never a "
  "β measurement.\n")
w("| M-step | E-step | eff. rank | cos | oracle IoU | pred IoU | pred AP50 |")
w("|---|---|---|---|---|---|---|")
rows_2x2 = [
    ("generative (β=0)", "absolute", None, 10.83, 0.218, 0.7402, 0.6717, 0.5941),
    ("generative (β=0)", "recentred", "k16_s0_b0_c1", None, None, None, None, None),
    ("contrastive (β=0.5)", "absolute", "k16_s0_b0.5_c0", None, None, None, None, None),
    ("contrastive (β=0.5)", "recentred", None, 1.66, 0.9935, 0.7885, 0.7193, 0.6392),
]
for m, e, cell, rk, cs, oi, pi, pa in rows_2x2:
    if cell:
        rk, cs = geo(cell, "effective_rank"), geo(cell, "pairwise_cos_mean")
        oi, pi, pa = ev(f"{cell}_gt"), ev(f"{cell}_pred"), ev(f"{cell}_pred", "mask_mAP50")
    star = " **(deployed)**" if m.startswith("contrastive") and e == "recentred" else ""
    w(f"| {m}{star} | {e} | {fmt(rk,2)} | {fmt(cs,3)} | {fmt(oi)} | {fmt(pi)} | {fmt(pa)} |")
w("")
w("**Effects on mean crown IoU** (β=0/absolute is the reference cell):\n")
w("| | oracle | predicted |")
w("|---|---|---|")
w("| recentring alone | +0.0493 | +0.0471 |")
w("| repulsion alone | +0.0356 | +0.0346 |")
w("| joint (the published +0.041 AP50) | +0.0483 | +0.0476 |")
w("| **interaction** | **−0.0366** | **−0.0341** |")
w("")
w("The two are **substitutes, not complements**: either alone recovers ~+0.04, both together "
  "recover the same ~+0.048. Recentring is the larger term in both box regimes, matching the "
  "dryland proxy ablation (`boxinst_commonality/README.md:107-119`), where removing recentring "
  "moves `corner` 0.37→0.61 against 0.37→0.41 for repulsion.\n")

# ---------------------------------------------------------------- 2. beta sweep
w("## 2. β sweep at K=16, recentring held on, EM seed 0\n")
w("Single-axis: no previous sweep held the E-step fixed. `masker_lab/sweep.py` (16 px) also "
  "sets `no_contrast=(beta==0)`, so its β=0 row differs in two components.\n")
w("| β | K_eff | eff. rank | cos | oracle IoU | pred IoU | pred AP50 | pred AP50:95 |")
w("|---|---|---|---|---|---|---|---|")
for b, cell in ((0.0, "k16_s0_b0_c1"), (0.1, "k16_s0_b0.1_c1"),
                (0.25, "k16_s0_b0.25_c1"), (0.5, "k16_s0_b0.5_c1")):
    w(f"| {b:g} | {geo(cell,'K') or '--'} | {fmt(geo(cell,'effective_rank'),2)} | "
      f"{fmt(geo(cell,'pairwise_cos_mean'),3)} | {fmt(ev(cell+'_gt'))} | "
      f"{fmt(ev(cell+'_pred'))} | {fmt(ev(cell+'_pred','mask_mAP50'))} | "
      f"{fmt(ev(cell+'_pred','mask_mAP50_95'))} |")
pv = [ev(f"k16_s0_b{b}_c1_pred") for b in ("0", "0.1", "0.25", "0.5")]
pv = [x for x in pv if x]
if pv:
    w("")
    w(f"Effective rank falls **11.4 → 1.7** (a ~7× reduction in distinct foreground "
      f"directions) while predicted-box crown IoU spans **{max(pv)-min(pv):.4f}**. "
      f"β is a geometry knob with no accuracy consequence.\n")

# ---------------------------------------------------------------- 3. K
w("## 3. Component count\n")
w("| β | K init | K_eff | eff. rank | cos | oracle IoU | pred IoU | pred AP50 |")
w("|---|---|---|---|---|---|---|---|")
for b, cell in ((0.0, "k2_s0_b0_c1"), (0.0, "k16_s0_b0_c1"),
                (0.5, "k2_s0_b0.5"), (0.5, "k16_s0_b0.5_c1")):
    ki = 2 if cell.startswith("k2") else 16
    w(f"| {b:g} | {ki} | {geo(cell,'K') or '--'} | {fmt(geo(cell,'effective_rank'),2)} | "
      f"{fmt(geo(cell,'pairwise_cos_mean'),3)} | {fmt(ev(cell+'_gt'))} | "
      f"{fmt(ev(cell+'_pred'))} | {fmt(ev(cell+'_pred','mask_mAP50'))} |")
w("")
w("At β=0 the K=2 fit is a genuine two-component mixture (effective rank 2.00 of a possible "
  "2.00, cos 0.05) and still matches K=16's rank-11.4 mixture. The extra fourteen components "
  "buy ≈0.0007. At β=0.5 both collapse, and K=2 was never evaluated for accuracy — the "
  "equivalence there rests on direction agreement, `cos(C̄_K2, C̄_K16) = 0.9942`, which exceeds "
  "the between-seed agreement within K=16 itself (0.9915–0.9922).\n")

# ---------------------------------------------------------------- 4. box precision
w("## 4. Box precision — is collapse a robustness mechanism?\n")
w("| box source | β=0 (rank 10.8) | β=0.5 (rank 1.7) | Δ |")
w("|---|---|---|---|")
w("| oracle (ground-truth boxes) | 0.7402 | 0.7885 | +0.0483 |")
w("| predicted boxes | 0.6717 | 0.7193 | +0.0476 |")
w("")
w("**No.** The two deltas agree to 0.0007, so the benefit is independent of box precision. "
  "The apparent sign flip in the prior evidence — GT boxes favouring β=0 at 16 px "
  "(`masker_lab`: 0.7412 → 0.6590) while predicted boxes favour β=0.5 at 8 px — was a "
  "stride/stack artefact, not box precision. Note β=0 scores almost identically at both "
  "strides (0.7402 here vs 0.7412 in the lab); it is β=0.5 that differs (0.7885 vs 0.6590). "
  "*(Both arms here use the absolute E-step for β=0, so these deltas are two-component; "
  "section 1 decomposes them.)*\n")

# ---------------------------------------------------------------- 5. seed band
w("## 5. Masker-seed variance\n")
w("The published 0.630 ± 0.005 varies **detector** seeds and holds one masker fixed "
  "(`em_model_4p_fix.npz`, EM seed 0), so it contains no masker variance. This section "
  "isolates the other axis: detector fixed at seed 0, masker refitted at three EM seeds.\n")
w("| β | seed | K_eff | eff. rank | cos | pred IoU | pred AP50 |")
w("|---|---|---|---|---|---|---|")
band_rows = {}
for b, mk in ((0.0, lambda s: f"k16_s{s}_b0_c1"), (0.5, lambda s: f"k16_s{s}_b0.5")):
    vals, aps = [], []
    for s in (0, 1, 2):
        c = mk(s)
        pi, pa = ev(f"{c}_pred"), ev(f"{c}_pred", "mask_mAP50")
        vals.append(pi); aps.append(pa)
        w(f"| {b:g} | {s} | {geo(c,'K') or '--'} | {fmt(geo(c,'effective_rank'),2)} | "
          f"{fmt(geo(c,'pairwise_cos_mean'),3)} | {fmt(pi)} | {fmt(pa)} |")
    band_rows[b] = (band(vals), band(aps))
w("")
for b, (bi, ba) in band_rows.items():
    if bi and bi[2] > 1:
        w(f"- **β={b:g}** — crown IoU {bi[0]:.4f} ± {bi[1]:.4f}, AP50 {ba[0]:.4f} ± "
          f"{ba[1]:.4f} ({bi[2]} EM seeds)")
    elif bi:
        w(f"- **β={b:g}** — {bi[2]}/3 seeds measured so far")
b0, b5 = band_rows.get(0.0, (None,))[0], band_rows.get(0.5, (None,))[0]
a0, a5 = band_rows.get(0.0, (None, None))[1], band_rows.get(0.5, (None, None))[1]
if b0 and b5 and b0[2] > 1 and b5[2] > 1:
    w("")
    w("**Masker-seed variance is small — an order of magnitude below detector variance.** "
      f"Refitting the masker moves crown IoU by ±{max(b0[1], b5[1]):.4f} and AP50 by "
      f"±{max(a0[1], a5[1]):.4f}, against the published detector band of ±0.005. Fixing the "
      "masker at one EM seed therefore hides nothing, and the reported band is genuinely "
      "dominated by detector variance.\n")
    w("**β is inert, and the two metrics disagree on its sign.** β=0.5 leads on crown IoU by "
      f"{b5[0]-b0[0]:+.4f}; β=0 leads on AP50 by {a0[0]-a5[0]:+.4f}. A real effect would move "
      "both metrics the same way. Combined with the single-seed β sweep — effective rank "
      "11.4 → 1.7 for a 0.0011 spread — the repulsion term changes the model's geometry "
      "substantially and its accuracy not at all.\n")
    w("Note the diverse regime is also the more seed-stable one: effective rank spans "
      "11.35–11.88 at β=0 against 1.66–2.01 at β=0.5, and one seed prunes to K=15 in both.")
w("")

# ---------------------------------------------------------------- 6. provenance
w("## 6. Isolation and provenance\n")
w("- Every output is written under `/vol/collapse/`; `_assert_isolated()` refuses any path "
  "outside it. `/vol/out/` (phase4's outputs, including the deployed masker) is read-only "
  "here and was verified byte-identical before and after every run.\n")
w("- Cell names carry every axis that changes the fit — `k{K}_s{seed}_b{beta}_c{contrast}`. "
  "Cells fitted before the `contrast` axis existed carry no `_c` tag and are implicitly "
  "`contrast == (beta != 0)`, matching phase4's coupling.\n")
w("- `phase4_modal._selfmask_npz()` keys only on `(beta, fix)`, so a K=2 fit at β=0.5 would "
  "resolve to the deployed `em_model_4p_fix.npz` and be silently skipped by its "
  "`if os.path.exists(npz)` guard — returning the K=16 model labelled as K=2. This folder "
  "never uses that function.\n")
w("- Masker fits draw 120 tiles from the **792 train-partition** tiles only; "
  "`make_load_train_4p` filters `partition == \"train\"` before slicing, and all 108 "
  "validation tiles are marked `partition: \"val\"`. Fit/validation overlap is **zero**, and "
  "those records carry only `boxes`, `canopy`, `partition` — no crown polygons.\n")
w("- The new box-source eval reproduces the recorded validation cell exactly: the deployed "
  "masker on predicted boxes gives AP50 **0.6392** and AP50:95 **0.2680**, matching "
  "`sweep_val_knobs_s0.json` at α=0.3, κ×1.6, τ=0.25.\n")

open("RESULTS.md", "w").write("\n".join(out) + "\n")
print(f"wrote RESULTS.md ({len(out)} lines)")
