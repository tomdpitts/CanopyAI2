"""CANONICAL headline tables (2026-10-27; D1 row added 2026-11-03; edge fix + R3 rows added 2026-11-04): Table 1 (OAM-TCD 439 test) and tab:sparse236, current rows only.

Reads scored JSONs only (canopy-neutral crowd arm, 2048^2, maxDets 600); writes results_439/TABLE1_CURRENT.{json,md}.
Every cell is asserted against the value quoted in chat/paper on 2026-10-27, so a silently changed source file fails
loudly instead of moving a headline number. Supersedes table_439.md (512 raster) and r2048/table_r2048.md
(pre-seam-fix baselines, 8 px LACE masker).

  .venv/bin/python -m boxinst_commonality_tcd_04.results_439.build_table1_current
"""
import json, os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NR = os.path.join(H, "native_raster")
SF, HT, TA = (os.path.join(NR, d) for d in ("seamfix", "hunt/test", "tierA"))
DD = os.path.join(H, "dino_det", "results")
BT = os.path.join(H, "box_supervised_baselines", "tierb")
OUT = os.path.join(H, "results_439", "TABLE1_CURRENT")

# (label, supervision, images, [scored files], expected (AP50, AP50:95, box AP50) at 4 dp, note)
T439 = [
    ("Restor Mask R-CNN", "masks", 4169, [f"{NR}/scored_restor_rpn1000_r2048_tree.json"], (0.6480, 0.3332, 0.6646),
     "released restor/tcd-mask-rcnn-r50, tree class, RPN topk 1000, score floor 0.05"),
    ("Detectree2", "masks", 900, [f"{SF}/scored_dt2_s0v2_r2048_seamfix.json"], (0.6297, 0.3099, 0.6505),
     "fine-tuned by us from 250312_flexi.pth; seam-fixed stitch (2026-10-14)"),
    ("SelvaBox -> SAM 3 (FT)", "box+SAM", 3024, [f"{SF}/preds_selvabox_seamfix_on_r2048_rankgmean_scored_md600.json"], (0.6141, 0.2502, 0.7618),
     "SelvaBox DINO-Swin-L -> SAM 3 selvamask-FT checkpoint (NOT zero-shot); seam-fixed; sqrt(s * SAM score) ranking (seamfix/selvabox_gmean_rescore.py); env gate seamfix/selvabox_env_gate_gmean.json"),
    ("Box2Mask Swin-L", "box", 900, [f"{SF}/scored_box2mask_swinl_s0_r2048_seamfix.json"], (0.5894, 0.2446, 0.6199), "seam-fixed"),
    ("Box2Mask R-50", "box", 900, [f"{SF}/scored_box2mask_r50_s0_r2048_seamfix.json"], (0.4505, 0.1646, 0.5122), "seam-fixed"),
    ("BoxTeacher Swin-B", "box", 900, [f"{BT}/results/20261101_192510/boxteacher/results/scored_boxteacher_sub_test_r2048.json"],
     (0.6642, 0.3049, 0.7033), "Tier B B2 (box_supervised_baselines/tierb/TIERB.md): trained by us on the 792 OAM-TCD train tiles, LACE canopy rule; 3x3 subtiles, seam-fixed stitch, product ranking"),
    ("LACE CenterNet detector -> SAM 3 zs", "box+SAM", 900, [f"{TA}/scored_a1_lace_s{s}_sam3zs_gmean.json" for s in (0, 1, 2)],
     (0.6479, 0.3008, 0.6515), "same LACE boxes; SAM 3 zero-shot masks; SAM's own ranking sqrt(s * SAM score)"),
    ("LACE (CenterNet variant)", "box", 900, [f"{HT}/scored_lace_s{s}_r2048_lace4px_simple.json" for s in (0, 1, 2)],
     (0.6545, 0.2803, 0.6664), "lace4px_simple masker (hunt/FROZEN_simple_2026-10-26.json), own 4 px posterior ranking"),
    ("LACE (D1 variant)", "box", 900, [f"{DD}/scored_dino_swinl_s0_test_sprime_r2048.json"],
     (0.7035, 0.3347, 0.7391), "dino_det/README.md pre-registration D1 (2026-11-03): detrex DINO-Swin-L, COCO-trained init, trained by us "
     "on the 792 OAM-TCD train tiles (boxes only, LACE canopy rule, Tier B crops + plateau + 896-subtile selection), best epoch 6; "
     "unchanged lace4px_simple masker, s' ranking (val-selected); single seed; paired tile bootstrap vs BoxTeacher +0.039 [0.032, 0.049]"),
    ("LACE (D1 variant) + edge fix", "box", 900, [f"{DD}/edgefix/scored_d1_edgefix_test_sprime_r2048.json"], (0.7044, 0.3366, 0.7391),
     "top/left border fix (GA.EDGE_FILL = True, adopted by Tom 2026-11-04; dino_det/SAM3_D1.md): masks only, s' ranking unchanged (asserted); run once, dino_det/edgefix_test.py"),
    ("LACE (D1 variant) + edge fix + R3", "box", 900, [f"{DD}/scored_dino_swinl_s0_test_R3_ef_r2048.json"], (0.7141, 0.3424, 0.7471),
     "POST-HOC (dino_det/README.md R3 + R3-edgefix, 2026-11-04): edge-fixed masks + R3 reranker (MLP 1x64 on s, log s'/s, log box h, max IoU with a higher-scored det; 25-model mean, val-trained); vs s'+fix AP50 +0.0097 [0.0069, 0.0124], AP50:95 +0.0059 [0.0046, 0.0073]; run once, dino_det/r3_edgefix.py"),
    ("LACE (D1 variant) + edge fix + R3 + σ2", "box", 900, [f"{DD}/scored_dino_swinl_s0_test_R3_ef_blur2_r2048.json"], (0.7142, 0.3444, 0.7471),
     "POST-HOC (dino_det/README.md R3-edgefix): R3 + edge fix row with masks Gaussian-smoothed sigma 2 px, threshold 0.5; vs R3 + fix AP50:95 +0.0020 [0.0013, 0.0032], AP50 +0.0001"),
]
TSP = [
    ("Detectree2", "masks", 900, [f"{SF}/scored_dt2_sparse_s0v2_r2048_seamfix.json"], (0.6362, 0.3532, 0.6444), "seam-fixed"),
    ("Box2Mask Swin-L", "box", 900, [f"{SF}/scored_box2mask_swinl_sparse_s0_r2048_seamfix.json"], (0.6044, 0.2618, 0.6265),
     "seam-fixed; same s0 checkpoint as the 439 row (iter 23422); Modal A100 predict 2026-10-27, fc-01M5XCK31BK2D5TW47B91DG2D6"),
    ("BoxTeacher Swin-B", "box", 900, [f"{BT}/results_sparse/scored_boxteacher_sub_sparse_r2048.json"],
     (0.7206, 0.3756, 0.7391), "frozen B2 checkpoint, inference only (2026-11-03, TIERB.md 'B2 sparse-236'), same protocol as its 439 row"),
    ("LACE (CenterNet variant)", "box", 900, [f"{HT}/scored_lace_sparse_s{s}_r2048_lace4px_simple.json" for s in (0, 1, 2)],
     (0.6785, 0.3192, 0.6744), "lace4px_simple"),
    ("LACE (D1 variant)", "box", 900, [f"{DD}/scored_dino_swinl_s0_sparse_sprime_r2048.json"],
     (0.7349, 0.3962, 0.7496), "dino_det/README.md pre-registration D1-sparse (2026-11-03): frozen D1 (epoch 6) + lace4px_simple, s' (val-selected for D1), single run"),
    ("LACE (D1 variant) + edge fix", "box", 900, [f"{DD}/edgefix/scored_d1_edgefix_sparse_sprime_r2048.json"], (0.7350, 0.3980, 0.7496),
     "top/left border fix (GA.EDGE_FILL = True, adopted by Tom 2026-11-04; dino_det/SAM3_D1.md): masks only, s' ranking unchanged (asserted); run once, dino_det/edgefix_test.py"),
    ("LACE (D1 variant) + edge fix + R3", "box", 900, [f"{DD}/scored_dino_swinl_s0_sparse_R3_ef_r2048.json"], (0.7434, 0.4031, 0.7570),
     "POST-HOC (dino_det/README.md R3 + R3-edgefix, 2026-11-04): edge-fixed masks + R3 reranker (MLP 1x64 on s, log s'/s, log box h, max IoU with a higher-scored det; 25-model mean, val-trained); vs s'+fix AP50 +0.0084 [0.0037, 0.0152], AP50:95 +0.0051 [0.0022, 0.0088]; run once, dino_det/r3_edgefix.py"),
    ("LACE (D1 variant) + edge fix + R3 + σ2", "box", 900, [f"{DD}/scored_dino_swinl_s0_sparse_R3_ef_blur2_r2048.json"], (0.7427, 0.4032, 0.7570),
     "POST-HOC (dino_det/README.md R3-edgefix): R3 + edge fix row with masks Gaussian-smoothed sigma 2 px, threshold 0.5; vs R3 + fix AP50:95 +0.0001 [-0.0016, 0.0018] (n.s.), AP50 -0.0007"),
]


def cells(files):
    v = []
    for fp in files:
        d = json.load(open(fp))
        p = d["protocol"]
        assert (p["res"], p["max_dets"]) == (2048, 600), (fp, p)
        c = d["canopy_neutral_crowd"]
        v.append((c["mask"]["AP50"], c["mask"]["AP50_95"], c["box"]["AP50"]))
    v = np.asarray(v)
    return v.mean(0), (v.std(0, ddof=1) if len(v) > 1 else None), v


def _hu(x, dp):
    """Round half UP (5 rounds up), on the float cleaned to 10 dp -- never Python's float/banker's rounding."""
    return Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)


def block(rows, blk, dp_check=4, dp=3):
    out, md = [], []
    for lab, sup, n, files, exp, note in rows:
        m, sd, per = cells(files)
        got = tuple(round(float(x), dp_check) for x in m)
        assert got == exp, f"{lab}: {got} != expected {exp}"
        fmt = lambda i: f"{_hu(m[i], dp)}" + (f" ± {_hu(sd[i], dp)}" if sd is not None else "")
        out.append({"method": lab, "supervision": sup, "images": n, "n_seeds": len(files),
                    "AP50": float(m[0]), "AP50_95": float(m[1]), "box_AP50": float(m[2]),
                    "sd_ddof1": None if sd is None else [float(x) for x in sd], "per_seed": per.tolist(),
                    "sources": [os.path.relpath(f, H) for f in files], "note": note})
        seeds = f", {len(files)} seeds" if len(files) > 1 else ""
        md.append(f"| {lab}{seeds} | {sup} | {n} | {fmt(0)} | {fmt(1)} | {fmt(2)} |")
    return out, md


def main():
    r439, m439 = block(T439, "439")
    rsp, msp = block(TSP, "sparse236")
    hdr = ["| Method | Supervision | Images | AP50 | AP50:95 | Box AP50 |", "|---|---|---|---|---|---|"]
    md = ["# CANONICAL headline tables — current as of 2026-11-04 (LACE (D1 variant) + edge fix / + R3 / + σ2 rows added 2026-11-04; D1 + BoxTeacher rows 2026-11-03; all other rows unchanged since 2026-10-27)",
          "", "Generated by `build_table1_current.py` from scored JSONs (asserted). This file is the single source of truth "
          "for 439/sparse236 headline numbers; every other table in the repo is either a per-experiment record or marked SUPERSEDED.",
          "", "## Table 1 — OAM-TCD 439 test (canopy-neutral, 2048², maxDets 600)", "", *hdr, *m439, "",
          "- LACE CenterNet detector → SAM 3 zs: same LACE (CenterNet variant) boxes, SAM 3 **zero-shot** masks, SAM's own ranking √(s·SAM). Box AP50 differs from LACE (CenterNet variant)'s because the ranking differs.",
          "- SelvaBox → SAM 3 uses the SelvaMask **fine-tuned** SAM 3 (`CanopyRS/sam3-multi-selvabox-selvamask-FT`), not zero-shot. Ranked by √(s·SAM) (CanopyRS aggregator key; same rule as the LACE→SAM 3 zs row) since 2026-11-03; det-score ranking gave 0.598 / 0.221 / 0.767 (`seamfix/scored_selvabox_seamfix_on_r2048_md600.json`). Mixed A100/M4 SAM scores gated: `native_raster/seamfix/selvabox_env_gate_gmean.json`, 44 tiles / 48,414 boxes, ΔAP50 −0.0003, ΔAP50:95 0.0000.",
          "- **Naming (Tom, 2026-11-03):** D1 *is* LACE — externally (paper, talks) it is just \"LACE\". In repo docs: **LACE (D1 variant)** = DINO-Swin-L detector + LACE masker; **LACE (CenterNet variant)** = the earlier frozen-DINOv3 CenterNet-style detector + LACE masker (the 0.655 row).",
          "- LACE masker = `lace4px_simple` 4 px masker (Tom's preferred masker 2026-10-26). paper.tex tab:oamtcd439 and tab:sparse236 carry these rows (2026-10-27); abstract/L662 prose, Methods L502/L504, tab:config, tab:compute and both qualitative figures still describe the 8 px masker — see `native_raster/hunt/ADOPT_AUDIT_2026-10-27.md`.",
          "- **LACE (D1 variant) (2026-11-03)** = LACE with its detector stage a DINO-Swin-L (detrex, COCO-trained init, never SelvaBox weights) trained by us on OAM-TCD boxes only, + the unchanged LACE masker and s′ ranking; single seed. In paper.tex Table 1 as \"LACE (ours)\" (2026-11-03). BoxTeacher Swin-B (Tier B B2) rows added 2026-11-03 for both tables (sparse236: frozen checkpoint, inference only). Comparison row not in this table: frozen-DINOv3 detector C2 + LACE masker 0.663 / 0.297 / box 0.693 (`dethead/README.md`). D1 − BoxTeacher paired tile bootstrap +0.039 [0.032, 0.049] (`dino_det/results/boot_test_mask_ap50.json`). D1 sparse236 row added 2026-11-03 (pre-registration D1-sparse); sparse236 D1 − BoxTeacher paired tile bootstrap +0.014 [−0.005, 0.033], CI includes 0 (`dino_det/results/boot_sparse_mask_ap50.json`).",
          "- **Edge fix + R3 rows (2026-11-04, Tom: update TABLE1_CURRENT with the R3 edge-fix results).** *+ edge fix* = top/left 6 px border bug in `guided_ablate_modal._prob` fixed (`EDGE_FILL = True`, adopted 2026-11-04); same boxes and s′ ranking. *+ R3* = the post-hoc R3 reranker (4 inputs: s, log s′/s, log box h, max IoU with a higher-scored det; MLP 1×64, 25-model mean, trained on val only); confirmed on sparse236 (CI vs s′ + fix excludes 0 on both splits, AP50 and AP50:95). *+ σ2* = masks Gaussian-smoothed σ 2 px: AP50:95 +0.002 on test (CI > 0), NOT confirmed on sparse236 (+0.000, n.s.). All three are single looks, scored once. R3 and σ2 are post-hoc (designed after test and sparse had been seen for R2); disclose as such. The plain LACE (D1 variant) row is kept as the published/pre-registered reference. paper.tex NOT updated (Tom decides). Full record: `dino_det/README.md` (R3, S2-final, R3-edgefix), `dino_det/results/r3_edgefix.json`.",
          "- ± = sd over 3 detector seeds, ddof 1. Baselines are single runs. 3 dp, round half up (5 rounds up); full precision in TABLE1_CURRENT.json.",
          "", "## tab:sparse236 — sparse-canopy holdout (236 tiles, same protocol)", "", *hdr, *msp, "",
          "## Sources", "", *[f"- {r['method']}: " + ", ".join(f"`{s}`" for s in r["sources"]) + f" — {r['note']}" for r in r439 + rsp], ""]
    json.dump({"generated_by": "results_439/build_table1_current.py", "date": "2026-11-04",
               "protocol": "canopy-neutral crowd, 2048^2, maxDets 600, pycocotools COCOeval",
               "table1_439": r439, "sparse236": rsp}, open(OUT + ".json", "w"), indent=1)
    open(OUT + ".md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
