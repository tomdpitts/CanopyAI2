"""Recall@IoU0.5 for the Restor RPN-budget rows of Table rpn, recovered from the stored
prediction files with the frozen scorer (score_coco: 512^2 raster, maxDets 600, floor 0.05,
canopy as iscrowd). COCOeval's eval['recall'][T,K,A,M] at T=0 (IoU 0.5) is the fraction of
the 25,705 GT crowns matched by any kept detection, i.e. recall at the end of the ranked list.
The 512 and 2000 rows are recorded in PROTOCOL_439.md (0.6442, 0.7804) and serve as the gate."""
import json, sys, numpy as np
sys.path.insert(0, '.')
from boxinst_commonality_tcd_04 import score_coco as S
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
gt = json.load(open('boxinst_commonality_tcd_04/test_gt.json'))
gt_cr, tid2img, tree_rles, _ = S.build_gt(gt, 512, with_canopy=True)
cg = COCO(); cg.dataset = gt_cr; cg.createIndex()
rows = {'512 (released)': 'restor_baseline/preds_restor.json',
        '1000 (Detectron2 default)': 'restor_baseline/preds_restor_rpn1000_treeonly.json',
        '2000': 'restor_baseline/preds_restor_rpn2000.json'}
out = {}
for name, path in rows.items():
    P = json.load(open('boxinst_commonality_tcd_04/' + path)); preds = P.get('preds', P)
    dets, _ = S.build_dt(preds, tid2img, 512, S.SCORE_FLOOR)
    cd = cg.loadRes([dict(d) for d in dets]); ev = COCOeval(cg, cd, iouType='segm')
    ev.params.iouThrs = S.IOU_THRS; ev.params.recThrs = S.REC_THRS; ev.params.maxDets = [S.MAX_DETS]
    ev.params.areaRng = [[0.0, 1e10]]; ev.params.areaRngLbl = ['all']
    ev.evaluate(); ev.accumulate()
    r = ev.eval['recall']; p = ev.eval['precision']
    i50 = int(np.argmin(np.abs(S.IOU_THRS - 0.5)))
    ap50 = float(p[i50, :, 0, 0, 0][p[i50, :, 0, 0, 0] > -1].mean())
    out[name] = {'preds': path, 'n_det': len(dets), 'dets_per_tile': round(len(dets) / 439, 1),
                 'recall_at_iou50': round(float(r[i50, 0, 0, 0]), 4), 'AP50_check': round(ap50, 4)}
    print(name, out[name], flush=True)
json.dump(out, open('boxinst_commonality_tcd_04/ablation/results/restor_rpn_recall.json', 'w'), indent=1)
