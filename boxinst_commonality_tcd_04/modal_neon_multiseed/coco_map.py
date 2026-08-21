"""COCO box AP (50:95) for the NEON 194-tile benchmark, offline from saved preds."""
import json, sys, contextlib, io, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT = json.load(open('neon_gt.json'))

def build(pref=None):
    keys = sorted(k for k in GT if pref is None or pref in k)
    idx = {k: i + 1 for i, k in enumerate(keys)}
    imgs = [{'id': idx[k], 'file_name': k, 'width': 400, 'height': 400} for k in keys]
    anns, a = [], 1
    for k in keys:
        for x1, y1, x2, y2 in GT[k]:
            anns.append({'id': a, 'image_id': idx[k], 'category_id': 1, 'iscrowd': 0,
                         'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1)})
            a += 1
    d = {'images': imgs, 'annotations': anns, 'categories': [{'id': 1, 'name': 'tree'}]}
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO(); c.dataset = d; c.createIndex()
    return c, idx

def ap(preds_path, coco, idx, maxdets=1000):
    P = json.load(open(preds_path))
    dets = []
    for k, i in idx.items():
        v = P.get(k, {'boxes': [], 'scores': []})
        for (x1, y1, x2, y2), s in zip(v['boxes'], v['scores']):
            dets.append({'image_id': i, 'category_id': 1, 'score': float(s),
                         'bbox': [x1, y1, x2 - x1, y2 - y1]})
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(dets)
        e = COCOeval(coco, dt, 'bbox')
        e.params.maxDets = [1, 10, maxdets]
        e.evaluate(); e.accumulate(); e.summarize()
    # compute AP directly (summarize's stats[0] assumes maxDets==100)
    pr = e.eval['precision']  # [T,R,K,A,M]; A=0 all areas, M=2 -> maxdets
    def _ap(t=None):
        x = pr[:, :, :, 0, 2] if t is None else pr[t:t+1, :, :, 0, 2]
        x = x[x > -1]
        return float(x.mean()) if x.size else float('nan')
    rc = e.eval['recall'][:, :, 0, 2]
    return {'AP': _ap(), 'AP50': _ap(0), 'AP75': _ap(5),
            'APs': e.stats[3], 'APm': e.stats[4], 'APl': e.stats[5],
            'AR': float(rc[rc > -1].mean())}

for scope, pref in [('GLOBAL(194)', None), ('NIWO(12)', 'NIWO')]:
    coco, idx = build(pref)
    print(f"\n=== {scope}  n={len(idx)} ===")
    print(f"{'arm':10} {'AP50:95':>8} {'AP50':>7} {'AP75':>7} {'APs':>7} {'APm':>7} {'APl':>7} {'AR1000':>7}")
    for arm, tpl in [('4phase', 'phase4/dl/preds_phase4_s{}.json'),
                     ('native', 'preds/preds_neon_s{}.json'),
                     ('4px',    'preds/preds_neon_s{}_4px.json')]:
        rows = [ap(tpl.format(s), coco, idx) for s in range(5)]
        m = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        sd = {k: np.std([r[k] for r in rows], ddof=1) for k in rows[0]}
        print(f"{arm:10} {m['AP']:.3f}±{sd['AP']:.3f} {m['AP50']:>7.3f} {m['AP75']:>7.3f} "
              f"{m['APs']:>7.3f} {m['APm']:>7.3f} {m['APl']:>7.3f} {m['AR']:>7.3f}")
        if arm == '4phase':
            print(f"{'':10} per-seed AP: " + " / ".join(f"{r['AP']:.3f}" for r in rows))
    # DeepForest
    try:
        r = ap('preds_deepforest.json', coco, idx)
        print(f"{'DeepForest':10} {r['AP']:.3f}       {r['AP50']:>7.3f} {r['AP75']:>7.3f} "
              f"{r['APs']:>7.3f} {r['APm']:>7.3f} {r['APl']:>7.3f} {r['AR']:>7.3f}")
    except Exception as ex:
        print('DeepForest skipped:', ex)
