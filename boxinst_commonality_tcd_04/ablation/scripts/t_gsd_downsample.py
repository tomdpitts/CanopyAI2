"""Crown loss under GSD downsampling, from OAM-TCD 439 test GT polygons only (no model).
Definitions: a coarse pixel is 'enclosed' if all four corners lie inside the polygon;
'centre-in' if its centre does; 'bbox side < 1 px' if the polygon's bounding box is narrower
than one coarse pixel in x or y. Native GSD 0.1 m/px, tiles 2048^2. Grid origin at tile (0,0).
Run: .venv/bin/python boxinst_commonality_tcd_04/ablation/scripts/t_gsd_downsample.py"""
import json, numpy as np
from matplotlib.path import Path
d=json.load(open('boxinst_commonality_tcd_04/test_gt.json'))
polys=[np.array(p,float).reshape(-1,2) for t in d.values() for p in t['trees']]
n=len(polys); bw=np.array([np.ptp(P[:,0]) for P in polys]); bh=np.array([np.ptp(P[:,1]) for P in polys])
out={'n_crowns':n,'native_m_per_px':0.1,'rows':{}}
for gsd in (0.5,1.0,2.0,3.0):
    s=gsd/0.1; zc=zf=0
    for P in polys:
        x0,y0=np.floor(P.min(0)/s); x1,y1=np.ceil(P.max(0)/s)
        gx,gy=np.meshgrid(np.arange(x0,x1+1),np.arange(y0,y1+1)); gx=gx.ravel(); gy=gy.ravel(); pth=Path(P)
        if not pth.contains_points(np.c_[(gx+0.5)*s,(gy+0.5)*s]).any(): zc+=1
        full=np.ones(gx.size,bool)
        for dx,dy in ((0,0),(1,0),(0,1),(1,1)): full&=pth.contains_points(np.c_[(gx+dx)*s,(gy+dy)*s])
        if not full.any(): zf+=1
    out['rows'][f'{gsd}m']={'frac_zero_enclosed_px':round(zf/n,4),'frac_zero_centre_in_px':round(zc/n,4),
                            'frac_bbox_side_lt_1px':round(float(((bw<s)|(bh<s)).mean()),4)}
    print(gsd, out['rows'][f'{gsd}m'])
json.dump(out,open('boxinst_commonality_tcd_04/ablation/results/gsd_downsample.json','w'),indent=1)
