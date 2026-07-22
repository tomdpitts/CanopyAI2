import json,os,sys; sys.path.insert(0,'.')
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import df_scorer
HERE=os.path.abspath('.'); GT=os.path.join(HERE,'neon_gt.json')
CACHE=os.path.join(HERE,'phase4','dl','pr_curves.json')
def curve(preds):
    c=df_scorer.score(preds,GT,iou=0.4)['pr_curve']
    return [(x['mean_recall'],x['mean_precision']) for x in c]
# compute + cache 4-phase and native seed curves
if os.path.exists(CACHE):
    cc=json.load(open(CACHE))
else:
    cc={'phase4':[],'native':[]}
    for s in range(5):
        cc['phase4'].append(curve(f'phase4/dl/preds_phase4_s{s}.json')); print('ph',s,flush=True)
        cc['native'].append(curve(f'preds/preds_neon_s{s}.json')); print('na',s,flush=True)
    json.dump(cc,open(CACHE,'w'))
# DeepForest 2.1.0 repro curve (cached)
df=json.load(open('deepforest_repro.json'))
dfc=sorted((p['mean_recall'],p['mean_precision']) for p in df['pr_curve'])
def band(seed_curves,grid):
    # interp precision onto common recall grid per seed
    M=[]
    for cv in seed_curves:
        pts=sorted(cv)                      # by recall
        R=[r for r,_ in pts]; P=[p for _,p in pts]
        M.append(np.interp(grid,R,P,left=P[0],right=np.nan))
    M=np.array(M)
    return np.nanmean(M,0),np.nanmin(M,0),np.nanmax(M,0)
grid=np.linspace(0,0.86,120)
pm,plo,phi=band(cc['phase4'],grid)
nm,nlo,nhi=band(cc['native'],grid)
fig,ax=plt.subplots(figsize=(6.6,6.2))
# native (context)
ax.plot(grid,nm,'--',color='#8d99ae',lw=1.6,label='Ours native 5-seed (interp. 8px), F1 0.689')
# 4-phase band
ax.fill_between(grid,plo,phi,color='#2a6f97',alpha=0.18,label='4-phase seed min–max')
ax.plot(grid,pm,'-',color='#2a6f97',lw=2.6,label='Ours 4-phase 5-seed (real 8px), F1 0.728')
# DeepForest 2.1.0 repro curve
dr=[r for r,_ in dfc]; dp=[p for _,p in dfc]
ax.plot(dr,dp,'-',color='#e07a5f',lw=2.0,label='DeepForest 2.1.0 (our repro), F1 0.726')
ax.plot(0.709,0.745,'s',color='#e07a5f',ms=9,label='DeepForest 2.1.0 best-F1 (P.745/R.709)')
# published paper point
ax.plot(0.790,0.659,'*',color='#d62828',ms=20,label='DeepForest published (P.659/R.790), F1 0.719')
ax.axvline(0.790,color='#d62828',ls=':',lw=0.8,alpha=0.45)
ax.set_xlabel('Recall (macro over 194 images)'); ax.set_ylabel('Precision (macro over 194 images)')
ax.set_xlim(0,1); ax.set_ylim(0.3,1); ax.grid(alpha=0.25)
ax.set_title('NEON crowns (RGB-only, IoU 0.4): 4-phase real-8px vs DeepForest 2.1.0')
ax.legend(loc='lower left',fontsize=7.6)
out=os.path.join(HERE,'phase4','pr_curve_phase4.png')
fig.tight_layout(); fig.savefig(out,dpi=145); print('wrote',out,flush=True)
