"""Fraction of arm-a tree detections reproduced exactly (box IoU>0.999, same 4dp score) in arm b."""
import json, numpy as np
A=json.load(open("arms_light.json"))
def tree(a,t):
    r=A[a][t]; sel=[i for i,c in enumerate(r["cls"]) if c==1]
    return np.array([r["boxes"][i] for i in sel],float).reshape(-1,4), np.array([r["scores"][i] for i in sel])
def iou(P,G):
    if len(P)==0 or len(G)==0: return np.zeros((len(P),len(G)))
    x0=np.maximum(P[:,None,0],G[None,:,0]);y0=np.maximum(P[:,None,1],G[None,:,1]);x1=np.minimum(P[:,None,2],G[None,:,2]);y1=np.minimum(P[:,None,3],G[None,:,3])
    I=np.clip(x1-x0,0,None)*np.clip(y1-y0,0,None);ap=(P[:,2]-P[:,0])*(P[:,3]-P[:,1]);ag=(G[:,2]-G[:,0])*(G[:,3]-G[:,1]);return I/(ap[:,None]+ag[None,:]-I+1e-9)
for a,b in (("512","1000"),("1000","2000")):
    for th in (0.0,0.5,0.7):
        n=exact=near=0
        for t in A[a]:
            Pa,sa=tree(a,t); Pb,sb=tree(b,t)
            m=sa>=th; Pa=Pa[m]; sa=sa[m]; n+=len(Pa)
            if len(Pa)==0 or len(Pb)==0: continue
            M=iou(Pa,Pb); j=M.argmax(1); best=M.max(1)
            exact+=int(((best>0.999)&(np.abs(sb[j]-sa)<1e-4)).sum()); near+=int((best>=0.9).sum())
        print(f"{a}->{b} tree dets score>={th}: {n}; exact {exact} ({exact/n:.3f}); IoU>=0.9 {near} ({near/n:.3f})")
