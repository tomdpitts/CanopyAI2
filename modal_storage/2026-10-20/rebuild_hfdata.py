"""Rebuild the HF restor/tcd cache that lived on Modal volume canopyai-deepforest-data:/hf_cache
(deleted 2026-10-20) and prove it identical to what LACE used.

Pins: revision d97d4da0ebbb6e249ae95ac5e19656babd972eb2 (the commit in every logged cache path
restor___tcd/default/0.0.0/<sha>; still hub HEAD on 2026-10-20), datasets==4.0.0 + numpy==2.2.6
(the Modal image pins in phase4_modal.py / tierb_modal.py).

Checks, every row of both splits (not the 20/split sample the manifest carries):
  1. decoded-RGB sha1 of the HF image == sha1 of the local data/tcd/<split>/<tile>.tif
     (the build_manifest.py recipe: np.asarray(PIL.convert("RGB")).tobytes())
  2. width/height and the coco_annotations string equal the local _meta.json
  3. every rgb_sha1 in phase4/manifest.json (40) and manifest_sparse.json (236) matches
Resumable: per-tile results append to verify.jsonl; done tiles are skipped.

    HF_HOME=<dir> <venv with datasets==4.0.0>/bin/python rebuild_hfdata.py <HF_HOME>
"""
import hashlib, json, os, sys, time

HF_HOME = sys.argv[1]
os.environ["HF_HOME"] = HF_HOME
REV = "d97d4da0ebbb6e249ae95ac5e19656babd972eb2"
REPO = "/Users/tompitts/dphil/CanopyAI2"
OUT = os.path.join(REPO, "modal_storage/2026-10-20")
LOG, JL = f"{OUT}/rebuild_hfdata.log", f"{OUT}/verify.jsonl"

import numpy as np
from PIL import Image
from datasets import load_dataset

Image.MAX_IMAGE_PIXELS = None
log = open(LOG, "a")
def say(*a):
    print(time.strftime("%H:%M:%S"), *a, file=log, flush=True)

sha = lambda img: hashlib.sha1(np.asarray(img.convert("RGB")).tobytes()).hexdigest()

say("load_dataset restor/tcd rev", REV, "HF_HOME", HF_HOME)
ds = {s: load_dataset("restor/tcd", split=s, revision=REV) for s in ("train", "test")}
for s, d in ds.items():
    say(s, "rows", len(d), "cache", d.cache_files[0]["filename"] if d.cache_files else None)
idx = {int(iid): (s, i) for s, d in ds.items() for i, iid in enumerate(d["image_id"])}

done = set()
if os.path.exists(JL):
    done = {json.loads(l)["tile"] for l in open(JL)}
jl = open(JL, "a")

tiles = []
for s in ("train", "test"):
    d = f"{REPO}/data/tcd/{s}"
    tiles += [(s, f[:-4]) for f in sorted(os.listdir(d)) if f.endswith(".tif")]
say("local tiles", len(tiles), "already done", len(done))

t0, bad = time.time(), 0
for k, (s, tid) in enumerate(tiles, 1):
    key = f"{s}/{tid}"
    if key in done:
        continue
    meta = json.load(open(f"{REPO}/data/tcd/{s}/{tid}_meta.json"))
    iid = int(meta["image_id"])
    rec = {"tile": key, "image_id": iid}
    if iid not in idx:
        rec["ok"] = False; rec["err"] = "image_id not in HF"
    else:
        hs, hi = idx[iid]
        row = ds[hs][hi]
        rec["hf_split"] = hs
        rec["sha_hf"] = sha(row["image"])
        rec["wh_ok"] = (row["width"], row["height"]) == (meta["width"], meta["height"])
        rec["ann_ok"] = row["coco_annotations"] == meta["coco_annotations"]
        try:
            rec["sha_local"] = sha(Image.open(f"{REPO}/data/tcd/{s}/{tid}.tif"))
            rec["ok"] = rec["sha_local"] == rec["sha_hf"] and rec["wh_ok"] and rec["ann_ok"] and hs == s
        except Exception as e:   # local copy damaged (4 zero-byte tifs from the 08-13 migration)
            rec["local_unreadable"] = repr(e)[:120]
            rec["ok"] = None     # no pixel reference; meta/annotation checks still recorded
    bad += rec["ok"] is False
    jl.write(json.dumps(rec) + "\n"); jl.flush()
    if k % 200 == 0:
        say(f"{k}/{len(tiles)} bad_so_far={bad} {(time.time()-t0)/60:.1f} min")

# manifest cross-check (independent of the local tifs)
res = {}
for f in ("boxinst_commonality_tcd_04/modal_tcd_multiseed/phase4/manifest.json",
          "boxinst_commonality_tcd_04/modal_sparse_tcd_multiseed/manifest_sparse.json"):
    m = json.load(open(f"{REPO}/{f}"))
    ok = n = 0; fails = []
    for part in m.values():
        for tid, r in part.items():
            if "rgb_sha1" not in r:
                continue
            n += 1
            hs, hi = idx[int(r["image_id"])]
            if sha(ds[hs][hi]["image"]) == r["rgb_sha1"]:
                ok += 1
            else:
                fails.append(tid)
    res[f] = {"n": n, "ok": ok, "fails": fails}
    say("manifest", f, f"{ok}/{n}")

rows = [json.loads(l) for l in open(JL)]
summary = {"revision": REV, "datasets": "4.0.0", "hf_rows": {s: len(d) for s, d in ds.items()},
           "local_tiles": len(rows), "local_ok": sum(r["ok"] is True for r in rows),
           "local_bad": [r for r in rows if r["ok"] is False][:50],
           "local_unreadable": [r for r in rows if r["ok"] is None], "manifests": res}
json.dump(summary, open(f"{OUT}/verify_summary.json", "w"), indent=1)
say("DONE", json.dumps({k: v for k, v in summary.items() if k != "local_bad"}))
