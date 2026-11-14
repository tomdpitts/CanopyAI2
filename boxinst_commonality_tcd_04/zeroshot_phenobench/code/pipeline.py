"""Full val run (PB_RUN=val): per image, 16-phase features computed IN MEMORY (not stored: 772 x 134 MB = 103 GB) and
all CPU arms of run.py applied immediately. Per-image checkpoint = the per-arm result files (run.process). Images whose
data is not yet downloaded are skipped and retried on the next pass, so this can run alongside fetch.py.
Venv: feat_cache/venv_feat_t457 (transformers 4.57.1 = the deployed feature stack); the CPU arms are numpy/pycocotools
and are parity-gated in this venv too (results/gate_parity_t457.json).

  cd zeroshot_phenobench/code && PB_RUN=val PYTHONPATH=/Users/tompitts/dphil/CanopyAI2 \
    /Users/tompitts/dphil/feat_cache/venv_feat_t457/bin/python -u pipeline.py
"""
import json, os, time

import numpy as np

from common import RES, RUN, stems, have_data, load_rgbf, features16, check_shift
import run as RUNMOD


def main():
    import torch, transformers
    from dapt.backbone import FrozenDinoV3Features
    check_shift(); assert transformers.__version__ == "4.57.1", transformers.__version__
    tree = RUNMOD.setup(); S = stems()
    net = FrozenDinoV3Features("web", layers=(24,), device="mps")
    print(f"[pb pipeline {RUN}] {len(S)} images, torch {torch.__version__}, transformers {transformers.__version__}", flush=True)
    T0 = time.time(); n_done = 0; tl = os.path.join(RES, "_timing.jsonl")
    for p in range(1000):
        left = [t for t in S if RUNMOD.todo_arms(t)]
        if not left:
            break
        ready = [t for t in left if have_data(t)]
        if not ready:
            print(f"  pass {p}: {len(left)} left, none downloaded yet; waiting 60 s", flush=True); time.sleep(60); continue
        for t in ready:
            t0 = time.time(); F4 = features16(net, load_rgbf(t)).astype(np.float32); tf = time.time() - t0
            msg = RUNMOD.process(t, F4, tree, f"[{S.index(t)+1}/{len(S)}]"); del F4
            n_done += 1; dt = time.time() - t0
            open(tl, "a").write(json.dumps({"image": t, "feat_s": round(tf, 2), "total_s": round(dt, 2)}) + "\n")
            print(f"{msg}  feat {tf:.1f}s tot {dt:.1f}s  done {n_done}, left {len(left) - n_done if p == 0 else '?'}  [{(time.time()-T0)/60:.1f} min]", flush=True)
    print(f"[pb pipeline {RUN}] done ({n_done} images this session)", flush=True)


if __name__ == "__main__":
    main()
