"""Modal CPU app: refit the LACE masker on the 4 px lattice (pre-registration: masker_refit4p/README.md).

Reads tcd04-phase4-vol:z4p_train (4 px cells, deployed whitening), writes tcd04-phase4-vol:out/refit4p/n<N>/.
Checkpoints after init and every EM iteration (vol.commit); a re-spawn resumes. Detached launch:
    .venv/bin/python boxinst_commonality_tcd_04/native_raster/detach.py deploy ../masker_refit4p/refit4p_modal.py
    .venv/bin/python boxinst_commonality_tcd_04/native_raster/detach.py spawn tcd04-masker-refit4p fit '{"n_tiles": 120}'
Logs: modal app logs tcd04-masker-refit4p
"""
import hashlib
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
STUB = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "stubs", "boxinst_commonality")
DEPLOYED = "/Users/tompitts/dphil/feat_cache/em_model_4p_fix.npz"
DEPLOYED_SHA1 = "2f0eaa3f0a17501a482b96b8feacf53f32a18d4e"
P = "/root/proj"

app = modal.App("tcd04-masker-refit4p")
vol = modal.Volume.from_name("tcd04-phase4-vol")
image = (modal.Image.debian_slim(python_version="3.11").pip_install("numpy==2.2.6", "pillow")
         .add_local_file(os.path.join(STUB, "__init__.py"), f"{P}/boxinst_commonality/__init__.py")
         .add_local_file(os.path.join(STUB, "em.py"), f"{P}/boxinst_commonality/em.py")
         .add_local_file(os.path.join(HERE, "fit4p.py"), f"{P}/fit4p.py")
         .add_local_file(os.path.join(PKG, "train_tiles_gt.json"), f"{P}/train_tiles_gt.json")
         .add_local_file(DEPLOYED, f"{P}/em_model_4p_fix.npz"))


@app.function(image=image, volumes={"/vol": vol}, cpu=8, memory=98304, timeout=6 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def fit(n_tiles: int):
    import sys
    sys.path.insert(0, P)
    import fit4p
    dep = f"{P}/em_model_4p_fix.npz"
    assert hashlib.sha1(open(dep, "rb").read()).hexdigest() == DEPLOYED_SHA1, "deployed masker sha1 mismatch"
    n = len([f for f in os.listdir("/vol/z4p_train") if f.endswith(".npy")])
    assert n == 900, n
    out = f"/vol/out/refit4p/n{n_tiles}"
    npz = fit4p.fit_nopca("/vol/z4p_train", f"{P}/train_tiles_gt.json", n_tiles, dep, out, commit=vol.commit)
    return {"npz": npz, "sha1": hashlib.sha1(open(npz, "rb").read()).hexdigest()}
