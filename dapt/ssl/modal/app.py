"""Clean, isolated Modal project for DINOv3 DAPT (arid continued-SSL of ViT-L).

Fresh app + dedicated volume, nothing shared with other Modal work. Single A100 with a
hard timeout well under the $30 cap. Steps: build a CUDA image with a patched dinov3
clone, stage pool + converted web checkpoint + config on the volume, run a CPU-cheap
pre-flight (assert weights load with 0 missing params + print effective LR), then a
single torchrun training run that DCP-checkpoints every 500 iters (the step-sweep).

Deploy/run: see the Run order section of dapt/ssl/SPEC.md. Nothing here launches
automatically.
"""
import os
import subprocess

import modal

APP_NAME = "dinov3-dapt-arid"          # dedicated, clean app
VOL_NAME = "dinov3-dapt-arid-vol"      # dedicated, clean volume
HERE = os.path.dirname(os.path.abspath(__file__))
SSL_DIR = os.path.dirname(HERE)        # dapt/ssl

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

# Image: debian_slim + EXPLICITLY PINNED torch/torchvision (PyPI linux wheels bundle
# CUDA 12.6). Do NOT use from_registry(pytorch/pytorch, add_python=...): add_python
# installs a standalone interpreter that cannot see the base image's conda torch, so
# torch would arrive only as an UNPINNED transitive dep of torchmetrics/dinov3.
# dinov3 is installed --no-deps (its requirements.txt is already satisfied below).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "pillow",
                 "omegaconf", "ftfy", "regex", "submitit", "termcolor",
                 "torchmetrics", "scikit-learn", "iopath", "pandas", "safetensors")
    .run_commands("git clone --depth 1 https://github.com/facebookresearch/dinov3 "
                  "/opt/dinov3 && pip install -e /opt/dinov3 --no-deps")
    # config + patches + dataset ONLY — pool tiles and the 1.1G teacher .pth live on
    # the VOLUME; without the ignore they'd be baked into the image on every deploy.
    .add_local_dir(SSL_DIR, "/opt/dapt_ssl",
                   ignore=["pool/**", "*.pth", "__pycache__/**", "modal/**"])
)

HARD_TIMEOUT_S = 3 * 3600      # 3h ceiling -> ~$6-7 on A100-40GB, comfortably < $30


@app.function(gpu="A100", volumes={"/vol": vol}, timeout=HARD_TIMEOUT_S, image=image,
              cpu=8, memory=32768)   # 8 vCPU for train.num_workers=8; 32 GiB RAM
def train(max_iters: int = 5000, batch: int = 32, lr_peak: float = 1e-4,
          seed: int = 0):
    """Pre-flight, then a single resumable DAPT run. Re-invoking resumes from the last
    DCP checkpoint (budget-safe: a timeout/kill loses no completed sweep points)."""
    repo = "/opt/dinov3"
    # guard: fail fast (unbilled-ish) if the image's python lacks CUDA torch
    import math
    import torch
    assert torch.cuda.is_available(), \
        "no CUDA torch in image -- fix image before burning GPU time"
    eff_lr = lr_peak * math.sqrt(batch * 1 / 1024)   # sqrt_wrt_1024, world=1
    print(f"[preflight] effective peak LR = {eff_lr:.2e} (want ~1-2e-5); "
          f"gpu={torch.cuda.get_device_name(0)}")
    assert 5e-6 <= eff_lr <= 5e-5, f"effective LR {eff_lr:.2e} outside continuation band"
    subprocess.run(["python", "/opt/dapt_ssl/apply_repo_patches.py", repo], check=True)

    cfg = "/opt/dapt_ssl/dinov3_dapt_vitl.yaml"
    # ---- pre-flight: build via the REAL config + assert the web weights load with
    # 0 missing params (mirrors dapt/ssl/local_smoke.py, which passes locally) ----
    pre = f"""
import sys, types, torch; sys.path.insert(0, '{repo}')
from dinov3.configs.config import get_cfg_from_args
from dinov3.models import build_model
cfg = get_cfg_from_args(types.SimpleNamespace(
    config_file='{cfg}', opts=[], output_dir=None))
gcs = cfg.crops.global_crops_size
teacher, _ = build_model(cfg.student, only_teacher=True,
    img_size=gcs if isinstance(gcs, int) else max(gcs), device='cuda')
sd = torch.load('/vol/dinov3_web_vitl_teacher.pth', map_location='cpu')['teacher']
bb = {{k[9:]: v for k, v in sd.items() if k.startswith('backbone.')}}
miss, unexp = teacher.load_state_dict(bb, strict=False)
params = {{n for n, _ in teacher.named_parameters()}}
miss_p = [m for m in miss if m in params]
print('PREFLIGHT missing params:', miss_p, '| unexpected:', unexp[:5])
assert not miss_p and not unexp, 'weight/arch mismatch -- fix before training'
print('PREFLIGHT OK (0 missing params)')
"""
    subprocess.run(["python", "-c", pre], check=True, cwd=repo)

    # ---- training (single node, 1 GPU). Resumes if <out_dir>/ckpt exists. ----
    # Seed-scoped output dir: prevents a different-seed invocation from silently
    # RESUMING another seed's DCP checkpoints. Resume = re-run with the SAME seed.
    out_dir = f"/vol/out_s{seed}"
    cmd = ["torchrun", "--nproc_per_node=1", "dinov3/train/train.py",
           "--config-file", cfg, "--output-dir", out_dir,
           "--seed", str(seed),               # -> setup_job/fix_random_seeds
           f"optim.epochs={max_iters // 500}",
           f"train.batch_size_per_gpu={batch}",
           f"train.seed={seed}",              # -> data-sampler stream (+iter offset)
           f"optim.lr={lr_peak}",   # v1 scheduler key; 'schedules.*' is NOT in schema
           # NOTE root must be pool/tiles, NOT pool/ — AridPool walks recursively and
           # pool/samples/ holds coverage-thumbnail PNGs that must never train.
           "train.dataset_path=AridPool:root=/vol/pool/tiles",
           "student.resume_from_teacher_chkpt=/vol/dinov3_web_vitl_teacher.pth"]
    subprocess.run(cmd, check=True, cwd=repo)
    vol.commit()
    print(f"training done; checkpoints under {out_dir}/ckpt/<iter>/")


@app.function(volumes={"/vol": vol}, timeout=1800, image=image, cpu=4, memory=16384)
def extract(seed: int = 0, iters: str = ""):
    """CPU-only: teacher-backbone .pth per DCP checkpoint -> /vol/export/.

    iters: comma-separated iteration dirs (e.g. "999,1999,4999"); default = all
    found under /vol/out_s<seed>/ckpt/. Then download with:
        modal volume get dinov3-dapt-arid-vol /export dapt/ssl/export
    """
    import glob
    import sys
    sys.path.insert(0, "/opt/dapt_ssl")
    from dcp_extract import extract as dcp_extract

    ckpt_root = f"/vol/out_s{seed}/ckpt"
    dirs = sorted(glob.glob(os.path.join(ckpt_root, "*")),
                  key=lambda p: int(os.path.basename(p)))
    if iters:
        keep = {i.strip() for i in iters.split(",")}
        dirs = [d for d in dirs if os.path.basename(d) in keep]
    if not dirs:
        raise SystemExit(f"no checkpoints found under {ckpt_root}")
    os.makedirs("/vol/export", exist_ok=True)
    for d in dirs:
        it = os.path.basename(d)
        out = f"/vol/export/teacher_s{seed}_i{it}.pth"
        if os.path.exists(out):
            print(f"skip {out} (exists)")
            continue
        dcp_extract(d, out)
    vol.commit()
    print("export done; files under /vol/export/")
