"""v3 DAPT — brand-new, isolated Modal app + volume (nothing shared with v1/v2).

Same generic DINOv3 image + repo patches as v1/v2 (reused from dapt/ssl/), but a fresh
app (`dinov3-dapt-v3`) and volume (`dinov3-dapt-v3-vol`), and P1K defaults (1000 iters,
DCP every 250, warmup 500). Pool = the v3 leakage-safe orthos (WON right50 etc.).

Runbook: dapt/ssl/SPEC.md, `## v3 study`. Nothing here launches automatically.
"""
import os
import subprocess

import modal

APP_NAME = "dinov3-dapt-v3"            # fresh, isolated from v1/v2
VOL_NAME = "dinov3-dapt-v3-vol"
HERE = os.path.dirname(os.path.abspath(__file__))
# shared generic SSL assets (yaml / repo patches / AridPool dataset / dcp_extract)
SHARED_SSL = os.path.abspath(os.path.join(HERE, "..", "..", "ssl"))

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "pillow",
                 "omegaconf", "ftfy", "regex", "submitit", "termcolor",
                 "torchmetrics", "scikit-learn", "iopath", "pandas", "safetensors")
    .run_commands("git clone --depth 1 https://github.com/facebookresearch/dinov3 "
                  "/opt/dinov3 && pip install -e /opt/dinov3 --no-deps")
    .add_local_dir(SHARED_SSL, "/opt/dapt_ssl",
                   ignore=["pool*/**", "*.pth", "export/**", "__pycache__/**",
                           "modal/**", "v3/**"])
)

HARD_TIMEOUT_S = 3600      # 1h ceiling — P1K trains in ~11 min; ~$0.5/seed


@app.function(gpu="A100", volumes={"/vol": vol}, timeout=HARD_TIMEOUT_S, image=image,
              cpu=8, memory=32768)
def train(max_iters: int = 1000, batch: int = 32, lr_peak: float = 1e-4,
          seed: int = 0, ckpt_period: int = 250):
    """P1K DAPT run on the v3 pool. Resumable; re-run with the same seed to continue."""
    repo = "/opt/dinov3"
    import math
    import torch
    assert torch.cuda.is_available(), "no CUDA torch in image"
    eff_lr = lr_peak * 4 * math.sqrt(batch * 1 / 1024)   # repo sqrt_wrt_1024 (x4)
    print(f"[preflight] effective peak LR = {eff_lr:.2e} (target ~1-2e-5); "
          f"gpu={torch.cuda.get_device_name(0)}")
    assert 5e-6 <= eff_lr <= 1e-4, f"effective LR {eff_lr:.2e} outside sane band"
    subprocess.run(["python", "/opt/dapt_ssl/apply_repo_patches.py", repo], check=True)

    cfg = "/opt/dapt_ssl/dinov3_dapt_vitl.yaml"
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
assert not miss_p and not unexp, ('weight/arch mismatch', miss_p, unexp[:5])
print('PREFLIGHT OK (0 missing params)')
"""
    subprocess.run(["python", "-c", pre], check=True, cwd=repo)

    out_dir = f"/vol/out_v3_s{seed}"
    cmd = ["torchrun", "--nproc_per_node=1", "dinov3/train/train.py",
           "--config-file", cfg, "--output-dir", out_dir,
           "--seed", str(seed),
           f"optim.epochs={max_iters // 500}",         # OFFICIAL_EPOCH_LENGTH=500
           f"train.batch_size_per_gpu={batch}",
           f"train.seed={seed}",
           f"optim.lr={lr_peak}",
           f"checkpointing.period={ckpt_period}",       # DCP at 250/500/750/1000
           "train.dataset_path=AridPool:root=/vol/pool/tiles",
           "student.resume_from_teacher_chkpt=/vol/dinov3_web_vitl_teacher.pth"]
    subprocess.run(cmd, check=True, cwd=repo)
    vol.commit()
    print(f"training done; checkpoints under {out_dir}/ckpt/<iter>/")


@app.function(volumes={"/vol": vol}, timeout=1800, image=image, cpu=4, memory=16384)
def extract(seed: int = 0, iters: str = ""):
    """CPU-only teacher-backbone extraction -> /vol/export/teacher_v3_s<seed>_i<it>.pth."""
    import glob
    import sys
    sys.path.insert(0, "/opt/dapt_ssl")
    from dcp_extract import extract as dcp_extract

    ckpt_root = f"/vol/out_v3_s{seed}/ckpt"
    dirs = sorted(glob.glob(os.path.join(ckpt_root, "*")),
                  key=lambda p: int(os.path.basename(p)))
    if iters:
        keep = {i.strip() for i in iters.split(",")}
        dirs = [d for d in dirs if os.path.basename(d) in keep]
    if not dirs:
        raise SystemExit(f"no checkpoints under {ckpt_root}")
    os.makedirs("/vol/export", exist_ok=True)
    for d in dirs:
        it = os.path.basename(d)
        out = f"/vol/export/teacher_v3_s{seed}_i{it}.pth"
        if os.path.exists(out):
            print(f"skip {out} (exists)")
            continue
        dcp_extract(d, out)
    vol.commit()
    print("export done; files under /vol/export/")
