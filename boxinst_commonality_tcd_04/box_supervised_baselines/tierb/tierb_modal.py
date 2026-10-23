"""Tier B on Modal: B1 BoxInst R-101 3x, B2 BoxTeacher Swin-B 3x, B3 MAL ViT-MAE-B labeler.
Protocol and run book: TIERB.md. Local control: tierb_ctl.py (never `modal run` this file for
anything long -- deploy, then spawn).

ROBUSTNESS (every step):
  * deployed app + spawned calls: nothing depends on the laptop staying awake;
  * every step is idempotent and resumes from the volume (DONE markers, per-tile outputs, atomic
    training checkpoints, `last_checkpoint` repaired if torn, per-tile stitch checkpoints);
  * GPU steps retry on failure (Modal also restarts preempted inputs, with the same call id, so the
    heartbeat lock is re-entrant); a duplicate spawn of a running step exits instead of racing it;
  * a WATCHDOG kills any child whose log and progress have both been silent for STALL_S; NCCL async
    error handling turns a hung collective into a crash; either way the attempt counts as a failure,
    and 3 consecutive failures from the same checkpoint mark the step FAILED (no retry loop);
  * the volume is committed every 2 min, at every new checkpoint/best, and strictly at step end;
  * one STOP flag halts all spend: B1/B2 trainers checkpoint and exit (killed if not out in 15 min),
    every other child is killed (all outputs are per tile / per epoch, so nothing is lost), and no
    new step starts;
  * the LACE gate (TIERB.md section 7) sets STOP and a persistent GATE_TRIGGERED latch that only an
    explicit, logged override releases.

THE CHAIN. Each step, on success, spawns the next in PLAN unless STOP/gate latch is set or the
chain's `until` index is reached. tierb_ctl.py starts it with until=HOLD (data, weights, all gates,
all smokes, then a pause to review projected cost); `tierb_ctl.py continue` releases it.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time

import modal

APP = "tcd04-tierb"
HERE = os.path.dirname(os.path.abspath(__file__))           # .../box_supervised_baselines/tierb
BSB = os.path.dirname(HERE)
PKG = os.path.dirname(BSB)                                   # boxinst_commonality_tcd_04
REPO = os.path.dirname(PKG)
STUBS = os.path.join(PKG, "modal_tcd_multiseed", "phase4", "stubs")
STUB_FILES = {"boxinst": ("__init__.py", "cache_feats.py"),           # static: this module is
              "boxinst_commonality": ("__init__.py", "em.py"),         # re-imported INSIDE the
              "boxinst_tcd": ("__init__.py", "build_canopy.py",        # container, where these
                              "cache.py", "prepare.py")}               # paths do not exist
P = "/root/proj"
PKG_R = f"{P}/boxinst_commonality_tcd_04"

BT_REPO, BT_COMMIT = "https://github.com/hustvl/BoxTeacher.git", "6692d9213df1de96ee59338634551b70319a5cb2"
MAL_REPO, MAL_COMMIT = "https://github.com/NVlabs/mask-auto-labeler.git", "0ab9be4fe6b17f3e21b5829f88db001ef2bd665c"
WEIGHTS = {
    "BoxInst_MS_R_101_3x.pth": "https://huggingface.co/tianzhi/AdelaiDet-BoxInst/resolve/main/BoxInst_MS_R_101_3x.pth",
    "boxteacher_swin_base_aug_3x.pth": "https://github.com/hustvl/BoxTeacher/releases/download/v1.1/boxteacher_swin_base_aug_3x.pth",
    "mal_vit-mae-base-16_coco_epoch6.ckpt": "gdrive:1VEhlZV-McaizuPBQxNj6x-Ip0xemeGdB",
}
GPU8 = "A100-40GB:8"                        # budget: 40 GB (gate G5 proves the real per-GPU batch fits)
GPU1 = "A100-40GB"
GPU_SMOKE, N_SMOKE = "A100-40GB:2", 2       # smokes: 2 GPUs at the released per-GPU batch
SMOKE_TILES = 16                            # smoke predict: first 16 val tiles
# Modal list prices (USD), modal.com/pricing checked 2026-10-15: A100-40GB $0.000583/s, CPU
# $0.0000131/physical core/s, memory $0.00000222/GiB/s; CPU and memory are billed ON TOP of the GPU.
USD_H = {"gpu": 0.000583 * 3600, "core": 0.0000131 * 3600, "gib": 0.00000222 * 3600}


def usd_per_hour(gpus, cores, mem_mib):
    return gpus * USD_H["gpu"] + cores * USD_H["core"] + mem_mib / 1024 * USD_H["gib"]


SPEC = {"gate": (1, 8, 49152), "smoke": (N_SMOKE, 12, 65536), "gpu8": (8, 24, 131072),
        "stage": (0, 16, 65536), "weights": (0, 4, 16384), "finalize": (0, 32, 131072),
        "status": (0, 1, 2048)}
RATE = {k: usd_per_hour(*v) for k, v in SPEC.items()}      # gpu8 ~ $18.9/h, smoke ~ $5.3/h
USD_8GPU_D2 = USD_8GPU_MAL = RATE["gpu8"]
BUDGET_USD = 75.0                           # Tom, 2026-10-15
HARD_STOP_USD = 72.0                        # live cap: leaves room for a graceful exit + builds
STARTUP_MIN = 4.0                           # billed container start before run_managed's clock
STALL_S = 45 * 60                          # watchdog: silence (no log line, no progress) -> kill
STOP_GRACE_S = 15 * 60                     # graceful STOP (B1/B2 train) -> kill if not out by then

app = modal.App(APP)
vol = modal.Volume.from_name("tcd04-tierb-vol", create_if_missing=True)
hfvol = modal.Volume.from_name("canopyai-deepforest-data")     # HF restor/tcd cache, read only


# ------------------------------------------------------------------ images
def _add_tierb(img):
    img = img.add_local_dir(HERE, "/root/tierb", copy=True,
                            ignore=["tierb_data/**", "gates/**", "**/__pycache__/**", "results/**",
                                    "calls.jsonl"])
    return img.add_local_dir(os.path.join(HERE, "tierb_data"), "/root/tierb_data", copy=True)


def _add_repo_files(img, rels):
    for rel in rels:
        img = img.add_local_file(os.path.join(REPO, rel), f"{P}/{rel}", copy=True)
    return img


# STAGE: phase4_modal.py's recipe for reading LACE's own HF copy (datasets 4.0.0 + offline cache).
img_stage = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets==4.0.0", "numpy==2.2.6", "pillow==12.3.0", "requests", "gdown==5.2.0")
    .env({"HF_HOME": "/hfdata/hf_cache", "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "1",
          "PYTHONPATH": "/root"})
)
img_stage = _add_tierb(img_stage)

# SCORER: the frozen scorer/stitcher/gtbox code in the SAME environment that produced every
# published row (local .venv: py3.10.20, numpy 2.2.6, Pillow 12.3.0, pycocotools 2.0.11 -- the PIL
# rasteriser decides GT pixels). finalize re-scores a stored LACE row and requires exact equality.
img_cpu = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.13.0", "numpy==2.2.6", "pillow==12.3.0", "pycocotools==2.0.11",
                 "contourpy==1.3.2", "transformers==5.12.1",
                 extra_index_url="https://download.pytorch.org/whl/cpu")
    .env({"PYTHONPATH": f"{P}:/root"})
)
img_cpu = _add_repo_files(img_cpu, [
    "dapt/__init__.py", "dapt/backbone.py", "dapt/targets.py", "dapt/decode.py", "dapt/eval.py",
    "dapt/head.py",
    *[f"boxinst_commonality_tcd_04/{f}" for f in (
        "__init__.py", "detector.py", "train_detector_tiles.py", "evaluate.py", "em.py",
        "prepare_test.py", "cache_test.py", "cache_train_tiles.py", "test_gt.json", "score_coco.py",
        "train_tiles_gt.json", "modal_tcd_multiseed/phase4/val_gt.json",
        "box_supervised_baselines/__init__.py", "box_supervised_baselines/gtbox_lib.py",
        "native_raster/stitch_res.py", "native_raster/stitch_seam.py", "native_raster/run_stitch_seam.py",
        "native_raster/gtbox_guided_raw/gtbox_lace_guided_r2048.jsonl",
        "native_raster/audit_2026-10-14/selvabox_evidence/gtbox_sam_persub.json",
        "native_raster/preds_lace_s0_r2048_cgf_r12_tau035_guidedrun.json",
        "native_raster/scored_lace_s0_r2048_cgf_r12_tau035_guidedrun.json",
        "native_raster/preds_lace_sparse_s0_r2048_cgf_r12_tau035_guidedrun.json",
        "native_raster/scored_lace_sparse_s0_r2048_cgf_r12_tau035_guidedrun.json")],
    "data/tcd_sparse/sparse_gt.json",
])
for _pkg, _files in STUB_FILES.items():
    for _f in _files:
        img_cpu = img_cpu.add_local_file(os.path.join(STUBS, _pkg, _f), f"{P}/{_pkg}/{_f}", copy=True)
img_cpu = _add_tierb(img_cpu)

# CUDA 11.3 base images still list NVIDIA's pre-2022 apt repo (rotated key) -> drop it before apt.
_APT_FIX = "rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia-ml*.list"
_PIP = "python -m pip install 'pip==24.0'"   # PL 1.8.1's metadata ('torch (>=1.9.*)') is refused by pip>=24.1

# GPU d2: BoxTeacher repo (vendored detectron2 0.6 + AdelaiDet) @6692d92, torch 1.12.1/cu113, py3.10.
_SED = [  # build-time fixes, each a known upstream install bug (TIERB.md section 6)
    "sed -i 's/^PROJECTS = {/PROJECTS = {}  # tierb: projects absent upstream (BoxTeacher issue #9)\\n_UNUSED = {/' setup.py",
    "sed -i '/ml_nms/d' AdelaiDet/adet/layers/csrc/vision.cpp && rm -rf AdelaiDet/adet/layers/csrc/ml_nms",
    "sed -i '/^from turtle import color/d' projects/BoxTeacher/boxteacher/dataset_mapper.py",
    "sed -i '/^from cv2 import log/d' projects/BoxTeacher/boxteacher/ema_hook.py",
    "sed -i 's/self.trainer.model.module/(self.trainer.model.module if hasattr(self.trainer.model, \"module\") else self.trainer.model)/g' projects/BoxTeacher/boxteacher/ema_hook.py",
]
img_d2 = (
    modal.Image.from_registry("nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04", add_python="3.10")
    .run_commands(_APT_FIX, _PIP)
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC", "FORCE_CUDA": "1",
          "TORCH_CUDA_ARCH_LIST": "8.0", "CUDA_HOME": "/usr/local/cuda", "CC": "gcc", "CXX": "g++"})
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0", "ninja-build")
    .pip_install("torch==1.12.1+cu113", "torchvision==0.13.1+cu113",
                 extra_index_url="https://download.pytorch.org/whl/cu113")
    .pip_install("setuptools==59.5.0", "wheel")
    .pip_install("numpy==1.23.5", "Pillow==9.5.0", "fvcore==0.1.5.post20221221", "iopath==0.1.9",
                 "pycocotools==2.0.7", "opencv-python-headless==4.8.1.78", "scikit-image==0.19.3",
                 "scipy==1.10.1", "timm==0.6.13", "termcolor", "yacs", "tabulate", "cloudpickle",
                 "tqdm", "tensorboard", "matplotlib==3.7.3", "omegaconf==2.1.2", "hydra-core==1.1.2",
                 "packaging", "pydot", "future", "shapely", "rapidfuzz", "editdistance")
    .run_commands(
        f"git clone {BT_REPO} /opt/BoxTeacher && cd /opt/BoxTeacher && git checkout {BT_COMMIT}",
        *[f"cd /opt/BoxTeacher && {c}" for c in _SED],
        "cd /opt/BoxTeacher && python setup.py build develop --no-deps",
        "cd /opt/BoxTeacher/AdelaiDet && python setup.py build develop --no-deps",
        "python -m pip install --no-deps --force-reinstall numpy==1.23.5 Pillow==9.5.0",
        "cd /opt/BoxTeacher/projects/BoxTeacher && python -c 'import numpy, PIL; assert numpy.__version__ == \"1.23.5\" "
        "and PIL.__version__ == \"9.5.0\"; import detectron2, adet, boxteacher; from adet import _C; "
        "from detectron2 import _C; from adet.config import get_cfg; print(\"d2 import OK\")'",
    )
    .env({"PYTHONPATH": f"/opt/BoxTeacher/projects/BoxTeacher:/root:{P}",
          "NCCL_ASYNC_ERROR_HANDLING": "1"})
)
img_d2 = _add_repo_files(img_d2, ["boxinst_commonality_tcd_04/train_tiles_gt.json"])
img_d2 = _add_tierb(img_d2)

# GPU mal: MAL @0ab9be4, torch 1.12.1/cu113, PL 1.8.1, fairscale 0.4.13, mmcv-full 1.7.0, py3.10.
img_mal = (
    modal.Image.from_registry("nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04", add_python="3.10")
    .run_commands(_APT_FIX, _PIP)
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC", "TORCH_HOME": "/opt/torch_home"})
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0")
    .pip_install("torch==1.12.1+cu113", "torchvision==0.13.1+cu113",
                 extra_index_url="https://download.pytorch.org/whl/cu113")
    .pip_install("numpy==1.23.5", "Pillow==9.5.0", "pytorch_lightning==1.8.1", "torchmetrics==0.10.2",
                 "timm==0.6.13", "lmdb", "pycocotools==2.0.7", "opencv-python-headless==4.5.5.64",
                 "scipy==1.10.1", "einops==0.6.0", "tensorboard", "protobuf==3.20.3",
                 "fairscale==0.4.13", "addict", "pyyaml", "packaging", "yapf==0.32.0")
    # --no-deps: mmcv-full 1.7.0 declares opencv-python>=3, which would pull OpenCV 5 + numpy 2
    .run_commands("python -m pip install --no-deps mmcv-full==1.7.0 "
                  "-f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html")
    .run_commands(
        f"git clone {MAL_REPO} /opt/mal && cd /opt/mal && git checkout {MAL_COMMIT}",
        "python -m pip install --no-deps --force-reinstall numpy==1.23.5",
        # MAL fetches MAE weights from dl.fbaipublicfiles.com at EVERY model build: cache them now
        "cd /opt/mal && python -c \"import torch.utils.model_zoo as mz; from models.vision_transformers "
        "import urls_dic; print(len(mz.load_url(urls_dic['vit-mae-base/16'], map_location='cpu')))\"",
        "cd /opt/mal && python -c 'import numpy, cv2; assert numpy.__version__ == \"1.23.5\" and "
        "cv2.__version__.startswith(\"4.5.5\"), (numpy.__version__, cv2.__version__); "
        "from models.mal import MAL; import pytorch_lightning, fairscale, mmcv; print(\"mal import OK\")'",
    )
    .env({"PYTHONPATH": f"/opt/mal:/root:{P}", "NCCL_ASYNC_ERROR_HANDLING": "1"})
)
img_mal = _add_repo_files(img_mal, ["boxinst_commonality_tcd_04/modal_tcd_multiseed/phase4/val_gt.json"])
img_mal = _add_tierb(img_mal)

VOLS = {"/vol": vol}
RETRY = modal.Retries(max_retries=8, backoff_coefficient=1.0, initial_delay=30.0)
RETRY1 = modal.Retries(max_retries=1, backoff_coefficient=1.0, initial_delay=30.0)

# ------------------------------------------------------------------ the plan
PLAN = [
    ("stage", None), ("fetch_weights", None),
    ("gate", "mal"), ("gate", "boxteacher"), ("gate", "boxinst"),
    ("smoke", "mal"), ("smoke", "boxteacher"),             # B1 shares B2's d2 machinery (budget)
    ("HOLD", None),                                         # review gates, smokes, projected cost
    ("train", "mal"), ("predict", "mal"), ("finalize", "mal"),
    ("train", "boxteacher"), ("predict", "boxteacher"), ("finalize", "boxteacher"),
    ("train", "boxinst"), ("predict", "boxinst"), ("finalize", "boxinst"),
]
HOLD_INDEX = [i for i, (s, _) in enumerate(PLAN) if s == "HOLD"][0]


def FN_NAME(step, m):
    if step in ("stage", "fetch_weights", "finalize"):
        return step
    fam = "mal" if m == "mal" else "d2"
    return {"gate": f"{fam}_gpu1", "smoke": f"{fam}_gpu2_smoke", "train": f"{fam}_gpu8_train",
            "predict": f"{fam}_gpu8_predict"}[step]


def skey(step, m):
    return f"{step}:{m}" if m else step


def _ctl():
    return modal.Dict.from_name("tcd04-tierb-control", create_if_missing=True)


def _status(key, **kw):
    try:
        d = _ctl()
        cur = d.get("status", {}) or {}
        cur[key] = dict(cur.get(key, {}), **kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        d.put("status", cur)
    except Exception as e:
        print(f"[status] WARNING could not record {key}: {e}", flush=True)


def _reload():
    try:
        vol.reload()              # warm-container reuse: see what other steps committed
    except Exception as e:
        print(f"[vol] reload skipped ({e})", flush=True)


def spend_total(d=None):
    d = d or _ctl()
    return round(sum(e["usd"] for e in (d.get("spend", None) or [])), 2)


def record_spend(key, minutes, usd_h, d=None):
    """Ledger of billed minutes x container rate (incl. CPU + memory), every attempt, every step."""
    d = d or _ctl()
    led = d.get("spend", None) or []
    led.append({"key": key, "minutes": round(minutes, 1), "usd_h": round(usd_h, 2),
                "usd": round(minutes / 60 * usd_h, 2), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    d.put("spend", led)
    return spend_total(d)


def advance(i):
    """Called by step i on success: spawn step i+1 unless stopped / gate-latched / held / inactive /
    over budget."""
    sys.path.insert(0, "/root")
    from tierb import tierb_common as TC
    d = _ctl()
    chain = d.get("chain", None) or {}
    nxt = i + 1
    if spend_total(d) >= HARD_STOP_USD:
        TC.set_stop(f"budget: ${spend_total(d)} spent of ${BUDGET_USD} (hard stop at ${HARD_STOP_USD})")
    if TC.stop_requested(d):
        _status("chain", state="STOPPED", at=nxt, reason=d.get("STOP", None))
        return None
    blk = TC.gate_blocks(d)
    if blk:
        _status("chain", state="GATE_LATCHED", at=nxt, reason=blk)
        return None
    if not chain.get("active") or nxt >= len(PLAN) or nxt > chain.get("until", -1):
        _status("chain", state="IDLE" if nxt < len(PLAN) else "COMPLETE", next=nxt,
                next_step=PLAN[nxt] if nxt < len(PLAN) else None)
        return None
    if PLAN[nxt][0] == "HOLD":
        _status("chain", state="HOLD", next=nxt + 1, note="review gates/smokes; tierb_ctl.py continue")
        return None
    return spawn_step(nxt)


def spawn_step(i):
    step, m = PLAN[i]
    fn = globals()[FN_NAME(step, m)]
    call = fn.spawn(i) if m is None else fn.spawn(i, m)
    _status(skey(step, m), state="SPAWNED", call=call.object_id, plan_index=i)
    return call.object_id


class StepFailed(Exception):
    """Deterministic failure: recorded, chain stopped, NOT retried."""


class SmokeHalted(Exception):
    """A smoke phase was stopped (global STOP) or lost to a duplicate: not a failure."""


def _phase_ok(r):
    if r["status"] in ("stopped", "stopped_before_start", "duplicate", "failed_no_progress"):
        raise SmokeHalted(r["status"])
    return r


def _failed(key, e, retrying):
    _status(key, state="ERROR_RETRYING" if retrying else "FAILED", error=repr(e)[:2000])


# ------------------------------------------------------------------ managed subprocess
def run_managed(key, cmd, cwd=None, progress=None, token=None, stop="kill", watch=None,
                max_failures=3, poll=15, stall_s=STALL_S, usd_h=0.0):
    """Run `cmd` under: a re-entrant heartbeat lock (released on EVERY path); volume commits every
    2 min, on request whenever `token()` (a checkpoint/best fingerprint) changes, and strictly at the
    end -- requested, never awaited, so the poll loop is always free to run `watch` and act on STOP;
    STOP handling ("kill", or "graceful" = touch /tmp/tierb_STOP then kill after STOP_GRACE_S); a
    watchdog that kills the child after `stall_s` without a log line, a progress change or a new
    checkpoint; a full log on the volume; attempt bookkeeping that counts real failures (rc != 0,
    stalls, and attempts that vanished without rc AND without checkpoint progress)."""
    sys.path.insert(0, "/root")
    from tierb import tierb_common as TC
    tok = token or (lambda: None)
    lock = TC.StageLock(key)
    ok, cur = lock.acquire()
    if not ok:
        print(f"[managed] {key}: another live call holds the lock ({cur}); exiting", flush=True)
        _status(key + "#dup", state="DUPLICATE_EXITED", holder=str(cur))
        return {"status": "duplicate", "holder": cur}
    d = _ctl()
    proc, com, logf, th = None, None, None, None
    try:
        if TC.stop_requested(d):
            return {"status": "stopped_before_start", "stop": d.get("STOP", None)}
        for f in ("/tmp/tierb_STOP", "/tmp/tierb_STOPPED"):
            if os.path.exists(f):
                os.remove(f)
        tag0 = tok()
        attempts = d.get(f"attempts:{key}", []) or []
        if attempts and attempts[-1].get("rc") is None:        # previous container vanished
            attempts[-1]["rc"] = "vanished"
            attempts[-1]["progressed"] = attempts[-1].get("ckpt") != tag0
        fails = 0
        for a in reversed(attempts):
            failed = (a.get("rc") not in (0, None) and not a.get("stopped")
                      and not (a.get("rc") == "vanished" and a.get("progressed")))
            if failed and a.get("ckpt") == tag0:
                fails += 1
            else:
                break
        if fails >= max_failures:
            _status(key, state="FAILED", reason=f"{fails} consecutive failures from checkpoint {tag0}")
            return {"status": "failed_no_progress", "attempts": attempts}
        attempts.append({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "ckpt": tag0, "rc": None})
        d.put(f"attempts:{key}", attempts)

        com = TC.Committer(vol, period=120).start()
        os.makedirs(f"{TC.RUNS}/logs", exist_ok=True)
        logp = f"{TC.RUNS}/logs/{key.replace(':', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logf = open(logp, "a", buffering=1, errors="replace")
        print(f"[managed] {key}: {' '.join(cmd)}\n[managed] log -> {logp}", flush=True)
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", bufsize=1, start_new_session=True)
        last_line = [time.time()]

        def pump():
            while True:
                try:
                    line = proc.stdout.readline()
                except Exception as e:             # keep draining whatever happens
                    sys.stdout.write(f"[pump] read error {e}\n")
                    continue
                if not line:
                    break
                last_line[0] = time.time()
                try:
                    sys.stdout.write(line)
                    logf.write(line)
                except Exception:
                    pass
        th = threading.Thread(target=pump, daemon=True)
        th.start()

        def killpg():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        last_tag, last_prog, last_prog_t = tag0, None, time.time()
        stop_at, stalled, t0 = None, False, time.time()
        spent0 = spend_total(d) + STARTUP_MIN / 60 * usd_h
        while proc.poll() is None:
            time.sleep(poll)
            try:
                info = None
                if progress and os.path.exists(progress):
                    try:
                        info = json.load(open(progress))
                    except Exception:
                        pass
                if watch:                          # first: rehearsal kills must never wait on I/O
                    watch(proc, info)
                live = spent0 + (time.time() - t0) / 3600 * usd_h
                if usd_h and live >= HARD_STOP_USD and not TC.stop_requested(d):
                    TC.set_stop(f"budget: live spend ${live:.2f} reached the ${HARD_STOP_USD} hard stop "
                                f"(budget ${BUDGET_USD}) during {key}; everything is resumable")
                if stop_at is None and TC.stop_requested(d):
                    stop_at = time.time()
                    print(f"[managed] STOP seen -> {stop}", flush=True)
                    if stop == "graceful":
                        open("/tmp/tierb_STOP", "w").write("stop")
                    else:
                        killpg()
                if stop_at is not None and time.time() - stop_at > STOP_GRACE_S:
                    killpg()
                if info != last_prog:
                    last_prog, last_prog_t = info, time.time()
                tag = tok()
                if tag != last_tag:
                    com.request()                  # background thread commits; we do not wait
                    last_tag, last_prog_t = tag, time.time()     # a new checkpoint IS progress
                lock.beat(info)
                if time.time() - max(last_line[0], last_prog_t) > stall_s:
                    print(f"[managed] WATCHDOG: {stall_s}s without log or progress -> kill", flush=True)
                    stalled = True
                    killpg()
            except Exception as e:                 # a control-plane hiccup must not orphan the child
                print(f"[managed] poll error {type(e).__name__}: {e}", flush=True)
        rc = proc.wait()
        stopped = stop_at is not None or os.path.exists("/tmp/tierb_STOPPED")
        commit_err = None
    finally:
        if proc is not None:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # orphaned DDP ranks of a dead rank 0
            except ProcessLookupError:
                pass
        if th is not None:
            th.join(timeout=30)
        if logf is not None:
            logf.close()
        if com is not None:
            try:
                com.close(strict=True)
            except Exception as e:
                commit_err = e
        if usd_h and proc is not None:
            try:
                record_spend(key, STARTUP_MIN + (time.time() - t0) / 60, usd_h, d)
            except Exception as e:
                print(f"[managed] could not record spend: {e}", flush=True)
        if proc is not None:
            try:
                att = d.get(f"attempts:{key}", []) or []
                if att:
                    att[-1].update(rc="stalled" if locals().get("stalled") else proc.returncode,
                                   end=time.strftime("%Y-%m-%d %H:%M:%S"),
                                   stopped=bool(locals().get("stopped")), end_ckpt=tok(),
                                   commit_error=repr(commit_err) if commit_err else None)
                    d.put(f"attempts:{key}", att)
            except Exception as e:
                print(f"[managed] could not record attempt: {e}", flush=True)
        lock.release()
    if commit_err is not None:
        raise RuntimeError(f"{key}: final volume commit failed: {commit_err}")
    status = "stopped" if stopped else ("ok" if rc == 0 and not stalled else "error")
    return {"status": status, "rc": "stalled" if stalled else rc, "log": logp,
            "minutes": round((time.time() - t0) / 60, 1)}


def _finish(i, key, res, done_marker=None):
    """error -> raise (Modal retries; the next attempt resumes); FAILED/stopped/dup -> record, stop;
    ok -> strict commit already done -> advance."""
    if res["status"] in ("duplicate", "stopped_before_start"):
        return res
    if res["status"] == "failed_no_progress":
        return res
    if res["status"] == "stopped":
        _status(key, state="STOPPED", log=res.get("log"))
        return res
    if res["status"] == "error" or (done_marker and not os.path.exists(done_marker)):
        _status(key, state="ERROR_RETRYING", log=res.get("log"), rc=str(res.get("rc")))
        raise RuntimeError(f"{key} failed rc={res.get('rc')}; see {res.get('log')}")
    _status(key, state="DONE", log=res.get("log"))
    advance(i)
    return res


# ------------------------------------------------------------------ CPU steps
@app.function(image=img_stage, volumes={"/vol": vol, "/hfdata": hfvol}, cpu=16, memory=65536,
              timeout=6 * 3600, retries=RETRY)
def stage(i: int):
    """HF restor/tcd (LACE's own copy, joined by image_id as phase4_modal.py did) -> /vol/rgb/<tid>.png
    for the 1,559 unique tiles; every pixel hash compared with the local-file hash in
    splits_manifest.json (the one local_missing tile is recorded from HF). Resumable."""
    sys.path.insert(0, "/root")
    import hashlib
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from datasets import load_dataset
    from PIL import Image
    from tierb import tierb_common as TC
    key = "stage"
    t_start = time.time()
    try:
        _reload()
        if TC.is_done("_shared", "stage"):
            _status(key, state="DONE(already)"); advance(i); return "already"
        man = TC.manifest()
        tiles = {}
        for split in ("train", "val", "test", "sparse"):
            for t, h in man["hashes"][split].items():
                tiles.setdefault(t, h)
        print(f"[stage] {len(tiles)} unique tiles", flush=True)
        idx, ds = {}, {}
        for split in ("train", "test"):
            ds[split] = load_dataset("restor/tcd", split=split)
            for k, iid in enumerate(ds[split]["image_id"]):
                idx[(split, int(iid))] = k
        os.makedirs(TC.RGB_DIR, exist_ok=True)
        partial = f"{TC.RGB_DIR}/_stage_partial"
        os.makedirs(partial, exist_ok=True)
        com = TC.Committer(vol, period=120).start()
        lk = threading.Lock()

        def one(t):
            pr = f"{partial}/{t}.json"
            if os.path.exists(pr) and os.path.exists(TC.rgb_path(t)):
                return json.load(open(pr))
            h = tiles[t]
            with lk:                                   # Arrow row access serialised; work is not
                row = ds[h["hf_split"]][idx[(h["hf_split"], h["image_id"])]]
                img = row["image"]
            arr = np.asarray(img.convert("RGB"))
            assert arr.shape == (2048, 2048, 3), (t, arr.shape)
            hh = hashlib.sha256(arr.tobytes()).hexdigest()
            if "local_missing" not in h:
                assert hh == h["rgb_sha256"], f"{t}: HF pixels != local file ({hh[:12]} vs {h['rgb_sha256'][:12]})"
            tmp = TC.rgb_path(t) + ".tmp.png"
            Image.fromarray(arr).save(tmp, compress_level=3)
            back = np.asarray(Image.open(tmp).convert("RGB"))
            assert hashlib.sha256(back.tobytes()).hexdigest() == hh, f"{t}: PNG round trip"
            os.replace(tmp, TC.rgb_path(t))
            r = {"rgb_sha256": hh, "image_id": h["image_id"], "hf_split": h["hf_split"],
                 "matched_local": "local_missing" not in h,
                 "png_sha256": hashlib.sha256(open(TC.rgb_path(t), "rb").read()).hexdigest()}
            TC.atomic_json(pr, r)
            return r
        out = {}
        with ThreadPoolExecutor(8) as ex:
            for k, (t, r) in enumerate(zip(sorted(tiles), ex.map(one, sorted(tiles)))):
                out[t] = r
                if (k + 1) % 100 == 0:
                    print(f"[stage] {k+1}/{len(tiles)}", flush=True)
        rec = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "source": "HF restor/tcd via canopyai-deepforest-data (LACE's copy)",
               "summary": {"n": len(out), "matched_local": sum(r["matched_local"] for r in out.values()),
                           "hf_only": sorted(t for t, r in out.items() if not r["matched_local"])},
               "tiles": out}
        assert rec["summary"]["matched_local"] == len(out) - len(man["checks"]["local_missing"])
        TC.atomic_json(TC.STAGE_REC, rec, indent=1)
        TC.mark_done("_shared", "stage", rec["summary"])
        com.close(strict=True)
        record_spend(key, STARTUP_MIN + (time.time() - t_start) / 60, RATE["stage"])
        _status(key, state="DONE", **rec["summary"])
        advance(i)
        return rec["summary"]
    except Exception as e:
        _failed(key, e, retrying=not isinstance(e, AssertionError))
        if isinstance(e, AssertionError):              # a hash mismatch will not fix itself
            return {"status": "FAILED", "error": repr(e)}
        raise


@app.function(image=img_stage, volumes=VOLS, cpu=4, memory=16384, timeout=2 * 3600, retries=RETRY)
def fetch_weights(i: int):
    """Released COCO box-supervised checkpoints -> /vol/weights, size-checked, sha256 recorded."""
    sys.path.insert(0, "/root")
    import hashlib
    import requests
    from tierb import tierb_common as TC
    key = "fetch_weights"
    t_start = time.time()
    try:
        _reload()
        os.makedirs(f"{TC.VOL}/weights", exist_ok=True)
        rec = {}
        for name, url in WEIGHTS.items():
            p = f"{TC.VOL}/weights/{name}"
            if not os.path.exists(p):
                tmp = p + ".part"
                if url.startswith("gdrive:"):
                    import gdown
                    out = gdown.download(id=url.split(":", 1)[1], output=tmp, quiet=False)
                    assert out and os.path.getsize(tmp) > 100 * 2 ** 20, f"{name}: gdown failed (quota page?)"
                else:
                    with requests.get(url, stream=True, timeout=600) as r:
                        r.raise_for_status()
                        want = int(r.headers.get("Content-Length", 0))
                        with open(tmp, "wb") as f:
                            for chunk in r.iter_content(1 << 22):
                                f.write(chunk)
                    if want:
                        assert os.path.getsize(tmp) == want, f"{name}: {os.path.getsize(tmp)} != {want} bytes"
                os.replace(tmp, p)
                vol.commit()
            rec[name] = {"url": url, "bytes": os.path.getsize(p),
                         "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
            print(f"[weights] {name} {rec[name]['bytes']/2**20:.0f} MB {rec[name]['sha256'][:12]}", flush=True)
        TC.atomic_json(f"{TC.VOL}/weights/WEIGHTS.json", rec, indent=1)
        TC.mark_done("_shared", "fetch_weights", rec)
        vol.commit()
        record_spend(key, STARTUP_MIN + (time.time() - t_start) / 60, RATE["weights"])
        _status(key, state="DONE")
        advance(i)
        return rec
    except Exception as e:
        _failed(key, e, retrying=True)
        raise


# ------------------------------------------------------------------ GPU steps: d2 (B1, B2)
D2_CWD = "/opt/BoxTeacher/projects/BoxTeacher"


def _d2(cmd, m, extra=()):
    return [sys.executable, "-m", "tierb.d2.d2_run", cmd, "--method", m, *extra]


def _d2_token(run):
    """Checkpoint fingerprint: the tagged checkpoint + the current best (commit on either)."""
    def f():
        a = os.path.join(run, "train", "last_checkpoint")
        b = os.path.join(run, "best.json")
        return (open(a).read().strip() if os.path.exists(a) else None,
                os.path.getmtime(b) if os.path.exists(b) else None)
    return f


@app.function(image=img_d2, volumes=VOLS, gpu=GPU1, cpu=8, memory=49152, timeout=2 * 3600,
              retries=RETRY1)
def d2_gpu1(i: int, m: str):
    sys.path.insert(0, "/root")
    from tierb import tierb_common as TC
    key = f"gate:{m}"
    _reload()
    if TC.is_done(m, "gate"):
        _status(key, state="DONE(already)"); advance(i); return "already"
    res = run_managed(key, _d2("gate", m), cwd=D2_CWD, stall_s=30 * 60, usd_h=RATE["gate"])
    return _finish(i, key, res, TC.done_marker(m, "gate"))


@app.function(image=img_d2, volumes=VOLS, gpu=GPU_SMOKE, cpu=12, memory=65536, timeout=3 * 3600,
              retries=RETRY1)
def d2_gpu2_smoke(i: int, m: str):
    """Preemption/STOP/resume rehearsal at full batch, then timing -> projected cost.
      A: train; at iter >= 50 set STOP -> clean checkpoint at a multiple of 10 (59/69) + exit 0.
      B: resume; SIGKILL while the epoch-end eval of iter 80 is running (tag = model_0000079, 80
         not yet in evals) = a preemption DURING eval.
      C: resume at 80 -> before_train re-runs the lost eval of 80 -> finish 120.
      P: predict the first 16 val tiles x 2 geometries, SIGKILLed once 4 tiles are written,
         resumed to completion.
    Runs on 2 GPUs at the released 2 images/GPU (same per-GPU work and memory as the real 8 x 2
    run); DDP collectives incl. the P7 empty-GPU path are exercised. Projections scale the measured
    eval/predict times by 2/8 GPUs and 16 -> 783 tiles; s/iter is per-GPU work, taken as is.
    Asserts each phase did what it claims; evals exactly {40, 80, 120}; best.json consistent."""
    sys.path.insert(0, "/root")
    from tierb import tierb_common as TC
    key = f"smoke:{m}"
    try:
        _reload()
        if TC.is_done(m + "_smoke", "smoke"):
            _status(key, state="DONE(already)"); advance(i); return "already"
        run = TC.run_dir(m + "_smoke")
        prog, tagp, evp = f"{run}/progress.json", f"{run}/train/last_checkpoint", f"{run}/evals.jsonl"
        cmd = _d2("train", m, ["--num-gpus", str(N_SMOKE), "--smoke", "--smoke-iters", "120",
                               "--smoke-period", "40"])
        tok = _d2_token(run)

        def stop_at(n):
            def w(proc, info):
                if info and info.get("iter", 0) >= n and not os.path.exists("/tmp/tierb_STOP"):
                    open("/tmp/tierb_STOP", "w").write("smoke")
            return w

        def kill_during_eval80(proc, info):
            tag = open(tagp).read().strip() if os.path.exists(tagp) else ""
            if tag == "model_0000079.pth" and 80 not in {r["iter"] for r in TC.read_jsonl(evp)}:
                os.killpg(proc.pid, signal.SIGKILL)
        if TC.stop_requested():
            raise SmokeHalted("STOP set")
        if not os.path.exists(f"{run}/smoke_A.json"):
            r = run_managed(key + ":A", cmd, cwd=D2_CWD, progress=prog, token=tok, watch=stop_at(50),
                            poll=2, stall_s=30 * 60, usd_h=RATE["smoke"])
            if r["status"] in ("stopped_before_start", "duplicate", "failed_no_progress"):
                raise SmokeHalted(r["status"])
            assert os.path.exists("/tmp/tierb_STOPPED"), f"A: STOP not honoured {r}"
            st = int(open("/tmp/tierb_STOPPED").read())
            assert st >= 49 and (st + 1) % 10 == 0, st
            assert open(tagp).read().strip() == f"model_{st:07d}.pth"
            TC.atomic_json(f"{run}/smoke_A.json", {"res": r, "stopped_at_iter": st})
        if not os.path.exists(f"{run}/smoke_B.json"):
            r = _phase_ok(run_managed(key + ":B", cmd, cwd=D2_CWD, progress=prog, token=tok,
                                      watch=kill_during_eval80, poll=2, stall_s=30 * 60, usd_h=RATE["smoke"]))
            assert r["rc"] == -9, f"B: not killed during the eval of 80 ({r}); rehearsal invalid"
            assert open(tagp).read().strip() == "model_0000079.pth"
            assert 80 not in {e["iter"] for e in TC.read_jsonl(evp)}
            TC.atomic_json(f"{run}/smoke_B.json", {"res": r})
        r = _phase_ok(run_managed(key + ":C", cmd, cwd=D2_CWD, progress=prog, token=tok, stall_s=30 * 60, usd_h=RATE["smoke"]))
        assert r["status"] == "ok", r
        ev = TC.read_jsonl(evp)
        assert [e["iter"] for e in ev] == [40, 80, 120], f"evals {[e['iter'] for e in ev]}"
        best = json.load(open(f"{run}/best.json"))
        assert best["val_box"]["AP50"] == max(e["val_box"]["AP50"] for e in ev)
        plat = json.load(open(f"{run}/plateau.json"))["state"]
        assert plat["decays"] == 2, f"plateau rule did not fire both decays: {plat}"   # smoke params
        lrs = {}
        for ln in open(f"{run}/train/metrics.json"):
            try:
                x = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "lr" in x:
                lrs[x["iteration"]] = x["lr"]
        a_it = max(k for k in lrs if 40 <= k < 80)
        b_it = max(k for k in lrs if 80 <= k < 120)
        assert abs(lrs[b_it] / lrs[a_it] - 0.1) < 1e-6, f"LR did not drop x0.1 after epoch 2: {lrs}"
        pcmd = _d2("predict", m, ["--num-gpus", str(N_SMOKE), "--smoke", "--splits", "val",
                                  "--limit-tiles", str(SMOKE_TILES)])
        t_p = time.time()
        wdir = f"{run}/predict/val/whole"

        def kill_predict(proc, info):
            n = len([x for x in os.listdir(wdir) if x.endswith(".json") and ".tmp" not in x]) \
                if os.path.isdir(wdir) else 0
            if n >= 4 and not os.path.exists(f"{run}/predict_killed"):
                open(f"{run}/predict_killed", "w").write(str(n))
                os.killpg(proc.pid, signal.SIGKILL)
        if not os.path.exists(f"{run}/smoke_P1.json"):
            if os.path.exists(f"{run}/predict_killed"):    # killed earlier; container lost before record
                TC.atomic_json(f"{run}/smoke_P1.json", {"res": "killed in an earlier attempt", "sec": None})
            else:
                rp = _phase_ok(run_managed(key + ":P1", pcmd, cwd=D2_CWD, watch=kill_predict, poll=2,
                                           stall_s=30 * 60, usd_h=RATE["smoke"]))
                assert rp["rc"] == -9 and os.path.exists(f"{run}/predict_killed"), f"P1 not killed: {rp}"
                TC.atomic_json(f"{run}/smoke_P1.json", {"res": rp, "sec": round(time.time() - t_p, 1)})
        t_p2 = time.time()
        rp = _phase_ok(run_managed(key + ":P2", pcmd, cwd=D2_CWD, stall_s=30 * 60, usd_h=RATE["smoke"]))
        assert rp["status"] == "ok", rp
        import statistics
        mets = []
        for ln in open(f"{run}/train/metrics.json"):
            try:
                x = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "time" in x and x.get("iteration", 0) > 85:
                mets.append(x["time"])
        spi = statistics.median(mets)
        ev_s = statistics.mean(e["sec"] for e in ev) * N_SMOKE / 8      # 108-tile eval, 8 GPUs
        p2 = time.time() - t_p2
        p1 = json.load(open(f"{run}/smoke_P1.json"))["sec"]
        p_sec = (p1 if p1 is not None else p2) + p2          # unknown P1 (lost record): assume = P2
        p_tiles = 108 + 439                            # val + test, x 2 geometries in both
        phrs = p_sec * N_SMOKE / 8 * p_tiles / SMOKE_TILES / 3600    # incl. 2 start-ups: high side
        by_ep = {E: round((E * 198 * spi + E * ev_s) / 3600 + phrs, 2) for E in (8, 12, 20)}
        hrs = by_ep[12] - phrs
        rep = {"smoke_gpus": N_SMOKE, "sec_per_iter": spi, "eval_sec_8gpu": ev_s,
               "plateau_smoke": plat, "lr_drop_checked": [a_it, b_it],
               "projected_hours_by_epochs_incl_predict": by_ep,
               "projected_usd_by_epochs": {E: round(h * USD_8GPU_D2, 1) for E, h in by_ep.items()},
               "projected_train_hours": round(hrs, 2),
               "projected_predict_hours": round(phrs, 2),
               "projected_usd": round((hrs + phrs) * USD_8GPU_D2, 1),
               "phases": {k: json.load(open(f"{run}/smoke_{k}.json")) for k in ("A", "B", "P1")},
               "evals": ev, "best": best}
        TC.report(m + "_smoke", "smoke", rep)
        TC.mark_done(m + "_smoke", "smoke", rep)
        vol.commit()
        _status(key, state="DONE", projected_train_hours=rep["projected_train_hours"],
                projected_usd=rep["projected_usd"])
        advance(i)
        return rep
    except SmokeHalted as e:
        _status(key, state="STOPPED", reason=str(e))
        return {"status": "STOPPED", "reason": str(e)}
    except AssertionError as e:                        # a rehearsal that did not happen as designed
        _failed(key, e, retrying=False)
        return {"status": "FAILED", "error": repr(e)}
    except Exception as e:
        _failed(key, e, retrying=True)
        raise


@app.function(image=img_d2, volumes=VOLS, gpu=GPU8, cpu=24, memory=131072, timeout=10 * 3600,
              retries=RETRY)
def d2_gpu8_train(i: int, m: str):
    sys.path.insert(0, "/root")
    from tierb import tierb_common as TC
    key = f"train:{m}"
    _reload()
    run = TC.run_dir(m)
    if not TC.is_done(m, "train"):
        res = run_managed(key, _d2("train", m, ["--num-gpus", "8"]), cwd=D2_CWD,
                          progress=f"{run}/progress.json", token=_d2_token(run), stop="graceful",
                          usd_h=RATE["gpu8"])
        if res["status"] != "ok" or not os.path.exists(TC.done_marker(m, "train")):
            return _finish(i, key, res, TC.done_marker(m, "train"))   # raises / records stop
        _status(key, state="DONE", log=res.get("log"))
    # predict in the SAME container (budget: no second 8-GPU start-up); plan step i+1 is the
    # standalone fallback and skips itself once predict is DONE
    pkey = f"predict:{m}"
    if TC.is_done(m, "predict"):
        _status(pkey, state="DONE(already)"); advance(i + 1); return "already"
    pres = run_managed(pkey, _d2("predict", m, ["--num-gpus", "8"]), cwd=D2_CWD,
                       token=_count_files(f"{run}/predict", (".done", "_tile.json")), usd_h=RATE["gpu8"])
    return _finish(i + 1, pkey, pres, TC.done_marker(m, "predict"))


def _count_files(d, suffixes):
    """Progress token = number of finished per-tile outputs (d2: whole `<tid>.json` + sub `<tid>.done`;
    MAL: `<tid>.soft.npz`). Temp files and per-subtile Prediction_* files are not counted."""
    suffixes = (suffixes,) if isinstance(suffixes, str) else suffixes

    def f():
        n = 0
        for root, _, fs in os.walk(d):
            for x in fs:
                if ".tmp" in x:
                    continue
                if x.endswith(".done") or x.endswith(".soft.npz") and ".soft.npz" in suffixes:
                    n += 1
                elif "_tile.json" in suffixes and x.endswith(".json") and not x.startswith("Prediction_") \
                        and os.path.basename(root) == "whole":
                    n += 1
        return n
    return f


@app.function(image=img_d2, volumes=VOLS, gpu=GPU8, cpu=24, memory=131072, timeout=8 * 3600,
              retries=RETRY)
def d2_gpu8_predict(i: int, m: str):
    sys.path.insert(0, "/root")
    from tierb import tierb_common as TC
    key = f"predict:{m}"
    _reload()
    if TC.is_done(m, "predict"):
        _status(key, state="DONE(already)"); advance(i); return "already"
    run = TC.run_dir(m)
    res = run_managed(key, _d2("predict", m, ["--num-gpus", "8"]), cwd=D2_CWD,
                      token=_count_files(f"{run}/predict", (".done", "_tile.json")), usd_h=RATE["gpu8"])
    return _finish(i, key, res, TC.done_marker(m, "predict"))


# ------------------------------------------------------------------ GPU steps: MAL (B3)
def _mal(cmd, extra=()):
    return [sys.executable, "/root/tierb/mal/mal_run.py", cmd, *extra]


def _mal_token(run):
    def f():
        d = os.path.join(run, "train", "ckpt")
        eps = sorted(x for x in os.listdir(d) if x.startswith("epoch=")) if os.path.isdir(d) else []
        b = os.path.join(run, "best.json")
        return (tuple(eps), os.path.getmtime(b) if os.path.exists(b) else None)
    return f


def _mal_resume(run):
    from tierb.mal.mal_run import last_good_ckpt
    c = last_good_ckpt(os.path.join(run, "train", "ckpt"))
    return ["--resume-ckpt", c] if c else []


@app.function(image=img_mal, volumes=VOLS, gpu=GPU1, cpu=8, memory=49152, timeout=2 * 3600,
              retries=RETRY1)
def mal_gpu1(i: int, m: str = "mal"):
    sys.path.insert(0, "/root")
    from tierb import tierb_common as TC
    key = "gate:mal"
    _reload()
    if TC.is_done("mal", "gate"):
        _status(key, state="DONE(already)"); advance(i); return "already"
    res = run_managed(key, _mal("gate"), cwd="/opt/mal", stall_s=40 * 60, usd_h=RATE["gate"])
    return _finish(i, key, res, TC.done_marker("mal", "gate"))


@app.function(image=img_mal, volumes=VOLS, gpu=GPU_SMOKE, cpu=12, memory=65536, timeout=3 * 3600,
              retries=RETRY1)
def mal_gpu2_smoke(i: int, m: str = "mal"):
    """A: 2 short epochs (30 batches each, progress every 2 batches); SIGKILL in epoch 1 once epoch
    0 is checkpointed and selected -- asserted. B: resume from epoch 0 -> finishes epoch 1; asserted
    no replay (global_step after epoch 1 == 60), finite losses after the resume, evals {0, 1} once.
    P: predict the val GT boxes of the first 16 val tiles for timing.
    Runs on 2 GPUs at MAL's per-GPU batch (4); projections scale val/predict by 2/8 GPUs and by
    boxes; s/step is per-GPU work, taken as is (980 steps/epoch at 8 GPUs)."""
    sys.path.insert(0, "/root")
    from tierb import tierb_common as TC
    key = "smoke:mal"
    try:
        _reload()
        if TC.is_done("mal_smoke", "smoke"):
            _status(key, state="DONE(already)"); advance(i); return "already"
        run = TC.run_dir("mal_smoke")
        prog = f"{run}/progress.json"
        base = _mal("train", ["--smoke", "--smoke-batches", "30", "--num-gpus", str(N_SMOKE)])

        def kill_in_epoch1(proc, info):
            if info and info.get("epoch", 0) >= 1 and info.get("batch", 0) >= 6 \
                    and os.path.exists(f"{run}/sel/epoch0.pt") \
                    and os.path.exists(f"{run}/train/ckpt/epoch=0.ckpt"):
                os.killpg(proc.pid, signal.SIGKILL)
        if TC.stop_requested():
            raise SmokeHalted("STOP set")
        if not os.path.exists(f"{run}/smoke_A.json"):
            r = _phase_ok(run_managed(key + ":A", base + _mal_resume(run), cwd="/opt/mal", progress=prog,
                                      token=_mal_token(run), watch=kill_in_epoch1, poll=2, stall_s=30 * 60, usd_h=RATE["smoke"]))
            assert r["rc"] == -9, f"A: not killed in epoch 1 ({r}); rehearsal invalid"
            TC.atomic_json(f"{run}/smoke_A.json", {"res": r})
        if not os.path.exists(f"{run}/smoke_B.json"):
            n_log0 = len(TC.read_jsonl(f"{run}/progress_log.jsonl"))
            tb = time.time()
            r = _phase_ok(run_managed(key + ":B", base + _mal_resume(run), cwd="/opt/mal", progress=prog,
                                      token=_mal_token(run), poll=5, stall_s=30 * 60, usd_h=RATE["smoke"]))
            assert r["status"] == "ok", r
            ev = TC.read_jsonl(f"{run}/evals.jsonl")
            assert [e["epoch"] for e in ev] == [0, 1], ev
            assert ev[1]["global_step"] == 60, f"epoch replayed? global_step {ev[1]['global_step']}"
            logB = TC.read_jsonl(f"{run}/progress_log.jsonl")[n_log0:]
            import math
            assert logB and all(math.isfinite(v) for x in logB for v in x["losses"].values()), "non-finite loss after resume"
            steps = [x for x in logB if x["epoch"] == 1]
            sps = (steps[-1]["t"] - steps[0]["t"]) / max(steps[-1]["global_step"] - steps[0]["global_step"], 1)
            TC.atomic_json(f"{run}/smoke_B.json", {"res": r, "sec_per_step": sps,
                                                   "phase_B_sec": round(time.time() - tb, 1)})
        B = json.load(open(f"{run}/smoke_B.json"))
        sps = B["sec_per_step"]
        ev = TC.read_jsonl(f"{run}/evals.jsonl")
        best = json.load(open(f"{run}/best.json"))
        plat = json.load(open(f"{run}/plateau.json"))["state"]
        assert plat["decays"] == 1, f"plateau rule did not fire on epoch 2: {plat}"     # smoke params
        tp = time.time()
        rp = _phase_ok(run_managed(key + ":P", _mal("predict", ["--smoke", "--sets", "val_gt",
                                                                "--num-gpus", str(N_SMOKE),
                                                                "--limit-tiles", str(SMOKE_TILES)]),
                                   cwd="/opt/mal", stall_s=30 * 60, usd_h=RATE["smoke"]))
        assert rp["status"] == "ok", rp
        p_sec = time.time() - tp
        val_s = statistics_mean([e["sec"] for e in ev]) * N_SMOKE / 8   # 108-tile val, 8 GPUs
        n_boxes_all = 207_370                         # the four MAL_SETS (tierb_data/DATA_MANIFEST.json)
        vb = json.load(open("/root/tierb_data/boxsets/val_gt.json"))["tiles"]
        n_smoke_boxes = sum(len(vb[t]["xywh"]) for t in sorted(vb)[:SMOKE_TILES])
        phrs = p_sec * N_SMOKE / 8 * n_boxes_all / max(n_smoke_boxes, 1) / 3600
        by_ep = {E: round((E * 980 * sps + E * val_s) / 3600 + phrs, 2) for E in (6, 10, 20)}
        hrs = by_ep[10] - phrs
        rep = {"smoke_gpus": N_SMOKE, "sec_per_step": sps, "val_sec_8gpu": val_s, "plateau_smoke": plat,
               "projected_hours_by_epochs_incl_predict": by_ep,
               "projected_usd_by_epochs": {E: round(h * USD_8GPU_MAL, 1) for E, h in by_ep.items()},
               "projected_train_hours": round(hrs, 2),
               "projected_predict_hours": round(phrs, 2),
               "projected_usd": round((hrs + phrs) * USD_8GPU_MAL, 1),
               "phase_B_sec": B["phase_B_sec"], "evals": ev, "best": best}
        TC.report("mal_smoke", "smoke", rep)
        TC.mark_done("mal_smoke", "smoke", rep)
        vol.commit()
        _status(key, state="DONE", projected_train_hours=rep["projected_train_hours"],
                projected_usd=rep["projected_usd"])
        advance(i)
        return rep
    except SmokeHalted as e:
        _status(key, state="STOPPED", reason=str(e))
        return {"status": "STOPPED", "reason": str(e)}
    except AssertionError as e:
        _failed(key, e, retrying=False)
        return {"status": "FAILED", "error": repr(e)}
    except Exception as e:
        _failed(key, e, retrying=True)
        raise


def statistics_mean(xs):
    return sum(xs) / max(len(xs), 1)


@app.function(image=img_mal, volumes=VOLS, gpu=GPU8, cpu=24, memory=131072, timeout=10 * 3600,
              retries=RETRY)
def mal_gpu8_train(i: int, m: str = "mal"):
    sys.path.insert(0, "/root")
    from tierb import tierb_common as TC
    key = "train:mal"
    _reload()
    run = TC.run_dir("mal")
    if not TC.is_done("mal", "train"):
        res = run_managed(key, _mal("train") + _mal_resume(run), cwd="/opt/mal",
                          progress=f"{run}/progress.json", token=_mal_token(run), stop="kill",
                          usd_h=RATE["gpu8"])
        if res["status"] != "ok" or not os.path.exists(TC.done_marker("mal", "train")):
            return _finish(i, key, res, TC.done_marker("mal", "train"))
        _status(key, state="DONE", log=res.get("log"))
    pkey = "predict:mal"                                    # same container (budget)
    if TC.is_done("mal", "predict"):
        _status(pkey, state="DONE(already)"); advance(i + 1); return "already"
    pres = run_managed(pkey, _mal("predict", ["--sets", MAL_SETS]), cwd="/opt/mal",
                       token=_count_files(f"{run}/predict", ".soft.npz"), usd_h=RATE["gpu8"])
    return _finish(i + 1, pkey, pres, TC.done_marker("mal", "predict"))


MAL_SETS = "val_gt,test_gt,sparse_gt,lace_s0"   # budget: LACE s1/s2, SelvaBox, LACE-sparse later if wanted


@app.function(image=img_mal, volumes=VOLS, gpu=GPU8, cpu=24, memory=131072, timeout=8 * 3600,
              retries=RETRY)
def mal_gpu8_predict(i: int, m: str = "mal"):
    sys.path.insert(0, "/root")
    from tierb import tierb_common as TC
    key = "predict:mal"
    _reload()
    if TC.is_done("mal", "predict"):
        _status(key, state="DONE(already)"); advance(i); return "already"
    run = TC.run_dir("mal")
    res = run_managed(key, _mal("predict", ["--sets", MAL_SETS]), cwd="/opt/mal",
                      token=_count_files(f"{run}/predict", ".soft.npz"), usd_h=RATE["gpu8"])
    return _finish(i, key, res, TC.done_marker("mal", "predict"))


# ------------------------------------------------------------------ CPU: finalize + LACE gate
@app.function(image=img_cpu, volumes=VOLS, cpu=32, memory=131072, timeout=10 * 3600, retries=RETRY)
def finalize(i: int, m: str):
    """Assemble/stitch/score with the FROZEN code (score_coco, stitch_seam/stitch_res, gtbox_lib),
    after re-scoring a stored LACE row to exact equality (scorer environment parity), then apply the
    pre-registered LACE gate. Writes /vol/runs/<m>/results/. Resumable per output file."""
    sys.path.insert(0, "/root")
    from tierb import finalize_lib as FL
    from tierb import tierb_common as TC
    key = f"finalize:{m}"
    t_start = time.time()
    try:
        _reload()
        gp = TC.run_dir(m, "results", "lace_gate.json")
        if TC.is_done(m, "finalize"):
            g = json.load(open(gp))
            if g["triggered"] and not TC.gate_overridden(m):   # re-latch: unstop alone cannot bypass
                TC.set_gate_triggered(m, g["why"])
                TC.set_stop(f"LACE gate (re-latched): {m}: {g['why']}")
            _status(key, state="DONE(already)", gate_triggered=g["triggered"])
            advance(i)
            return "already"
        com = TC.Committer(vol, period=120).start()
        res = FL.finalize_mal() if m == "mal" else FL.finalize_d2(m)
        gate = FL.lace_gate(m, res)
        TC.atomic_json(gp, gate, indent=1)
        if gate["triggered"]:
            TC.set_gate_triggered(m, gate["why"])
            TC.set_stop(f"LACE gate: {m} clearly beats LACE ({gate['why']})")
        TC.mark_done(m, "finalize", {"summary": res["summary"], "gate": gate})
        com.close(strict=True)
        record_spend(key, STARTUP_MIN + (time.time() - t_start) / 60, RATE["finalize"])
        _status(key, state="DONE", gate_triggered=gate["triggered"], why=gate.get("why"))
        advance(i)
        return {"summary": res["summary"], "gate": gate}
    except FL.ScorerParityError as e:                  # never score Tier B in a drifted environment
        TC.set_stop(f"scorer parity failed in finalize:{m}: {e}")
        _failed(key, e, retrying=False)
        return {"status": "FAILED", "error": repr(e)}
    except Exception as e:
        _failed(key, e, retrying=True)
        raise


@app.function(image=img_cpu, volumes=VOLS, cpu=1, memory=2048, timeout=600)
def report_status():
    sys.path.insert(0, "/root")
    from tierb import tierb_common as TC
    _reload()
    d = _ctl()
    led = d.get("spend", None) or []
    proj = {}
    for m in ("mal_smoke", "boxteacher_smoke"):
        p = TC.run_dir(m, "reports", "smoke.json")
        if os.path.exists(p):
            proj[m.replace("_smoke", "")] = json.load(open(p)).get("projected_usd")
    out = {"budget": {"budget_usd": BUDGET_USD, "hard_stop_usd": HARD_STOP_USD,
                      "spent_usd": spend_total(d), "ledger": led,
                      "projected_usd_from_smokes": proj,
                      "note": "B1 projection ~= B2's train x (BoxInst/BoxTeacher speed); no B1 smoke"},
           "STOP": d.get("STOP", None), "GATE_TRIGGERED": d.get("GATE_TRIGGERED", None),
           "GATE_OVERRIDE": d.get("GATE_OVERRIDE", None), "chain": d.get("chain", None),
           "status": d.get("status", None),
           "locks": {k: v for k, v in d.items() if str(k).startswith("lock:")},
           "attempts": {k: v for k, v in d.items() if str(k).startswith("attempts:")}, "progress": {}}
    for m in ("mal", "boxteacher", "boxinst", "mal_smoke", "boxteacher_smoke", "boxinst_smoke"):
        for f in ("progress.json", "best.json"):
            p = TC.run_dir(m, f)
            if os.path.exists(p):
                out["progress"].setdefault(m, {})[f] = json.load(open(p))
        for f in ("smoke.json", "gate.json"):
            p = TC.run_dir(m, "reports", f)
            if os.path.exists(p):
                r = json.load(open(p))
                out["progress"].setdefault(m, {})[f] = {k: r[k] for k in r
                                                        if k in ("pass", "projected_train_hours",
                                                                 "projected_predict_hours", "projected_usd",
                                                                 "sec_per_iter", "sec_per_step", "seconds")}
    return out
