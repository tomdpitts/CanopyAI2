"""Container-side helpers shared by every Tier B image (stdlib + numpy + PIL only at import).

Three jobs:
  1. DATA IDENTITY. Images are read from the stage written by `tierb_modal.py::stage` and
     checked against `splits_manifest.json` + the stage record before any GPU work, so "the
     same images as LACE" is an assertion in every container, not a belief.
  2. SURVIVING DROPOUTS. Modal preempts containers at random. Every long stage (a) holds a
     heartbeat lock so a duplicate spawn exits instead of racing it, (b) writes its outputs
     atomically (tmp + os.replace) and commits the volume on a timer and at every checkpoint,
     and (c) resumes from what is already on the volume. Re-running any stage is always safe.
  3. THE SPEND SWITCH. `stop_requested()` reads a Modal Dict flag that every training and
     inference loop polls; `tierb_ctl.py stop` (or the LACE gate) sets it. A stopped stage
     saves, commits and returns normally (no retry is triggered).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

import numpy as np

VOL_NAME = "tcd04-tierb-vol"
CTL_NAME = "tcd04-tierb-control"          # modal.Dict: STOP flag, locks, heartbeats
VOL = "/vol"
RGB_DIR = f"{VOL}/rgb"                    # <tid>.png, written by `stage`
STAGE_REC = f"{VOL}/rgb/_stage_record.json"
MANIFEST = "/root/tierb/splits_manifest.json"
DATA = "/root/tierb_data"                 # prep_local.py outputs, shipped in the images
RUNS = f"{VOL}/runs"
LOCK_STALE_S = 180                        # heartbeat every <=15 s; 3 min of silence = dead holder


# ----------------------------------------------------------------- atomic IO
def atomic_json(path, obj, indent=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush(); os.fsync(f.fileno())


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    for ln in open(path):
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:      # a torn last line from a killed container
                pass
    return out


# ----------------------------------------------------------------- volume commits
class Committer:
    """Commit the volume every `period` s from a daemon thread, and on demand. Commits are
    what make writes survive a preemption; Modal's own background commit is not relied on."""

    def __init__(self, vol, period=180):
        self.vol, self.period = vol, period
        self._stop = threading.Event()
        self._ev = threading.Event()
        self._lock = threading.Lock()
        self.n = 0
        self.t = threading.Thread(target=self._run, daemon=True)

    def request(self):
        """Ask the background thread to commit now; never blocks the caller (the poll loop must
        stay free to act on watch callbacks and STOP)."""
        self._ev.set()

    def start(self):
        self.t.start(); return self

    def commit(self, strict=False):
        """Volume.commit() persists via the commit RPC and THEN reloads; the reload fails while
        files are open on the volume (trainers keep log/metrics handles open). That error therefore
        means 'committed, view not reloaded', which is all a writer needs. strict=True (used after the
        child has exited and nothing is open) retries and raises, so a step never advances on an
        uncommitted volume."""
        with self._lock:
            for k in range(3 if strict else 1):
                try:
                    self.vol.commit(); self.n += 1
                    return True
                except Exception as e:
                    if "open files" in str(e):
                        self.n += 1
                        return True
                    print(f"[commit] WARNING {type(e).__name__}: {e}", flush=True)
                    if strict:
                        time.sleep(10 * (k + 1))
            if strict:
                raise RuntimeError("volume commit failed 3 times")
            return False

    def _run(self):
        while not self._stop.is_set():
            self._ev.wait(self.period)
            self._ev.clear()
            if self._stop.is_set():
                break
            self.commit()

    def close(self, strict=True):
        self._stop.set()
        self._ev.set()
        self.t.join(timeout=600)
        return self.commit(strict=strict)


# ----------------------------------------------------------------- control: STOP + locks
def ctl():
    import modal
    return modal.Dict.from_name(CTL_NAME, create_if_missing=True)


def stop_requested(d=None):
    try:
        v = (d or ctl()).get("STOP", None)
    except Exception as e:                        # control plane hiccup: keep going, log it
        print(f"[ctl] WARNING cannot read STOP ({type(e).__name__}); assuming not set", flush=True)
        return None
    return v


def set_stop(reason, d=None):
    (d or ctl()).put("STOP", {"reason": reason, "time": time.strftime("%Y-%m-%d %H:%M:%S")})


def set_gate_triggered(method, why, d=None):
    """Persistent latch, separate from STOP: `unstop` alone cannot release a fired LACE gate;
    only an explicit GATE_OVERRIDE (tierb_ctl.py override-gate "reason") can."""
    d = d or ctl()
    cur = d.get("GATE_TRIGGERED", None) or {}
    cur[method] = {"why": why, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    d.put("GATE_TRIGGERED", cur)


def gate_overridden(method, d=None):
    return method in ((d or ctl()).get("GATE_OVERRIDE", None) or {})


def gate_blocks(d=None):
    """None if the chain may proceed, else the reason it may not. Overrides are PER METHOD: a
    released B3 trigger never releases a later B2 or B1 trigger."""
    d = d or ctl()
    trig = d.get("GATE_TRIGGERED", None) or {}
    ov = d.get("GATE_OVERRIDE", None) or {}
    open_ = {m: v for m, v in trig.items() if m not in ov}
    return f"LACE gate fired: {open_}" if open_ else None


def _call_id():
    try:
        import modal
        return modal.current_function_call_id() or f"local-{os.getpid()}"
    except Exception:
        return f"local-{os.getpid()}"


class StageLock:
    """One live call per (method, stage). A second spawn of the same stage exits early instead
    of racing the first on the same checkpoint directory. Re-entrant for the SAME call id, which
    is what Modal re-uses when it restarts a preempted input. A lock whose heartbeat is older
    than LOCK_STALE_S belongs to a dead container and is taken over."""

    def __init__(self, key, d=None):
        self.key, self.d, self.me = f"lock:{key}", d or ctl(), _call_id()

    def acquire(self):
        now = time.time()
        cur = self.d.get(self.key, None)
        if cur and cur["call"] != self.me and now - cur["beat"] < LOCK_STALE_S:
            return False, cur
        self.d.put(self.key, {"call": self.me, "beat": now, "since": cur["since"]
                              if cur and cur["call"] == self.me else now})
        time.sleep(2)                              # last-writer-wins check
        cur = self.d.get(self.key, None)
        return (cur is not None and cur["call"] == self.me), cur

    def beat(self, info=None):
        try:
            cur = self.d.get(self.key, None) or {}
            self.d.put(self.key, dict(cur, call=self.me, beat=time.time(), info=info))
        except Exception as e:
            print(f"[lock] WARNING heartbeat failed ({type(e).__name__})", flush=True)

    def release(self):
        try:
            cur = self.d.get(self.key, None)
            if cur and cur["call"] == self.me:
                self.d.pop(self.key)
        except Exception:
            pass


# ----------------------------------------------------------------- data identity
def manifest():
    return json.load(open(MANIFEST))


def stage_record():
    return json.load(open(STAGE_REC))


def rgb_path(tid):
    return os.path.join(RGB_DIR, f"{tid}.png")


def load_rgb(tid, verify=None):
    """(2048,2048,3) uint8. `verify` = expected rgb sha256 (from the stage record)."""
    from PIL import Image
    a = np.asarray(Image.open(rgb_path(tid)).convert("RGB"))
    if verify is not None:
        h = hashlib.sha256(a.tobytes()).hexdigest()
        assert h == verify, f"{tid}: pixel hash {h[:12]} != staged {verify[:12]}"
    assert a.shape == (2048, 2048, 3), (tid, a.shape)
    return a


def verify_split(split, max_workers=16):
    """Re-hash every staged image of `split` against the stage record and the manifest.
    Returns a record for the run report. Fails loudly on any mismatch."""
    from concurrent.futures import ThreadPoolExecutor
    man, rec = manifest(), stage_record()
    tids = man["ids"][split]
    want = {t: rec["tiles"][t]["rgb_sha256"] for t in tids}

    def one(t):
        load_rgb(t, verify=want[t])
        return t
    t0 = time.time()
    with ThreadPoolExecutor(max_workers) as ex:
        n = sum(1 for _ in ex.map(one, tids))
    assert n == len(tids) == man["counts"][split]
    return {"split": split, "n": n, "verified_against": "stage record rgb_sha256",
            "stage_vs_local": rec["summary"], "seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------- canopy (LACE's rule)
def canopy_px(polys, res=2048):
    """`train_detector_tiles.canopy_px`, verbatim: the canopy raster LACE's detector ignores."""
    from PIL import Image, ImageDraw
    m = Image.new("L", (res, res), 0)
    d = ImageDraw.Draw(m)
    for poly in polys:
        if poly and len(poly) >= 6:
            d.polygon([tuple(v) for v in np.asarray(poly).reshape(-1, 2)], fill=1)
    return np.asarray(m, bool)


# ----------------------------------------------------------------- run bookkeeping
def run_dir(method, *parts):
    return os.path.join(RUNS, method, *parts)


def done_marker(method, stage):
    return run_dir(method, "DONE", f"{stage}.json")


def is_done(method, stage):
    return os.path.exists(done_marker(method, stage))


def mark_done(method, stage, payload):
    atomic_json(done_marker(method, stage), dict(payload, stage=stage, method=method,
                                                 time=time.strftime("%Y-%m-%d %H:%M:%S")),
                indent=1)


def report(method, name, payload):
    atomic_json(run_dir(method, "reports", f"{name}.json"), payload, indent=1)
