"""Local control for Tier B (B1 BoxInst, B2 BoxTeacher, B3 MAL). Nothing here runs a model.

    python tierb_ctl.py plan                  # the step list with indices
    python tierb_ctl.py deploy                # modal deploy tierb_modal.py (re-run after any edit)
    python tierb_ctl.py start                 # chain from step 0 up to HOLD (data, weights, gates, smokes)
    python tierb_ctl.py continue [index]      # release HOLD: train/predict/finalize B3 -> B2 -> B1
    python tierb_ctl.py status                # chain, per-step state, locks, gate latch, live progress
    python tierb_ctl.py stop "reason"         # STOP: trainers checkpoint and exit; everything else killed
    python tierb_ctl.py unstop                # clear STOP (does not restart anything)
    python tierb_ctl.py spawn <index>         # (re)spawn one plan step; safe: steps skip if DONE, resume otherwise
    python tierb_ctl.py unlock [key]          # drop heartbeat lock(s) after a hard `modal app stop` (else wait 3 min)
    python tierb_ctl.py reset-attempts [key]  # clear a FAILED step's attempt history before a deliberate retry
    python tierb_ctl.py override-gate <m> "why"  # the ONLY way past a fired LACE gate, per method; logged
    python tierb_ctl.py pull [--with-preds]   # copy reports/results/evals/logs into tierb/results/<timestamp>/

Every step is idempotent and resumable, so `spawn` / `start` / `continue` can be repeated after any
dropout, cancellation or STOP without losing or duplicating work. Hard cancel: `modal app stop
tcd04-tierb`, then `deploy`, `unlock`, and `spawn <index>` of the interrupted step.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
MODAL = os.path.join(REPO, ".venv", "bin", "modal")
APP = "tcd04-tierb"
VOL = "tcd04-tierb-vol"
RUNS = ("_shared", "mal", "boxteacher", "boxinst", "mal_smoke", "boxteacher_smoke", "boxinst_smoke")


def _plan():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tierb_modal", os.path.join(HERE, "tierb_modal.py"))
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    return tm.PLAN, tm.HOLD_INDEX, tm.FN_NAME


def _ctl():
    import modal
    return modal.Dict.from_name("tcd04-tierb-control", create_if_missing=True)


def _guard(d):
    if d.get("STOP", None):
        sys.exit(f"STOP is set ({d.get('STOP')}); `unstop` first if that is intended")
    trig = d.get("GATE_TRIGGERED", None) or {}
    ov = d.get("GATE_OVERRIDE", None) or {}
    open_ = {m: v for m, v in trig.items() if m not in ov}
    if open_:
        sys.exit(f"LACE gate fired ({open_}); spend stays paused. "
                 f"`override-gate <method> \"why\"` only on Tom's decision")


def _spawn(i):
    import modal
    plan, _, fn_name = _plan()
    step, m = plan[i]
    fn = fn_name(step, m)
    f = modal.Function.from_name(APP, fn)
    call = f.spawn(i) if m is None else f.spawn(i, m)
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "index": i, "step": step, "method": m,
           "fn": fn, "call_id": call.object_id}
    with open(os.path.join(HERE, "calls.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    return rec


def _vget(remote, local_dir):
    """`modal volume get` into an EXISTING directory (it then writes paths relative to the remote
    parent). Returns True if fetched; a missing remote path is reported, other errors raise."""
    os.makedirs(local_dir, exist_ok=True)
    p = subprocess.run([MODAL, "volume", "get", "--force", VOL, remote, local_dir],
                       capture_output=True, text=True)
    if p.returncode != 0:
        msg = (p.stderr or p.stdout).strip()
        if "not found" in msg.lower() or "no such" in msg.lower():
            print(f"  (absent) {remote}")
            return False
        raise RuntimeError(f"modal volume get {remote} failed: {msg}")
    return True


def _vls(remote):
    p = subprocess.run([MODAL, "volume", "ls", "--json", VOL, remote], capture_output=True, text=True)
    if p.returncode != 0:
        return []
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return []


def pull(with_preds):
    dest = os.path.join(HERE, "results", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dest)
    for m in RUNS:
        md = os.path.join(dest, m)
        for sub in ("reports", "DONE", "evals.jsonl", "best.json", "progress.json",
                    "train/metrics.json", "train/log.txt", "train/config.yaml"):
            _vget(f"runs/{m}/{sub}", md if "/" not in sub else os.path.join(md, os.path.dirname(sub)))
        for e in _vls(f"runs/{m}/results"):
            name = os.path.basename(e.get("Filename") or e.get("filename") or e.get("path") or "")
            if not name:
                continue
            if name.startswith("preds_") and not with_preds:
                continue
            _vget(f"runs/{m}/results/{name}", os.path.join(md, "results"))
    _vget("runs/logs", dest)
    _vget("rgb/_stage_record.json", dest)
    _vget("weights/WEIGHTS.json", dest)
    print(f"-> {dest}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "plan":
        plan, hold, _ = _plan()
        for i, (s, m) in enumerate(plan):
            print(f"{i:2d}  {s:14s} {m or ''}{'   <- review point' if i == hold else ''}")
    elif cmd == "deploy":
        subprocess.run([MODAL, "deploy", os.path.join(HERE, "tierb_modal.py")], check=True)
    elif cmd == "start":
        plan, hold, _ = _plan()
        d = _ctl()
        _guard(d)
        d.put("chain", {"active": True, "until": hold, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
        _spawn(0)
    elif cmd == "continue":
        plan, hold, _ = _plan()
        d = _ctl()
        _guard(d)
        d.put("chain", {"active": True, "until": len(plan) - 1,
                        "released": time.strftime("%Y-%m-%d %H:%M:%S")})
        _spawn(int(rest[0]) if rest else hold + 1)
    elif cmd == "spawn":
        _guard(_ctl())
        _spawn(int(rest[0]))
    elif cmd == "stop":
        _ctl().put("STOP", {"reason": " ".join(rest) or "manual", "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        print("STOP set")
    elif cmd == "unstop":
        _ctl().pop("STOP", None)
        print("STOP cleared (nothing restarted; a fired LACE gate still blocks: see override-gate)")
    elif cmd == "unlock":
        d = _ctl()
        keys = [k for k, _ in d.items() if str(k).startswith("lock:") and (not rest or k == f"lock:{rest[0]}")]
        for k in keys:
            d.pop(k, None)
        print(f"removed {keys}")
    elif cmd == "reset-attempts":
        d = _ctl()
        keys = [k for k, _ in d.items() if str(k).startswith("attempts:") and (not rest or k == f"attempts:{rest[0]}")]
        for k in keys:
            d.pop(k, None)
        print(f"removed {keys}")
    elif cmd == "override-gate":
        assert len(rest) >= 2 and rest[0] in ("mal", "boxteacher", "boxinst"), \
            'usage: override-gate <mal|boxteacher|boxinst> "reason (Tom\'s decision)"'
        d = _ctl()
        ov = d.get("GATE_OVERRIDE", None) or {}
        ov[rest[0]] = {"reason": " ".join(rest[1:]), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        d.put("GATE_OVERRIDE", ov)
        print(f"GATE_OVERRIDE set for {rest[0]} (other methods' gates still bind)")
    elif cmd == "status":
        import modal
        r = modal.Function.from_name(APP, "report_status").remote()
        print(json.dumps(r, indent=1, default=str))
    elif cmd == "pull":
        pull("--with-preds" in rest)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
