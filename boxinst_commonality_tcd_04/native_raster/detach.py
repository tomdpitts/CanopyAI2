"""Detached Modal launches — run a job server-side with no client attached.

WHY. `modal run` creates an EPHEMERAL app bound to the local client process: if the laptop
sleeps, the network drops, or the terminal closes, the client dies and Modal stops the app.
For the CPU sweeps that only costs the in-flight cell (everything else is checkpointed to the
volume). For the GPU rows — Restor, DetecTree2, SelvaBox/SAM 3 — it would throw away a real
A100 pass, so those must be spawned instead.

`modal deploy` publishes the app persistently; `.spawn()` then starts a call that runs to
completion server-side whether or not anything is listening. The call id is recorded in
`calls.json` so a result can be collected later from any machine, in any session — which is
the same discipline as persisting masks: keep the handle to the expensive thing.

Usage:
    python detach.py deploy val_knobs_2048_modal.py
    python detach.py spawn tcd04-native-raster-valknobs sweep --res 2048
    python detach.py status                     # every recorded call
    python detach.py status fc-XXXX             # one call, and collect its result
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CALLS = os.path.join(HERE, "calls.json")


def _load():
    return json.load(open(CALLS)) if os.path.exists(CALLS) else []


def _save(rows):
    with open(CALLS, "w") as f:
        json.dump(rows, f, indent=2)


def deploy(path: str):
    """Publish the app so its functions can be spawned. Idempotent; re-run after edits."""
    p = path if os.path.isabs(path) else os.path.join(HERE, path)
    print(f"[detach] modal deploy {p}", flush=True)
    subprocess.run([sys.executable, "-m", "modal", "deploy", p], check=True)


def spawn(app_name: str, fn_name: str, kwargs: dict, note: str = ""):
    """Start a detached call. Returns immediately with a durable call id."""
    import modal
    fn = modal.Function.from_name(app_name, fn_name)
    call = fn.spawn(**kwargs)
    row = {"call_id": call.object_id, "app": app_name, "fn": fn_name,
           "kwargs": kwargs, "note": note}
    rows = _load()
    rows.append(row)
    _save(rows)
    print(json.dumps(row, indent=2))
    print(f"\n[detach] running server-side. Safe to close this terminal.\n"
          f"[detach] collect with:  python detach.py status {call.object_id}")
    return row


def status(call_id: str | None):
    """Poll one call (or all recorded ones). A finished call returns its result here."""
    import modal
    rows = _load()
    if call_id:
        rows = [r for r in rows if r["call_id"] == call_id] or [{"call_id": call_id}]
    if not rows:
        print("[detach] no recorded calls")
        return
    for r in rows:
        fc = modal.FunctionCall.from_id(r["call_id"])
        try:
            res = fc.get(timeout=0)
            state = "DONE"
        except TimeoutError:
            res, state = None, "running"
        except Exception as e:                      # failed / cancelled / expired
            res, state = f"{type(e).__name__}: {e}", "ERROR"
        print(f"{r['call_id']}  {state:8} {r.get('fn','')} {r.get('kwargs','')} "
              f"{r.get('note','')}")
        if state == "DONE":
            out = json.dumps(res, indent=2)
            print(out[:2000] + ("\n... (truncated)" if len(out) > 2000 else ""))
        elif state == "ERROR":
            print(f"  {res}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("deploy"); d.add_argument("path")
    s = sub.add_parser("spawn")
    s.add_argument("app"); s.add_argument("fn")
    s.add_argument("--note", default="")
    s.add_argument("kv", nargs="*", help="--key value pairs passed to the function")
    # NOTE: function kwargs arrive as unknown args (e.g. --res 2048); parse_known_args
    # below folds them into kv rather than rejecting them.
    t = sub.add_parser("status"); t.add_argument("call_id", nargs="?")

    a, extra = ap.parse_known_args()
    if getattr(a, "kv", None) is not None:
        a.kv = list(a.kv) + extra
    if a.cmd == "deploy":
        deploy(a.path)
    elif a.cmd == "spawn":
        kw, i = {}, 0
        while i < len(a.kv):
            k = a.kv[i].lstrip("-").replace("-", "_")
            v = a.kv[i + 1]
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            kw[k] = v
            i += 2
        spawn(a.app, a.fn, kw, a.note)
    else:
        status(a.call_id)


if __name__ == "__main__":
    main()
