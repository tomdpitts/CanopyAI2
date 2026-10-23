"""Run MAL's own main.py unmodified, with only its hard-coded dataset paths pointed at ours.
Used by mal_gate.py G5 as the reference the tile-batched inference must equal bitwise.

    python mal_cli_launch.py '{"coco": ..., "img_dir": ...}' <main.py args...>
"""
import json
import runpy
import sys

sys.path.insert(0, "/opt/mal")
sys.path.insert(0, "/root")

from tierb.mal.mal_run import patch_datapaths  # noqa: E402

cfg = json.loads(sys.argv[1])
sys.argv = ["main.py"] + sys.argv[2:]
patch_datapaths(train_json=cfg["coco"], val_json=cfg["coco"], val_img_dir=cfg["img_dir"])
try:
    runpy.run_path("/opt/mal/main.py", run_name="__main__")
except AttributeError as e:            # released: ends with args.val_ann_path after writing labels
    print(f"[mal_cli_launch] expected trailing AttributeError: {e}", flush=True)
