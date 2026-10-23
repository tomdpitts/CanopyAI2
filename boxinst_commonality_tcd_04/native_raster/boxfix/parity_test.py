"""Gate: 'base' variant in a test sweep == gtbox_guided_raw/gtbox_lace_guided_r2048.jsonl per-crown IoUs (published LACE arm)."""
import json, os, sys
import numpy as np
NR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sw = json.load(open(sys.argv[1])); ref = {}
for l in open(os.path.join(NR, "gtbox_guided_raw", "gtbox_lace_guided_r2048.jsonl")):
    r = json.loads(l); ref[r["tile"]] = r["iou"]
w = max(float(np.abs(np.asarray(t["iou"]["base"]) - np.asarray(ref[t["tile"]])).max()) for t in sw["tiles"] if len(t["valid"]))
print(f"test parity: {len(sw['tiles'])} tiles max|diff| {w:.1e}")
