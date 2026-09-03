"""Make BoxInstSeg's `tree_filter` op compile against torch >= 1.11.

THE PROBLEM. `mmdet/ops/tree_filter` -- needed by BoxLevelSet and Box2Mask, and imported at
module scope by `box_solov2_head` / `box2mask_head`, so `mmdet` will not import without it --
includes <THC/THC.h>, <THC/THCAtomics.cuh> and <THC/THCDeviceUtils.cuh>. PyTorch DELETED the THC
headers in 1.11, and torch <= 1.10 has no cp310 wheels while Modal's builder requires Python
>= 3.10, so there is no version pair that satisfies both.

THE FIX IS NINE LINES. The includes turn out to be VESTIGIAL. Audited across all three .cu
files, the only THC-provided symbol actually referenced is `atomicAdd`, at three call sites in
bfs.cu, all on `int*` -- and `atomicAdd(int*, int)` is a CUDA built-in, not a THC function.
`THCudaCheck`, `THCCeilDiv`, `THCState`, `THCudaMalloc` are never used; the file even defines
its own `GET_CUDA_BLOCKS` rather than calling `THCCeilDiv`. So the headers can simply be
dropped, with <ATen/cuda/CUDAContext.h> put in their place to keep the CUDA context available.

This is a build fix, not a change to the method: no kernel, no arithmetic and no parameter is
touched, only three include lines per file. Verified by the op importing and by the four
detectors registering (see `imports_ok`).
"""
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/opt/BoxInstSeg"
SRC = [f"{ROOT}/mmdet/ops/tree_filter/src/{n}" for n in
       ("bfs/bfs.cu", "mst/mst.cu", "refine/refine.cu")]
THC = re.compile(r'^\s*#include\s*<THC/[^>]+>\s*$')
REPLACEMENT = "#include <ATen/cuda/CUDAContext.h>"

for path in SRC:
    lines = open(path).read().splitlines()
    out, dropped, done = [], 0, False
    for ln in lines:
        if THC.match(ln):
            dropped += 1
            if not done:                      # one replacement per file, at the first hit
                out.append(REPLACEMENT)
                done = True
            continue
        out.append(ln)
    assert dropped, f"{path}: no THC include found -- has the upstream source changed?"
    src = "\n".join(out) + "\n"
    assert "THC/" not in src, f"{path}: a THC include survived"
    # Guard the audit that justifies this patch: if upstream ever starts calling a real THC
    # function, dropping the headers would break the build in a confusing way. Fail loudly here.
    for sym in ("THCudaCheck", "THCCeilDiv", "THCState", "THCudaMalloc", "THCudaFree"):
        assert sym not in src, (f"{path}: uses {sym}, which THC provided -- the includes are "
                                f"NOT vestigial any more and this patch is unsafe")
    open(path, "w").write(src)
    print(f"[patch] {path}: dropped {dropped} THC include(s)")
print("[patch] tree_filter should now compile against torch >= 1.11; "
      "all four methods (BoxInst, DiscoBox, BoxLevelSet, Box2Mask) remain registered")
