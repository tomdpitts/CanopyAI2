"""Flat-folder image dataset for DINOv3 SSL (the repo ships no ImageFolder).

Copied into the cloned repo as dinov3/data/datasets/arid_pool.py and registered by
apply_repo_patches.py. Implements the minimal ExtendedVisionDataset interface:
get_image_data (raw bytes), get_target (dummy; discarded for SSL), __len__.
"""
import os

from dinov3.data.datasets.extended import ExtendedVisionDataset

_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


class AridPool(ExtendedVisionDataset):
    def __init__(self, *, root, extra=None, split=None, transforms=None,
                 transform=None, target_transform=None):
        # NB: ExtendedVisionDataset.__init__ is (image_decoder, target_decoder,
        # *args, **kwargs) — positional args mis-bind root to image_decoder.
        # Pass by keyword and let the decoders default.
        super().__init__(root=root, transforms=transforms, transform=transform,
                         target_transform=target_transform)
        self.samples = sorted(
            os.path.join(dp, f)
            for dp, _, fs in os.walk(root)
            for f in fs if os.path.splitext(f)[1].lower() in _EXTS)
        if not self.samples:
            raise RuntimeError(f"AridPool found no images under {root!r}")

    def get_image_data(self, index: int) -> bytes:
        with open(self.samples[index], "rb") as fh:
            return fh.read()

    def get_target(self, index: int):
        return 0                      # SSL discards targets (target_transform -> ())

    def __len__(self) -> int:
        return len(self.samples)
