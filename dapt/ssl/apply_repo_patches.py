"""Patch a fresh dinov3 clone for DAPT: register the AridPool flat-folder dataset and
add an artifact-free random k*90 rotation to the SSL augmentations (valid for overhead
imagery — 90-degree multiples keep square crops with NO black-corner fill, unlike
arbitrary RandomRotation). Idempotent.

Usage:  python apply_repo_patches.py /path/to/dinov3
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def patch(path, anchor, insert, tag):
    with open(path) as f:
        src = f.read()
    if tag in src:
        print(f"  already patched: {os.path.basename(path)} ({tag})")
        return
    assert anchor in src, f"anchor not found in {path}:\n{anchor!r}"
    with open(path, "w") as f:
        f.write(src.replace(anchor, insert, 1))
    print(f"  patched {os.path.basename(path)} ({tag})")


def main(repo):
    ds = os.path.join(repo, "dinov3/data/datasets")
    # 1. drop in the dataset class
    shutil.copy(os.path.join(HERE, "arid_pool_dataset.py"),
                os.path.join(ds, "arid_pool.py"))
    print("  copied arid_pool.py")

    # 2. export it
    patch(os.path.join(ds, "__init__.py"),
          "from .nyu import NYU",
          "from .nyu import NYU\nfrom .arid_pool import AridPool  # DAPT",
          "arid_pool import AridPool")

    # 3. register in the dataset-string dispatch
    loaders = os.path.join(repo, "dinov3/data/loaders.py")
    patch(loaders,
          '    else:\n        raise ValueError(f\'Unsupported dataset "{name}"\')',
          '    elif name == "AridPool":\n        from dinov3.data.datasets import AridPool\n'
          '        class_ = AridPool\n'
          '    else:\n        raise ValueError(f\'Unsupported dataset "{name}"\')',
          'name == "AridPool"')

    # 4. artifact-free random k*90 rotation, added to global + local geometric aug
    aug = os.path.join(repo, "dinov3/data/augmentations.py")
    with open(aug) as f:
        src = f.read()
    if "class RandomRot90" not in src:
        # class def after the torchvision import. NOTE: must use v2.functional.rotate
        # (dispatches PIL *and* tensor); raw torch.rot90 raises NotImplementedError on
        # the PIL images in the geometric stage. Crops are square -> 90-deg rotation
        # is shape-safe without expand.
        src = src.replace(
            "from torchvision.transforms import v2",
            "from torchvision.transforms import v2\nimport torch\n\n"
            "class RandomRot90(v2.Transform):\n"
            "    # torchvision >=0.21 dispatches to transform(); older to _transform()\n"
            "    def transform(self, inpt, params):\n"
            "        k = int(torch.randint(0, 4, ()))\n"
            "        return v2.functional.rotate(inpt, 90.0 * k) if k else inpt\n"
            "    _transform = transform\n",
            1)
        # insert after each horizontal flip (global + local)
        src = src.replace(
            "v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),",
            "v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),\n"
            "                RandomRot90(),")
        with open(aug, "w") as f:
            f.write(src)
        print("  patched augmentations.py (RandomRot90 x2)")
    else:
        print("  already patched: augmentations.py (RandomRot90)")

    # 5. backbone-only teacher checkpoint: the released web weights have no DINO/iBOT
    # heads and no rope_embed.periods buffer, but init_fsdp_model_from_checkpoint does
    # a STRICT load over the whole student ModuleDict -> crash. init_weights() has
    # already random-inited the heads and computed the rope buffer BEFORE this load,
    # so a non-strict load that keeps those values is exactly the continued-SSL init
    # we want. Assert the missing set is ONLY heads + rope periods (anything else
    # missing, or any unexpected key, is a real arch mismatch and must still fail).
    ckpt = os.path.join(repo, "dinov3/checkpointer/checkpointer.py")
    patch(ckpt,
          "        model.load_state_dict(\n"
          "            {\n"
          "                key: tensor\n"
          "                for key, tensor in chkpt.items()\n"
          "                if not any(skip_load_key in key for skip_load_key in skip_load_keys)\n"
          "            }\n"
          "        )",
          "        incompat = model.load_state_dict(\n"
          "            {\n"
          "                key: tensor\n"
          "                for key, tensor in chkpt.items()\n"
          "                if not any(skip_load_key in key for skip_load_key in skip_load_keys)\n"
          "            },\n"
          "            strict=False,  # DAPT: backbone-only teacher ckpt\n"
          "        )\n"
          "        bad_missing = [k for k in incompat.missing_keys\n"
          "                       if not (k.startswith(('dino_head.', 'ibot_head.'))\n"
          "                               or k.endswith('rope_embed.periods'))]\n"
          "        assert not bad_missing and not incompat.unexpected_keys, (\n"
          "            f'DAPT non-strict load: bad_missing={bad_missing[:5]} '\n"
          "            f'unexpected={incompat.unexpected_keys[:5]}')\n"
          "        logger.info(f'DAPT non-strict load OK: kept init_weights() values for '\n"
          "                    f'{len(incompat.missing_keys)} keys (dino/ibot heads + rope periods)')",
          "DAPT: backbone-only teacher ckpt")

    print("repo patches applied.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.environ.get("DINOV3_REPO", "./dinov3"))
