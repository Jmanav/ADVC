"""
scripts/gen_tiny_imagenet_map.py

Generate the Tiny-ImageNet synset → ImageNet-1k index mapping and print it as a
Python dict literal to paste into utils/datasets.py (`_TINY_IMAGENET_TO_IMAGENET`).

Why
---
DeiT-S keeps its pretrained 1000-way ImageNet-1k head. Tiny-ImageNet's 200
classes are all ImageNet-1k synsets, so each Tiny-ImageNet folder ("nXXXXXXXX")
must map to its position in the canonical ILSVRC2012 ordering the head was
trained with. This script derives that mapping reproducibly.

Source of the canonical ordering
---------------------------------
timm packages the ImageNet-1k synset list in ILSVRC2012 order via
`timm.data.imagenet_info.ImageNetInfo`. We use it as the authority so the
mapping matches exactly the indices the timm-loaded DeiT-S head produces.
(timm is already a project dependency — see requirements.)

Correctness safeguard
----------------------
Tiny-ImageNet is a strict subset of ImageNet-1k, so ALL 200 wnids MUST resolve.
A missing wnid means the wrong ordering/source was used — the script asserts on
this rather than silently emitting a bad map (which would only show up later as
collapsed accuracy).

Usage
-----
    python scripts/gen_tiny_imagenet_map.py \
        --wnids /kaggle/working/tiny-imagenet-200/wnids.txt

Then copy the printed dict into utils/datasets.py.
"""

import argparse
from pathlib import Path


def build_imagenet_synset_to_idx() -> dict[str, int]:
    """Return {synset: idx} in canonical ILSVRC2012 order via timm.

    Raises:
        RuntimeError: if timm's ImageNet synset list cannot be obtained.
    """
    try:
        from timm.data.imagenet_info import ImageNetInfo
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Could not import timm.data.imagenet_info.ImageNetInfo. "
            "Ensure a recent timm is installed."
        ) from e

    info = ImageNetInfo()  # defaults to imagenet-1k
    # index_to_label_name across 0..999 gives the synset (WordNet ID) per index.
    synset_to_idx: dict[str, int] = {}
    for idx in range(info.num_classes()):
        synset = info.index_to_label_name(idx)
        synset_to_idx[synset] = idx

    if len(synset_to_idx) != 1000:
        raise RuntimeError(
            f"Expected 1000 ImageNet synsets, got {len(synset_to_idx)}. "
            "timm ImageNet info looks wrong."
        )
    return synset_to_idx


def read_wnids(wnids_path: Path) -> list[str]:
    """Read Tiny-ImageNet's wnids.txt (200 WordNet IDs, one per line)."""
    with open(wnids_path, "r") as f:
        wnids = [line.strip() for line in f if line.strip()]
    if len(wnids) != 200:
        raise ValueError(
            f"Expected 200 wnids in {wnids_path}, got {len(wnids)}."
        )
    return wnids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wnids",
        type=str,
        default="/kaggle/working/tiny-imagenet-200/wnids.txt",
        help="Path to Tiny-ImageNet wnids.txt",
    )
    args = parser.parse_args()

    synset_to_idx = build_imagenet_synset_to_idx()
    wnids = read_wnids(Path(args.wnids))

    mapping: dict[str, int] = {}
    missing: list[str] = []
    for wnid in wnids:
        if wnid in synset_to_idx:
            mapping[wnid] = synset_to_idx[wnid]
        else:
            missing.append(wnid)

    # Primary correctness safeguard.
    assert not missing, (
        f"{len(missing)} Tiny-ImageNet wnids did not resolve to an ImageNet-1k "
        f"index (wrong ordering/source?). First few: {missing[:5]}"
    )
    assert len(mapping) == 200, f"Expected 200 entries, got {len(mapping)}."
    assert all(0 <= v < 1000 for v in mapping.values()), "Index out of [0,1000)."

    # Spot-checks a human can eyeball (known ImageNet indices).
    spot = {"n01443537": "goldfish", "n01944390": "snail", "n02509815": "lesser panda"}
    print("# Spot-checks (synset -> idx):")
    for s, name in spot.items():
        if s in mapping:
            print(f"#   {s} ({name}) -> {mapping[s]}")

    print(f"# Resolved {len(mapping)}/200 wnids, all indices in [0, 1000).")
    print("# Paste the following into utils/datasets.py as _TINY_IMAGENET_TO_IMAGENET:\n")
    print("_TINY_IMAGENET_TO_IMAGENET: dict[str, int] = {")
    for wnid in sorted(mapping, key=lambda w: mapping[w]):
        print(f'    "{wnid}": {mapping[wnid]},')
    print("}")


if __name__ == "__main__":
    main()
