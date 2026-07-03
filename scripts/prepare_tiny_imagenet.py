"""
scripts/prepare_tiny_imagenet.py

Reshape the standard tiny-imagenet-200 zip layout into ImageFolder-ready
directories so the existing loaders (which expect ImageNette-style per-class
folders) work with no code changes.

Input layout (as downloaded — Stanford CS231n zip)
--------------------------------------------------
    tiny-imagenet-200/
      wnids.txt
      train/nXXXX/images/*.JPEG        <- nested images/ subfolder
      val/images/*.JPEG               <- FLAT, no class subfolders
      val/val_annotations.txt         <- filename \t wnid \t bbox...

Output layout (created by this script)
--------------------------------------
    tiny-imagenet-200/
      train_if/nXXXX/*.JPEG           <- flattened
      val_if/nXXXX/*.JPEG            <- regrouped by wnid

configs/base.yaml already points tiny-imagenet val_dir/train_dir at val_if/
train_if. This script is idempotent — re-running skips already-populated dirs.

Symlinks are used when possible (avoids duplicating ~110k files); on platforms
where symlinks are unavailable (common on Kaggle/Windows) it falls back to copy.

Usage
-----
    python scripts/prepare_tiny_imagenet.py \
        --root /kaggle/working/tiny-imagenet-200
"""

import argparse
import os
import shutil
from pathlib import Path


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink src→dst, falling back to copy if symlinks aren't permitted."""
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def prepare_train(root: Path) -> None:
    """Flatten train/nXXXX/images/*.JPEG → train_if/nXXXX/*.JPEG."""
    src_train = root / "train"
    dst_train = root / "train_if"
    if not src_train.is_dir():
        raise FileNotFoundError(f"Missing {src_train}")

    wnids = [d.name for d in src_train.iterdir() if d.is_dir()]
    print(f"[prep] train: {len(wnids)} classes")
    for wnid in wnids:
        src_imgs = src_train / wnid / "images"
        # Some mirrors already flatten; handle both.
        src_imgs = src_imgs if src_imgs.is_dir() else src_train / wnid
        dst_cls = dst_train / wnid
        if dst_cls.is_dir() and any(dst_cls.iterdir()):
            continue  # idempotent
        dst_cls.mkdir(parents=True, exist_ok=True)
        for img in src_imgs.glob("*.JPEG"):
            _link_or_copy(img.resolve(), dst_cls / img.name)
    print(f"[prep] train_if ready at {dst_train}")


def prepare_val(root: Path) -> None:
    """Regroup flat val/images/*.JPEG into val_if/nXXXX/*.JPEG via annotations."""
    src_val_imgs = root / "val" / "images"
    ann_path = root / "val" / "val_annotations.txt"
    dst_val = root / "val_if"
    if not src_val_imgs.is_dir():
        raise FileNotFoundError(f"Missing {src_val_imgs}")
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing {ann_path}")

    # filename -> wnid
    file_to_wnid: dict[str, str] = {}
    with open(ann_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                file_to_wnid[parts[0]] = parts[1]
    print(f"[prep] val: {len(file_to_wnid)} annotated images")

    for fname, wnid in file_to_wnid.items():
        dst_cls = dst_val / wnid
        dst_cls.mkdir(parents=True, exist_ok=True)
        dst_img = dst_cls / fname
        if dst_img.exists():
            continue  # idempotent
        src_img = (src_val_imgs / fname).resolve()
        if src_img.exists():
            _link_or_copy(src_img, dst_img)
    print(f"[prep] val_if ready at {dst_val}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default="/kaggle/working/tiny-imagenet-200",
        help="Path to the extracted tiny-imagenet-200 directory.",
    )
    args = parser.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(f"Tiny-ImageNet root not found: {root}")

    prepare_train(root)
    prepare_val(root)
    print("[prep] Done. Point configs at train_if/ and val_if/ (already set).")


if __name__ == "__main__":
    main()
