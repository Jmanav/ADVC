"""
scripts/build_tensor_cache.py

Build the pre-decoded uint8 tensor cache (REVISION_PLAN.md C-01) for one
dataset, using the repo's own dataset code (load_config + build_remapped_folder
+ build_eval_transform) so ordering, RGB conversion and label remapping are
identical to the live pipeline. Single process (the H100 slice has 1 CPU core).

Outputs (in --out):
    <dataset>_val_eval_u8.pt   (+ .json)   bit-exact eval cache
    <dataset>_train_train_u8.pt (+ .json)  full-resolution decoded train cache

Usage:
    python scripts/build_tensor_cache.py --dataset imagenette --split val \
        --out /path/to/cache --verify 200
    python scripts/build_tensor_cache.py --dataset imagenette --split both \
        --out /path/to/cache --verify 200 [--val-dir D] [--train-dir D]
    python scripts/build_tensor_cache.py ... --limit 50 --verify 20   # smoke run

--verify K (eval): re-loads K random indices through BOTH the original
ImageFolder + build_eval_transform path and the cache and asserts torch.equal
on the normalized tensors and equal labels/paths. (train): asserts the cached
full-resolution uint8 image equals the freshly decoded RGB image, labels/paths
match, and an augmented sample has shape [3,S,S] and dtype float32.
"""

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models.loader import load_config  # noqa: E402
from utils.datasets import build_eval_transform, build_remapped_folder  # noqa: E402
from utils.tensor_cache import (  # noqa: E402
    build_eval_cache,
    build_train_cache,
    cache_filename,
    load_eval_cache,
    load_train_cache,
    save_cache,
)


def _set_seeds(seed: int) -> None:
    """Seed Python, NumPy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def config_for_dataset(config_path: str, name: str) -> dict:
    """Return the config with dataset `name` active.

    load_config() flattens whichever dataset `configs/base.yaml` names as
    active. If that is a different dataset, re-apply load_config's exact merge
    rule to a fresh copy of the raw YAML with `dataset.name` set to `name`
    (no file is modified). Self-checked against load_config for the active one.
    """
    cfg = load_config(config_path)
    if cfg["dataset"]["name"] == name:
        return cfg
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    ds = raw.get("dataset", {})
    if not isinstance(ds.get(name), dict):
        raise SystemExit(f"Dataset {name!r} has no sub-block under `dataset:` in {config_path}.")
    # Self-check: our merge reproduces load_config for the file's active dataset.
    check = copy.deepcopy(raw)
    act = check["dataset"].get("name")
    if act and isinstance(check["dataset"].get(act), dict):
        for k, v in check["dataset"][act].items():
            check["dataset"][k] = v
    assert check == cfg, "merge rule drifted from models.loader.load_config"
    ds["name"] = name
    for k, v in ds[name].items():  # same rule as load_config
        ds[k] = v
    return raw


def _verify_eval(cache_path: str, root: str, cfg: dict, k: int, seed: int) -> None:
    """Assert bit-exactness of K random cached samples vs the original pipeline."""
    ref = build_remapped_folder(root, build_eval_transform(cfg), cfg)
    cached = load_eval_cache(cache_path, cfg)
    n = len(cached)
    assert n <= len(ref)
    idx = random.Random(seed).sample(range(n), min(k, n))
    if not idx:
        raise AssertionError("eval cache is empty — nothing to verify")
    for i in idx:
        x_ref, y_ref = ref[i]
        x_c, y_c = cached[i]
        assert os.path.normpath(cached.samples[i][0]) == os.path.normpath(ref.samples[i][0]), f"path mismatch at {i}"
        assert y_c == y_ref == cached.targets[i] == ref.targets[i], f"label mismatch at {i}"
        assert x_c.dtype == x_ref.dtype and x_c.shape == x_ref.shape, f"dtype/shape mismatch at {i}"
        if not torch.equal(x_c, x_ref):
            d = (x_c - x_ref).abs().max().item()
            raise AssertionError(f"NOT bit-exact at index {i} ({ref.samples[i][0]}): max|diff|={d}")
    print(f"[verify] eval: {len(idx)}/{len(idx)} random samples bit-exact (torch.equal) "
          f"vs ImageFolder+build_eval_transform. dtype={x_ref.dtype}, shape={tuple(x_ref.shape)}")


def _verify_train(cache_path: str, root: str, cfg: dict, k: int, seed: int) -> None:
    """Check cached decoded images equal fresh decodes; check augmented output."""
    ref = build_remapped_folder(root, None, cfg)
    cached = load_train_cache(cache_path, cfg)
    n = len(cached)
    idx = random.Random(seed).sample(range(n), min(k, n))
    s = int(cfg["dataset"]["image_size"])
    for i in idx:
        path, y_ref = ref.samples[i]
        assert os.path.normpath(cached.samples[i][0]) == os.path.normpath(path), f"path mismatch at {i}"
        assert cached.targets[i] == y_ref, f"label mismatch at {i}"
        fresh = torch.from_numpy(np.array(ref.loader(path), dtype=np.uint8, copy=True)).permute(2, 0, 1)
        assert torch.equal(cached.decoded(i), fresh), f"decoded image mismatch at {i} ({path})"
        x, _ = cached[i]
        assert x.shape == (3, s, s) and x.dtype == torch.get_default_dtype(), f"aug output {x.shape} {x.dtype}"
    print(f"[verify] train: {len(idx)} samples — decoded uint8 identical to fresh decode; "
          f"augmented output [3,{s},{s}] {torch.get_default_dtype()} (augmentation itself is NOT "
          f"pixel-identical to the PIL path, by design).")


def _report(obj: dict, path: str, elapsed: float) -> None:
    """Print count, payload bytes, file bytes, sha256 and elapsed time."""
    m = obj["meta"]
    fb = os.path.getsize(path)
    print(f"[cache] wrote {path}")
    print(f"        count={m['count']}  payload_bytes={m['nbytes']:,} ({m['nbytes'] / 2**20:.1f} MiB)  "
          f"file_bytes={fb:,}  sha256={m['sha256']}  elapsed={elapsed:.1f}s")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset sub-block name, e.g. imagenette")
    ap.add_argument("--split", choices=["val", "train", "both"], required=True)
    ap.add_argument("--out", required=True, help="output directory for cache files")
    ap.add_argument("--config", default=str(_ROOT / "configs/base.yaml"))
    ap.add_argument("--val-dir", default=None, help="override dataset.val_dir")
    ap.add_argument("--train-dir", default=None, help="override dataset.train_dir")
    ap.add_argument("--limit", type=int, default=None, help="cache only the first N samples (smoke test)")
    ap.add_argument("--verify", type=int, default=0, metavar="K", help="verify K random samples after build")
    ap.add_argument("--max-ram-gb", type=float, default=20.0,
                    help="abort train build before decoding if buffer exceeds this (GiB)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    _set_seeds(args.seed)
    torch.set_num_threads(1)  # 1-core slice; also avoids oversubscription
    cfg = config_for_dataset(args.config, args.dataset)
    if args.val_dir:
        cfg["dataset"]["val_dir"] = args.val_dir
    if args.train_dir:
        cfg["dataset"]["train_dir"] = args.train_dir
    name = cfg["dataset"]["name"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[cache] dataset={name} resize={cfg['dataset'].get('resize', 256)} "
          f"image_size={cfg['dataset']['image_size']} label_mode={cfg['dataset'].get('label_mode', 'imagenet')}")

    if args.split in ("val", "both"):
        root = cfg["dataset"]["val_dir"]
        path = str(out / cache_filename(name, "val", "eval"))
        t0 = time.time()
        obj = build_eval_cache(root, cfg, "val", limit=args.limit, progress=not args.no_progress)
        save_cache(obj, path)
        _report(obj, path, time.time() - t0)
        del obj
        if args.verify:
            _verify_eval(path, root, cfg, args.verify, args.seed)

    if args.split in ("train", "both"):
        root = cfg["dataset"]["train_dir"]
        path = str(out / cache_filename(name, "train", "train"))
        t0 = time.time()
        obj = build_train_cache(root, cfg, "train", limit=args.limit, progress=not args.no_progress,
                                max_bytes=int(args.max_ram_gb * 2**30))
        save_cache(obj, path)
        _report(obj, path, time.time() - t0)
        del obj
        if args.verify:
            _verify_train(path, root, cfg, args.verify, args.seed)


if __name__ == "__main__":
    main()
