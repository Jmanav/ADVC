"""
tests/test_tensor_cache.py

CPU-only tests for utils/tensor_cache.py on a tiny synthetic ImageFolder:
  - eval cache is bit-exact (torch.equal) vs ImageFolder + build_eval_transform,
    including grayscale / RGBA / palette sources (exercises convert('RGB'));
  - cfg mismatch is rejected on load; uint8 (normalize_on_gpu) mode works;
  - train cache stores the exact decoded image and its augmentation has the
    right shape/dtype, is reproducible under a seed and varies across seeds.

Run:  pytest -q tests/test_tensor_cache.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from utils.datasets import build_eval_transform, build_remapped_folder  # noqa: E402
from utils.tensor_cache import (  # noqa: E402
    build_eval_cache,
    build_train_cache,
    cache_filename,
    load_eval_cache,
    load_train_cache,
    normalize_batch,
    save_cache,
)

# (class, filename, (W, H), PIL mode) — odd sizes, mixed formats/modes.
_FAKE = [
    ("cls_a", "a0.jpg", (53, 41), "RGB"),
    ("cls_a", "a1.png", (40, 67), "L"),
    ("cls_a", "a2.png", (45, 45), "RGBA"),
    ("cls_b", "b0.jpg", (61, 39), "RGB"),
    ("cls_b", "b1.png", (37, 50), "P"),
    ("cls_b", "b2.jpeg", (70, 44), "L"),
]


def _make_folder(root: Path) -> Path:
    """Write the fake ImageFolder with deterministic random pixels."""
    rng = np.random.default_rng(0)
    for cls, fname, (w, h), mode in _FAKE:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        if mode == "RGB":
            img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")
        elif mode == "RGBA":
            img = Image.fromarray(rng.integers(0, 256, (h, w, 4), dtype=np.uint8), "RGBA")
        else:
            img = Image.fromarray(rng.integers(0, 256, (h, w), dtype=np.uint8), "L")
            if mode == "P":
                img = img.convert("P")
        img.save(d / fname)
    return root


def _cfg(root: Path) -> dict:
    """Minimal config: label_mode raw -> remap_subset_labels is a no-op."""
    return {"dataset": {
        "name": "fake",
        "image_size": 32,
        "resize": 36,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "label_mode": "raw",
        "val_dir": str(root),
        "train_dir": str(root),
    }}


@pytest.fixture()
def folder(tmp_path: Path) -> Path:
    """Fake ImageFolder root."""
    return _make_folder(tmp_path / "data")


def test_eval_cache_bit_exact(folder: Path, tmp_path: Path) -> None:
    """Every cached sample equals the original pipeline exactly."""
    cfg = _cfg(folder)
    obj = build_eval_cache(str(folder), cfg, "val", progress=False)
    path = tmp_path / "cache" / cache_filename("fake", "val", "eval")
    save_cache(obj, str(path))
    assert Path(str(path) + ".json").is_file()

    ref = build_remapped_folder(str(folder), build_eval_transform(cfg), cfg)
    ds = load_eval_cache(str(path), cfg)
    assert len(ds) == len(ref) == len(_FAKE)
    assert ds.targets == ref.targets
    assert ds.classes == ref.classes
    assert [Path(p) for p, _ in ds.samples] == [Path(p) for p, _ in ref.samples]
    for i in range(len(ref)):
        x_ref, y_ref = ref[i]
        x, y = ds[i]
        assert y == y_ref
        assert x.dtype == x_ref.dtype and x.shape == x_ref.shape == (3, 32, 32)
        assert torch.equal(x, x_ref), f"index {i} not bit-exact"


def test_eval_cache_uint8_mode_and_cpu_normalize(folder: Path, tmp_path: Path) -> None:
    """normalize_on_gpu returns uint8; normalize_batch on CPU matches the pipeline."""
    cfg = _cfg(folder)
    path = tmp_path / cache_filename("fake", "val", "eval")
    save_cache(build_eval_cache(str(folder), cfg, "val", progress=False), str(path))
    ds8 = load_eval_cache(str(path), cfg, normalize_on_gpu=True)
    u8, _ = ds8[0]
    assert u8.dtype == torch.uint8 and u8.shape == (3, 32, 32)
    batch = torch.stack([ds8[i][0] for i in range(len(ds8))])
    xb = normalize_batch(batch, cfg["dataset"]["mean"], cfg["dataset"]["std"])
    ref = build_remapped_folder(str(folder), build_eval_transform(cfg), cfg)
    expect = torch.stack([ref[i][0] for i in range(len(ref))])
    # Same op sequence on CPU; allow 1-ulp-scale tolerance since bit-identity
    # of the batched helper is not part of the contract.
    assert torch.allclose(xb, expect, rtol=0, atol=1e-6)


def test_eval_cache_rejects_cfg_mismatch(folder: Path, tmp_path: Path) -> None:
    """Changed preprocessing in cfg must raise a clear error."""
    cfg = _cfg(folder)
    path = tmp_path / cache_filename("fake", "val", "eval")
    save_cache(build_eval_cache(str(folder), cfg, "val", progress=False), str(path))
    for key, val in [("resize", 40), ("image_size", 28), ("name", "other"),
                     ("mean", [0.5, 0.5, 0.5]), ("label_mode", "imagenet")]:
        bad = _cfg(folder)
        bad["dataset"][key] = val
        with pytest.raises(ValueError, match="does not match"):
            load_eval_cache(str(path), bad)


def test_eval_cache_limit(folder: Path) -> None:
    """--limit caches a prefix in ImageFolder order."""
    cfg = _cfg(folder)
    obj = build_eval_cache(str(folder), cfg, "val", limit=2, progress=False)
    assert obj["images"].shape == (2, 3, 32, 32) and obj["meta"]["limit"] == 2


def test_train_cache(folder: Path, tmp_path: Path) -> None:
    """Decoded images are exact; augmentation has right shape/dtype and seeds."""
    cfg = _cfg(folder)
    path = tmp_path / cache_filename("fake", "train", "train")
    save_cache(build_train_cache(str(folder), cfg, "train", progress=False), str(path))
    ds = load_train_cache(str(path), cfg)
    ref = build_remapped_folder(str(folder), None, cfg)
    assert ds.targets == ref.targets

    for i, (p, _) in enumerate(ref.samples):
        fresh = torch.from_numpy(np.array(ref.loader(p), dtype=np.uint8)).permute(2, 0, 1)
        assert torch.equal(ds.decoded(i), fresh)
        x, _ = ds[i]
        assert x.shape == (3, 32, 32) and x.dtype == torch.get_default_dtype()

    def draw(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.stack([ds[i][0] for i in range(len(ds))])

    assert torch.equal(draw(0), draw(0)), "augmentation not reproducible under a fixed seed"
    assert not torch.equal(draw(0), draw(1)), "augmentation did not vary with the seed"

    ds8 = load_train_cache(str(path), cfg, normalize_on_gpu=True)
    u8, _ = ds8[0]
    assert u8.dtype == torch.uint8 and u8.shape == (3, 32, 32)
