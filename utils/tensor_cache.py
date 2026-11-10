"""
utils/tensor_cache.py

Pre-decoded uint8 tensor cache for ImageFolder datasets (REVISION_PLAN.md C-01).

Why this exists
---------------
The Imagenette runs on the H100 MIG slice have ONE CPU core. JPEG decoding +
PIL resizing on that core is the input bottleneck (Imagenette training ran at
~23 samples/s). This module decodes every image once, stores it as uint8, and
serves tensors from RAM afterwards. TinyImageNet runs on Kaggle (4 cores) and
does not need it.

Two cache kinds
---------------
EVAL cache  (`<dataset>_<split>_eval_u8.pt`)  — BIT-EXACT with the current path
    ImageFolder (+ utils.datasets label remap) + build_eval_transform =
    Resize(resize) -> CenterCrop(image_size) -> ToTensor -> Normalize.
    We run the *same* Resize and CenterCrop objects (taken from
    build_eval_transform) on the *same* PIL image that ImageFolder's loader
    produces (torchvision default_loader: open + .convert('RGB')), then store
    `torch.from_numpy(np.array(img, np.uint8)).permute(2, 0, 1)` — exactly what
    ToTensor does before its division. At load time ToTensor's
    `.to(default_float_dtype).div(255)` and torchvision's own
    `transforms.functional.normalize` are re-applied on CPU, so the normalized
    tensor is identical (torch.equal) to the original pipeline.
    `scripts/build_tensor_cache.py --verify K` checks this on the real data.

TRAIN cache (`<dataset>_<split>_train_u8.pt`) — NOT bit-identical (by design)
    Training uses RandomResizedCrop + RandomHorizontalFlip on PIL images. A
    fixed-crop cache would silently remove the augmentation, so the train cache
    stores the DECODED FULL-RESOLUTION RGB images (variable sizes, one flat
    uint8 buffer + per-image offsets/shapes). At load time torchvision
    RandomResizedCrop(image_size, antialias=True) + RandomHorizontalFlip run on
    the uint8 CHW tensor, then /255 and Normalize.
    IMPORTANT (state this in the paper's methods): tensor-based bilinear
    antialiased resizing is NOT numerically identical to PIL's resize, and the
    RNG draw sequence of the tensor transforms is not guaranteed to match the
    PIL path. The augmentation *distribution* (scale/ratio/flip probability)
    is the same; the pixels are not. Do not claim equivalence.

File format (torch.save dict, loaded with weights_only=True when available)
---------------------------------------------------------------------------
eval:  {"format": "advc-tensor-cache", "version": 1, "kind": "eval",
        "images": uint8 [N, 3, S, S] (CHW, RGB), "labels": int64 [N],
        "meta": {...}}
train: {"format": "advc-tensor-cache", "version": 1, "kind": "train",
        "buffer": uint8 [sum(3*H_i*W_i)], "offsets": int64 [N],
        "shapes": int64 [N, 3] (C, H, W), "labels": int64 [N], "meta": {...}}
meta: dataset name, split, root (as resolved at build time), rel_paths
      (POSIX, relative to root, ImageFolder order), classes, label_mode,
      image_size, resize, mean, std, transform description, count, nbytes,
      sha256, limit, torch/torchvision versions.
A JSON sidecar `<file>.json` holds the same meta minus rel_paths.

sha256 definition: SHA-256 over, in order, the raw C-contiguous bytes of
  eval:  images (uint8), labels (little-endian int64)
  train: buffer (uint8), shapes, offsets, labels (little-endian int64 each).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

CACHE_FORMAT = "advc-tensor-cache"
CACHE_VERSION = 1
_HASH_CHUNK = 64 * 1024 * 1024  # bytes per sha256 update


# ── Naming ────────────────────────────────────────────────────────────────────

def cache_filename(dataset: str, split: str, kind: str) -> str:
    """Return the canonical cache filename, e.g. 'imagenette_val_eval_u8.pt'.

    Args:
        dataset: Dataset name (cfg["dataset"]["name"]).
        split: "val" or "train".
        kind: "eval" or "train".
    """
    if kind not in ("eval", "train"):
        raise ValueError(f"kind must be 'eval' or 'train', got {kind!r}")
    return f"{dataset}_{split}_{kind}_u8.pt"


# ── Hashing ───────────────────────────────────────────────────────────────────

def _update_hash(h: "hashlib._Hash", t: torch.Tensor) -> None:
    """Feed a CPU tensor's raw C-contiguous bytes into `h` in chunks."""
    flat = t.contiguous().reshape(-1)
    if flat.dtype != torch.uint8:
        # Explicit little-endian int64 bytes for integer metadata tensors.
        h.update(np.ascontiguousarray(flat.numpy().astype("<i8", copy=False)).tobytes())
        return
    n = flat.numel()
    for a in range(0, n, _HASH_CHUNK):
        h.update(flat[a:a + _HASH_CHUNK].numpy())


def compute_sha256(tensors: list[torch.Tensor]) -> str:
    """SHA-256 over the concatenated raw bytes of `tensors` (see module doc)."""
    h = hashlib.sha256()
    for t in tensors:
        _update_hash(h, t)
    return h.hexdigest()


def _payload_tensors(obj: dict) -> list[torch.Tensor]:
    """Tensors covered by the sha256, in hashing order, for a cache dict."""
    if obj["kind"] == "eval":
        return [obj["images"], obj["labels"]]
    return [obj["buffer"], obj["shapes"], obj["offsets"], obj["labels"]]


# ── Config helpers ────────────────────────────────────────────────────────────

def _cfg_signature(cfg: dict) -> dict[str, Any]:
    """Preprocessing-relevant config fields that the cache depends on.

    Defaults mirror utils/datasets.py (resize -> 256) and remap_subset_labels
    (label_mode -> 'imagenet').
    """
    ds = cfg["dataset"]
    return {
        "dataset": str(ds["name"]),
        "image_size": int(ds["image_size"]),
        "resize": int(ds.get("resize", 256)),
        "mean": [float(x) for x in ds["mean"]],
        "std": [float(x) for x in ds["std"]],
        "label_mode": str(ds.get("label_mode", "imagenet")),
    }


def _validate_meta(meta: dict, cfg: dict, path: str) -> None:
    """Raise ValueError if the cache was built with a different preprocessing."""
    want = _cfg_signature(cfg)
    bad = [
        f"  {k}: cache={meta.get(k)!r} cfg={v!r}"
        for k, v in want.items()
        if meta.get(k) != v
    ]
    if bad:
        raise ValueError(
            f"Tensor cache {path} does not match the active config "
            f"(rebuild it with scripts/build_tensor_cache.py):\n" + "\n".join(bad)
        )


def _version_info() -> dict[str, str]:
    """Library versions recorded in the cache metadata."""
    # str(): torch.__version__ is a TorchVersion subclass, which the
    # weights_only unpickler may refuse.
    return {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__)}


# ── Save / load ───────────────────────────────────────────────────────────────

def save_cache(obj: dict, out_path: str) -> None:
    """torch.save the cache dict and write a JSON sidecar (meta minus rel_paths)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(out))  # atomic: never leave a half-written cache
    side = {k: v for k, v in obj["meta"].items() if k != "rel_paths"}
    side["file"] = out.name
    side["file_bytes"] = out.stat().st_size
    with open(str(out) + ".json", "w") as f:
        json.dump(side, f, indent=2)


def _torch_load(path: str, mmap: bool) -> dict:
    """torch.load with weights_only/mmap when the installed torch supports them."""
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": True}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        # Older torch without weights_only / mmap arguments.
        return torch.load(path, map_location="cpu")


def _load_checked(path: str, cfg: dict, kind: str, verify_hash: bool, mmap: bool) -> dict:
    """Load a cache file and check format, kind, config signature and sha256."""
    obj = _torch_load(str(path), mmap)
    if not isinstance(obj, dict) or obj.get("format") != CACHE_FORMAT:
        raise ValueError(f"{path} is not an ADVC tensor cache.")
    if obj.get("version") != CACHE_VERSION:
        raise ValueError(f"{path}: cache version {obj.get('version')} != {CACHE_VERSION}; rebuild.")
    if obj.get("kind") != kind:
        raise ValueError(f"{path}: expected a {kind!r} cache, got {obj.get('kind')!r}.")
    meta = obj["meta"]
    _validate_meta(meta, cfg, str(path))
    if meta.get("limit") is not None:
        print(f"[tensor_cache] WARNING: {path} was built with --limit {meta['limit']} "
              f"({meta['count']} samples) — NOT the full split; do not use for paper numbers.")
    if verify_hash:
        got = compute_sha256(_payload_tensors(obj))
        if got != meta["sha256"]:
            raise ValueError(f"{path}: sha256 mismatch (file {got} != meta {meta['sha256']}). Corrupt cache.")
    return obj


def _samples_from_meta(meta: dict, labels: torch.Tensor) -> list[tuple[str, int]]:
    """Rebuild ImageFolder-style (path, label) samples from root + rel paths."""
    root = meta["root"]
    return [
        (os.path.join(root, *rel.split("/")), int(lbl))
        for rel, lbl in zip(meta["rel_paths"], labels.tolist())
    ]


# ── Datasets ──────────────────────────────────────────────────────────────────

class _CacheDatasetBase(Dataset):
    """Shared ImageFolder-mirroring attributes: targets, samples, classes."""

    def __init__(self, obj: dict) -> None:
        self.meta: dict = obj["meta"]
        self.labels: torch.Tensor = obj["labels"]
        self.targets: list[int] = [int(x) for x in self.labels.tolist()]
        self.samples: list[tuple[str, int]] = _samples_from_meta(self.meta, self.labels)
        self.imgs = self.samples  # ImageFolder alias
        self.classes: list[str] = list(self.meta["classes"])
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.root: str = self.meta["root"]
        self._mean: list[float] = list(self.meta["mean"])
        self._std: list[float] = list(self.meta["std"])

    def __len__(self) -> int:
        """Number of cached samples."""
        return len(self.targets)

    def _to_normalized(self, u8: torch.Tensor) -> torch.Tensor:
        """Reproduce ToTensor's /255 and Normalize exactly (CPU)."""
        x = u8.to(dtype=torch.get_default_dtype()).div(255)  # == ToTensor on uint8
        return TF.normalize(x, mean=self._mean, std=self._std)  # == T.Normalize


class EvalTensorCache(_CacheDatasetBase):
    """Eval cache dataset: index -> (normalized float CHW tensor, label).

    With `normalize_on_gpu=True` it returns (uint8 CHW tensor, label) instead;
    the default DataLoader collate stacks these into a uint8 [B,3,S,S] batch
    that must then be passed through `normalize_batch(batch, mean, std)` on the
    GPU (4x less host->device traffic, no per-sample CPU float work).
    """

    def __init__(self, obj: dict, normalize_on_gpu: bool = False) -> None:
        super().__init__(obj)
        self.images: torch.Tensor = obj["images"]
        self.normalize_on_gpu = normalize_on_gpu

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return (image, label) for index `idx`."""
        u8 = self.images[idx]
        if self.normalize_on_gpu:
            return u8, self.targets[idx]
        return self._to_normalized(u8), self.targets[idx]


def _build_train_aug(image_size: int, scale: list[float], ratio: list[float]) -> Callable:
    """RandomResizedCrop(antialias=True) + RandomHorizontalFlip for uint8 tensors.

    Prefers torchvision.transforms.v2; falls back to v1 tensor transforms on
    older torchvision. Bilinear interpolation, as in the PIL training path.
    """
    try:
        from torchvision.transforms import v2 as T2
        return T2.Compose([
            T2.RandomResizedCrop(image_size, scale=tuple(scale), ratio=tuple(ratio),
                                 antialias=True),
            T2.RandomHorizontalFlip(),
        ])
    except ImportError:
        import torchvision.transforms as T1
        return T1.Compose([
            T1.RandomResizedCrop(image_size, scale=tuple(scale), ratio=tuple(ratio),
                                 antialias=True),
            T1.RandomHorizontalFlip(),
        ])


class TrainTensorCache(_CacheDatasetBase):
    """Train cache dataset: decoded full image -> random crop/flip -> normalize.

    NOT bit-identical to the PIL RandomResizedCrop path (see module docstring).
    Randomness comes from torch's global RNG (as in the PIL path), so seed with
    torch.manual_seed for reproducible augmentation in a single process.
    With `normalize_on_gpu=True` returns the augmented uint8 [3,S,S] tensor.
    """

    def __init__(self, obj: dict, normalize_on_gpu: bool = False) -> None:
        super().__init__(obj)
        self.buffer: torch.Tensor = obj["buffer"]
        self.offsets: list[int] = [int(x) for x in obj["offsets"].tolist()]
        self.shapes: list[tuple[int, int, int]] = [tuple(int(v) for v in s) for s in obj["shapes"].tolist()]
        self.normalize_on_gpu = normalize_on_gpu
        self.aug = _build_train_aug(int(self.meta["image_size"]),
                                    self.meta["rrc_scale"], self.meta["rrc_ratio"])

    def decoded(self, idx: int) -> torch.Tensor:
        """Return the cached full-resolution uint8 CHW image (a view, no copy)."""
        c, h, w = self.shapes[idx]
        off = self.offsets[idx]
        return self.buffer[off:off + c * h * w].view(c, h, w)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return (augmented image, label) for index `idx`."""
        u8 = self.aug(self.decoded(idx))
        if self.normalize_on_gpu:
            return u8, self.targets[idx]
        return self._to_normalized(u8), self.targets[idx]


def normalize_batch(u8_batch: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    """Normalize a uint8 [B,3,H,W] batch on its device: (x/255 - mean) / std.

    For use with `normalize_on_gpu=True`. The divisor is a device tensor (not a
    Python scalar) so CUDA performs a true division rather than a multiply by
    the reciprocal. Bit-identity with the CPU path on GPU is UNVERIFIED — if a
    run needs exact parity, use the default CPU normalization.
    """
    dev = u8_batch.device
    dt = torch.get_default_dtype()
    x = u8_batch.to(dtype=dt).div(torch.tensor(255.0, dtype=dt, device=dev))
    m = torch.as_tensor(mean, dtype=dt, device=dev).view(1, -1, 1, 1)
    s = torch.as_tensor(std, dtype=dt, device=dev).view(1, -1, 1, 1)
    return x.sub(m).div(s)


def load_eval_cache(path: str, cfg: dict, normalize_on_gpu: bool = False,
                    verify_hash: bool = True, mmap: bool = False) -> Dataset:
    """Load an eval cache as an ImageFolder-like Dataset.

    Args:
        path: Path to `<dataset>_<split>_eval_u8.pt`.
        cfg: Parsed config (load_config); dataset name / image_size / resize /
            mean / std / label_mode must match the cache metadata.
        normalize_on_gpu: Return uint8 images; normalize with normalize_batch().
        verify_hash: Recompute and check the sha256 on load (~1 pass over data).
        mmap: Memory-map the file instead of reading it into RAM (torch>=2.1).

    Returns:
        EvalTensorCache yielding (normalized float [3,S,S], label), with
        `.targets`, `.samples`, `.classes` like a remapped ImageFolder.
    """
    obj = _load_checked(path, cfg, "eval", verify_hash, mmap)
    return EvalTensorCache(obj, normalize_on_gpu=normalize_on_gpu)


def load_train_cache(path: str, cfg: dict, normalize_on_gpu: bool = False,
                     verify_hash: bool = True, mmap: bool = False) -> Dataset:
    """Load a train cache as an augmenting Dataset (see TrainTensorCache).

    Args are as for load_eval_cache. The augmentation is NOT pixel-identical
    to the PIL training transform (tensor resize vs PIL resize).
    """
    obj = _load_checked(path, cfg, "train", verify_hash, mmap)
    return TrainTensorCache(obj, normalize_on_gpu=normalize_on_gpu)


# ── Building ──────────────────────────────────────────────────────────────────

def _progress(it, total: int, desc: str, enabled: bool):
    """Wrap an iterable in tqdm if available/enabled."""
    if not enabled:
        return it
    try:
        from tqdm import tqdm
        return tqdm(it, total=total, desc=desc)
    except ImportError:
        return it


def _base_meta(cfg: dict, split: str, root: str, ds, n: int, limit: Optional[int]) -> dict:
    """Common metadata for both cache kinds."""
    meta = _cfg_signature(cfg)
    meta.update({
        "split": split,
        "root": root,
        "rel_paths": [Path(os.path.relpath(p, root)).as_posix() for p, _ in ds.samples[:n]],
        "classes": list(ds.classes),
        "count": n,
        "limit": limit,
        "versions": _version_info(),
    })
    return meta


def _eval_geometry(cfg: dict):
    """Return (Compose[Resize, CenterCrop], description) from build_eval_transform.

    Uses the repo's own transform objects so the cache can never drift from the
    live pipeline; refuses if build_eval_transform's structure has changed.
    """
    import torchvision.transforms as T
    from utils.datasets import build_eval_transform

    ts = build_eval_transform(cfg).transforms
    kinds = [type(t) for t in ts]
    if kinds != [T.Resize, T.CenterCrop, T.ToTensor, T.Normalize]:
        raise RuntimeError(
            f"build_eval_transform is now {[k.__name__ for k in kinds]}; the eval "
            f"cache assumes [Resize, CenterCrop, ToTensor, Normalize]. Update tensor_cache.py."
        )
    norm = ts[3]
    if [float(x) for x in norm.mean] != _cfg_signature(cfg)["mean"] or \
       [float(x) for x in norm.std] != _cfg_signature(cfg)["std"]:
        raise RuntimeError("Normalize mean/std differ from cfg — unexpected.")
    return T.Compose(ts[:2]), [repr(t) for t in ts]


def build_eval_cache(root: str, cfg: dict, split: str, limit: Optional[int] = None,
                     progress: bool = True) -> dict:
    """Decode + Resize + CenterCrop every image once into a uint8 [N,3,S,S] tensor.

    Uses build_remapped_folder (same ImageFolder ordering, same default_loader
    with .convert('RGB'), same label remap) with transform = the first two
    stages of build_eval_transform.

    Args:
        root: Split directory (absolute or project-relative, as in config).
        cfg: Parsed config with the target dataset active.
        split: Split name recorded in metadata ("val"/"train").
        limit: Cache only the first `limit` samples (debug only).
        progress: Show a tqdm bar.

    Returns:
        Cache dict ready for save_cache().
    """
    from models.loader import resolve_data_path
    from utils.datasets import build_remapped_folder, _ROOT

    geom, tdesc = _eval_geometry(cfg)
    ds = build_remapped_folder(root, geom, cfg)
    n = len(ds) if limit is None else min(int(limit), len(ds))
    s = int(cfg["dataset"]["image_size"])

    images = torch.empty((n, 3, s, s), dtype=torch.uint8)  # preallocated
    labels = torch.empty((n,), dtype=torch.int64)
    for i in _progress(range(n), n, f"eval-cache {split}", progress):
        img, lbl = ds[i]  # PIL RGB after Resize+CenterCrop (ImageFolder path)
        if img.mode != "RGB":
            raise RuntimeError(f"{ds.samples[i][0]}: PIL mode {img.mode}, expected RGB")
        arr = np.array(img, dtype=np.uint8, copy=True)  # what ToTensor does
        if arr.shape != (s, s, 3):
            raise RuntimeError(f"{ds.samples[i][0]}: crop shape {arr.shape} != {(s, s, 3)}")
        images[i] = torch.from_numpy(arr).permute(2, 0, 1)
        labels[i] = int(lbl)

    meta = _base_meta(cfg, split, str(resolve_data_path(_ROOT, root)), ds, n, limit)
    meta.update({"kind": "eval", "transform": tdesc,
                 "stored_stage": "uint8 CHW after Resize+CenterCrop, before ToTensor",
                 "nbytes": images.numel()})
    obj = {"format": CACHE_FORMAT, "version": CACHE_VERSION, "kind": "eval",
           "images": images, "labels": labels, "meta": meta}
    meta["sha256"] = compute_sha256(_payload_tensors(obj))
    return obj


def scan_image_sizes(ds, n: int, progress: bool = True) -> list[tuple[int, int]]:
    """Read (W, H) from image headers only (no full decode) for the first n samples."""
    from PIL import Image

    sizes = []
    for i in _progress(range(n), n, "header scan", progress):
        with Image.open(ds.samples[i][0]) as im:
            sizes.append(im.size)
    return sizes


def build_train_cache(root: str, cfg: dict, split: str, limit: Optional[int] = None,
                      progress: bool = True, max_bytes: Optional[int] = None) -> dict:
    """Decode every image once at FULL resolution into one flat uint8 buffer.

    Pass 1 reads image headers to size the buffer (printed before decoding);
    pass 2 decodes with ImageFolder's own loader (default_loader -> RGB).

    Args:
        root, cfg, split, limit, progress: as for build_eval_cache.
        max_bytes: Abort before decoding if the buffer would exceed this.

    Returns:
        Cache dict ready for save_cache().
    """
    import torchvision.transforms as T
    from models.loader import resolve_data_path
    from utils.datasets import build_remapped_folder, build_train_transform, _ROOT

    tts = build_train_transform(cfg).transforms
    rrc = tts[0]
    if not isinstance(rrc, T.RandomResizedCrop) or not isinstance(tts[1], T.RandomHorizontalFlip):
        raise RuntimeError("build_train_transform no longer starts with RandomResizedCrop+HFlip; update tensor_cache.py.")

    ds = build_remapped_folder(root, None, cfg)
    n = len(ds) if limit is None else min(int(limit), len(ds))

    sizes = scan_image_sizes(ds, n, progress)
    shapes = torch.tensor([[3, h, w] for (w, h) in sizes], dtype=torch.int64).reshape(-1, 3)
    numel = shapes.prod(dim=1)
    offsets = torch.zeros((n,), dtype=torch.int64)
    if n > 1:
        offsets[1:] = torch.cumsum(numel, 0)[:-1]
    total = int(numel.sum().item()) if n else 0
    print(f"[tensor_cache] train buffer: {n} images, {total:,} bytes "
          f"({total / 2**30:.2f} GiB) — computed from image headers")
    if max_bytes is not None and total > max_bytes:
        raise MemoryError(f"Train cache would need {total:,} bytes > limit {max_bytes:,}; aborting before decode.")

    buffer = torch.empty((total,), dtype=torch.uint8)
    labels = torch.empty((n,), dtype=torch.int64)
    for i in _progress(range(n), n, f"train-cache {split}", progress):
        path, lbl = ds.samples[i]
        img = ds.loader(path)  # default_loader: open + convert('RGB')
        arr = np.array(img, dtype=np.uint8, copy=True)
        c, h, w = (int(v) for v in shapes[i].tolist())
        if arr.shape != (h, w, c):
            raise RuntimeError(f"{path}: decoded shape {arr.shape} != header {(h, w, c)}")
        off = int(offsets[i])
        buffer[off:off + c * h * w] = torch.from_numpy(arr).permute(2, 0, 1).reshape(-1)
        labels[i] = int(lbl)

    meta = _base_meta(cfg, split, str(resolve_data_path(_ROOT, root)), ds, n, limit)
    meta.update({"kind": "train", "transform": [repr(t) for t in tts],
                 "stored_stage": "uint8 CHW full-resolution decoded RGB (no resize)",
                 "rrc_scale": [float(x) for x in rrc.scale],
                 "rrc_ratio": [float(x) for x in rrc.ratio],
                 "load_time_aug": "torchvision tensor RandomResizedCrop(antialias=True, bilinear) "
                                  "+ RandomHorizontalFlip; NOT pixel-identical to PIL path",
                 "nbytes": total})
    obj = {"format": CACHE_FORMAT, "version": CACHE_VERSION, "kind": "train",
           "buffer": buffer, "offsets": offsets, "shapes": shapes, "labels": labels,
           "meta": meta}
    meta["sha256"] = compute_sha256(_payload_tensors(obj))
    return obj
