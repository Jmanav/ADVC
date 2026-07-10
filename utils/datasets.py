"""
utils/datasets.py

Single source of truth for dataset label remapping and transform construction,
shared by every experiment and utility script.

Why this exists
---------------
DeiT-S is loaded with its pretrained 1000-way ImageNet-1k head and that head is
never re-trained. The experiments run on *subsets* of ImageNet-1k (ImageNette =
10 classes, Tiny-ImageNet = 200 classes) whose folders are named by WordNet
synset (e.g. "n01440764"). torchvision's ImageFolder assigns those folders
contiguous labels 0..K-1, which do NOT match the pretrained head's output
indices. Each `_*_TO_IMAGENET` table below remaps a subset folder's synset to
its true ImageNet-1k index, so the frozen head works with no fine-tuning.

Everything is driven by `cfg["dataset"]["name"]`, so switching datasets is a
pure config change (see configs/base.yaml).
"""

from pathlib import Path

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from models.loader import resolve_data_path

_ROOT = Path(__file__).resolve().parent.parent


# ── Subset synset → ImageNet-1k index maps ────────────────────────────────────

# ImageNette (10 classes). Verbatim from the original per-file tables.
_IMAGENETTE_TO_IMAGENET: dict[str, int] = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

# Tiny-ImageNet (200 classes) is built at runtime from the dataset's wnids.txt
# and timm's canonical ImageNet-1k ordering — see _build_tiny_imagenet_map().
# No hand-maintained literal, so it can never drift from the pretrained head.
# Cached after first build (keyed by resolved wnids.txt path).
_TINY_MAP_CACHE: dict[str, dict[str, int]] = {}


def _tiny_wnids_path(cfg: dict) -> str:
    """Resolve the path to Tiny-ImageNet's wnids.txt.

    Uses `dataset.wnids` from config if set; otherwise derives it as the sibling
    of the (parent of) train_dir — i.e. the tiny-imagenet-200 root that also
    contains train_if/val_if. Falls back to <train_dir>/../wnids.txt.
    """
    ds_cfg = cfg["dataset"]
    if ds_cfg.get("wnids"):
        return str(resolve_data_path(_ROOT, ds_cfg["wnids"]))
    # train_dir is like <root>/train_if — wnids.txt lives in <root>.
    train_dir = resolve_data_path(_ROOT, ds_cfg["train_dir"])
    return str(Path(train_dir).parent / "wnids.txt")


def _build_tiny_imagenet_map(cfg: dict) -> dict[str, int]:
    """Build Tiny-ImageNet synset→ImageNet-1k index map from wnids.txt + timm.

    Tiny-ImageNet's 200 classes are all ImageNet-1k synsets, so each maps to its
    position in timm's canonical ILSVRC2012 ordering (the ordering the pretrained
    DeiT-S head produces). All 200 MUST resolve — a miss means the wrong ordering
    and would silently collapse accuracy, so we assert instead.
    """
    wnids_path = _tiny_wnids_path(cfg)
    if wnids_path in _TINY_MAP_CACHE:
        return _TINY_MAP_CACHE[wnids_path]

    if not Path(wnids_path).is_file():
        raise FileNotFoundError(
            f"Tiny-ImageNet wnids.txt not found at {wnids_path}. "
            f"Set dataset.wnids in configs/base.yaml, or ensure train_dir's "
            f"parent contains wnids.txt."
        )

    from timm.data.imagenet_info import ImageNetInfo

    info = ImageNetInfo()  # imagenet-1k
    synset_to_idx = {
        info.index_to_label_name(i): i for i in range(info.num_classes())
    }

    with open(wnids_path) as f:
        wnids = [line.strip() for line in f if line.strip()]

    mapping: dict[str, int] = {}
    missing = []
    for w in wnids:
        if w in synset_to_idx:
            mapping[w] = synset_to_idx[w]
        else:
            missing.append(w)

    assert not missing, (
        f"{len(missing)} Tiny-ImageNet wnids did not resolve to an ImageNet-1k "
        f"index (wrong timm ordering/source?). First few: {missing[:5]}"
    )
    assert len(mapping) == 200, f"Expected 200 entries, got {len(mapping)}."

    _TINY_MAP_CACHE[wnids_path] = mapping
    print(f"[datasets] Built Tiny-ImageNet map: {len(mapping)} classes from {wnids_path}")
    return mapping


def synset_to_imagenet_map(cfg: dict) -> dict[str, int]:
    """Return the synset→ImageNet-1k index map for the active dataset.

    ImageNette uses a static table; Tiny-ImageNet is built (and cached) at
    runtime from wnids.txt + timm's ImageNet ordering.

    Args:
        cfg: Parsed base.yaml config (with active dataset flattened).

    Returns:
        Mapping from folder synset string to ImageNet-1k output index.
    """
    name = cfg["dataset"]["name"]
    if name == "imagenette":
        return _IMAGENETTE_TO_IMAGENET
    if name == "tiny-imagenet":
        return _build_tiny_imagenet_map(cfg)
    raise ValueError(
        f"Unknown dataset.name={name!r}. Known: ['imagenette', 'tiny-imagenet']."
    )


def remap_subset_labels(dataset: ImageFolder, cfg: dict) -> ImageFolder:
    """Remap ImageFolder targets to ImageNet-1k indices for subset datasets.

    No-op when:
      - the active dataset uses `label_mode: raw` (the model has a fine-tuned
        K-way head, so raw 0..K-1 ImageFolder labels are correct — e.g.
        tiny-imagenet with a fine-tuned 200-way head), or
      - the dataset already has >= 1000 classes (full ImageNet).

    Otherwise (label_mode `imagenet`, the default) each folder's synset is mapped
    to its ImageNet-1k index so the frozen pretrained 1000-way head works with no
    fine-tuning. Unknown synsets fall back to their original ImageFolder label.

    Args:
        dataset: An ImageFolder over a subset of ImageNet-1k.
        cfg: Parsed base.yaml config selecting label_mode / synset map.

    Returns:
        The same dataset object with `.samples` / `.targets` remapped (or unchanged
        under label_mode: raw).
    """
    # label_mode: raw → keep contiguous 0..K-1 labels for a fine-tuned K-way head.
    if cfg["dataset"].get("label_mode", "imagenet") == "raw":
        return dataset
    if len(dataset.classes) >= 1000:
        return dataset
    mapping = synset_to_imagenet_map(cfg)
    new_samples = []
    for path, lbl in dataset.samples:
        synset = dataset.classes[lbl]
        new_lbl = mapping.get(synset, lbl)
        new_samples.append((path, new_lbl))
    dataset.samples = new_samples
    dataset.targets = [lbl for _, lbl in new_samples]
    return dataset


# ── Transforms ────────────────────────────────────────────────────────────────

def build_eval_transform(cfg: dict) -> T.Compose:
    """Evaluation transform: Resize → CenterCrop → ToTensor → Normalize.

    `resize` is read per-dataset (defaults to 256 when absent, preserving the
    original ImageNette preprocessing exactly).
    """
    ds_cfg = cfg["dataset"]
    return T.Compose([
        T.Resize(ds_cfg.get("resize", 256)),
        T.CenterCrop(ds_cfg["image_size"]),
        T.ToTensor(),
        T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
    ])


def build_train_transform(cfg: dict) -> T.Compose:
    """Training transform: RandomResizedCrop → HFlip → ToTensor → Normalize."""
    ds_cfg = cfg["dataset"]
    return T.Compose([
        T.RandomResizedCrop(ds_cfg["image_size"]),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=ds_cfg["mean"], std=ds_cfg["std"]),
    ])


# ── ImageFolder + remap in one call ───────────────────────────────────────────

def build_remapped_folder(root: str, transform: T.Compose, cfg: dict) -> ImageFolder:
    """Build an ImageFolder at `root` and remap its labels to ImageNet-1k.

    Args:
        root: Dataset directory (as given in config, absolute or project-relative).
        transform: Transform pipeline to attach to the ImageFolder.
        cfg: Parsed base.yaml config selecting the synset map.

    Returns:
        A label-remapped ImageFolder ready for subsetting / DataLoader.
    """
    dataset = ImageFolder(root=str(resolve_data_path(_ROOT, root)), transform=transform)
    return remap_subset_labels(dataset, cfg)


if __name__ == "__main__":
    # Sanity check: build the eval transform and report the active map size.
    from models.loader import load_config

    cfg = load_config(str(_ROOT / "configs/base.yaml"))
    name = cfg["dataset"]["name"]
    print(f"Active dataset: {name}")
    print(f"resize={cfg['dataset'].get('resize', 256)}, image_size={cfg['dataset']['image_size']}")
    try:
        m = synset_to_imagenet_map(cfg)
        print(f"Synset map entries: {len(m)}")
    except (ValueError, FileNotFoundError, AssertionError, ImportError) as e:
        print(f"[warn] could not build synset map ({type(e).__name__}): {e}")
    _ = build_eval_transform(cfg)
    _ = build_train_transform(cfg)
    print("Transforms built OK.")
