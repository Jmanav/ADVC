"""
scripts/diagnose_at_lr.py

CEILING PROBE — is the AT recipe under-powered for tiny-imagenet, or is a
frozen-head DeiT-S on tiny-imagenet fundamentally low-signal?

Background
----------
Phase-2 AT recovered robustness well on imagenette (fp32 fgsm 0.614 -> 0.806) but
failed on tiny-imagenet (fp32 fgsm 0.048 -> 0.140; int8/int4 got WORSE). The
identical code + config produced both, so it is not a plain code/LR bug. The
suspected cause is that tiny-imagenet is a much harder frozen-head transfer task
(200 classes, 64px upscaled 8x, clean_acc ~0.48) so the gentle last-4-blocks /
low-LR / 7-epoch recipe has little signal to amplify.

This probe settles it cheaply. It runs SHORT AT on a small tiny-imagenet subset
under escalating recipes and reports FGSM robust_acc before/after each:

  1. baseline  : current config LR, last 4 blocks unfrozen  (reproduces failure)
  2. higher-lr : lr=1e-4,            last 4 blocks unfrozen
  3. lr+capacity: lr=1e-4,           last 8 blocks unfrozen

Reference points: imagenette AT reached 0.806 ("what good looks like"); the failed
full tiny-imagenet run reached 0.140.

Decision rule (printed as a verdict)
------------------------------------
  - If (2) or (3) pushes FGSM robust_acc clearly up (>0.30) -> the recipe was
    under-powered; apply the winning setting to base.yaml / _freeze_backbone and
    rerun Phase 2.
  - If even (3) stays near 0.14 -> frozen-head tiny-imagenet is genuinely
    low-signal; that is a paper-scope decision, not a code fix. Do NOT spend GPU
    on a full rerun.

Correctness
-----------
Uses `utils.datasets.build_remapped_folder` so tiny-imagenet's 200 wnid folders
are remapped to their true ImageNet-1000 indices (the frozen 1000-way head needs
this — a raw ImageFolder would train against 0-199 labels and be meaningless).
Writes NOTHING persistent: no checkpoints (save_every_epoch=False), no CSVs. Safe
to run in any Kaggle session without disturbing real results or resume state.

Usage (Kaggle, after Cell 4 data-prep has produced the tiny_if dirs):
    python scripts/diagnose_at_lr.py                    # full 3-setting probe (fp32)
    python scripts/diagnose_at_lr.py --n 800 --epochs 3 # bigger/longer probe
    python scripts/diagnose_at_lr.py --compression int8 # probe a quantized level
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchattacks
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Resolve project root so sibling packages import cleanly.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models.loader import load_config, load_model  # noqa: E402
from utils.datasets import (  # noqa: E402
    build_eval_transform,
    build_train_transform,
    build_remapped_folder,
)


class _LogitsWrapper(nn.Module):
    """Unwrap HuggingFace ImageClassifierOutput to a plain (N, C) tensor.

    Mirrors the wrapper used in defenses/adversarial_training.py so FGSM sees a
    plain tensor for both timm (fp32) and HuggingFace-style (int8/int4) models.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out.logits if hasattr(out, "logits") else out


def _set_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward pass returning a plain (N, C) logits tensor."""
    out = model(x)
    return out.logits if hasattr(out, "logits") else out


def _fgsm_robust_acc(
    model: nn.Module,
    loader: DataLoader,
    fgsm: torchattacks.FGSM,
    device: str,
) -> tuple[float, float]:
    """Measure (clean_acc, fgsm_robust_acc) on the loader.

    Args:
        model:  Model to evaluate (eval mode set internally, restored after).
        loader: DataLoader of ImageNet-remapped (images, labels).
        fgsm:   Configured FGSM attack bound to a _LogitsWrapper of `model`.
        device: Device string.

    Returns:
        (clean_acc, robust_acc), both in [0, 1].
    """
    was_training = model.training
    model.eval()
    clean_correct = adv_correct = total = 0
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_correct += (_logits(model, images).argmax(1) == labels).sum().item()

        adv = fgsm(images, labels)
        with torch.no_grad():
            adv_correct += (_logits(model, adv).argmax(1) == labels).sum().item()

        total += labels.size(0)

    if was_training:
        model.train()
    return clean_correct / total, adv_correct / total


def _run_setting(
    name: str,
    lr: float,
    unfreeze_blocks: int,
    cfg: dict,
    compression: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mean: list,
    std: list,
    at_eps: float,
    epochs: int,
    device: str,
) -> dict:
    """Run one probe setting: fresh model -> measure BEFORE -> short AT -> AFTER.

    Each setting reloads the model from scratch so settings don't contaminate
    each other. Returns a dict of the measured numbers for the summary table.
    """
    from defenses.adversarial_training import adversarial_train

    print("\n" + "-" * 70)
    print(f">>> Setting '{name}': lr={lr:g}, unfreeze_blocks={unfreeze_blocks}, "
          f"epochs={epochs}")
    print("-" * 70)

    _set_seeds(cfg["seed"])  # identical data order / init per setting
    model = load_model("deit_small", compression, cfg, device=device)

    fgsm_eval = torchattacks.FGSM(_LogitsWrapper(model), eps=at_eps)
    fgsm_eval.set_normalization_used(mean=mean, std=std)
    fgsm_eval.set_model_training_mode(model_training=False, batchnorm_training=False)

    clean_b, robust_b = _fgsm_robust_acc(model, val_loader, fgsm_eval, device)
    print(f"[{name}] BEFORE AT  clean={clean_b:.4f}  fgsm_robust={robust_b:.4f}")

    diag_cfg = {
        **cfg,
        "defense": {
            **cfg["defense"],
            "epochs": epochs,
            "lr": lr,                 # scalar -> used directly for any compression
            "save_every_epoch": False,  # write nothing to disk
        },
    }
    adversarial_train(
        model, train_loader, diag_cfg,
        compression=compression,
        num_unfrozen_blocks=unfreeze_blocks,
    )

    clean_a, robust_a = _fgsm_robust_acc(model, val_loader, fgsm_eval, device)
    print(f"[{name}] AFTER  AT  clean={clean_a:.4f}  fgsm_robust={robust_a:.4f}")

    # Free GPU memory before the next setting reloads a fresh model.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": name, "lr": lr, "blocks": unfreeze_blocks,
        "clean_before": clean_b, "robust_before": robust_b,
        "clean_after": clean_a, "robust_after": robust_a,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="AT ceiling probe for tiny-imagenet.")
    ap.add_argument("--epochs", type=int, default=2,
                    help="AT epochs per setting (default 2 — cheap).")
    ap.add_argument("--n", type=int, default=500,
                    help="Train/val subset size per split (default 500 each).")
    ap.add_argument("--compression", default="fp32",
                    choices=["fp32", "int8", "int4"],
                    help="Compression level (default fp32 — most trainable capacity, "
                         "clearest signal).")
    args = ap.parse_args()

    cfg = load_config(str(_ROOT / "configs/base.yaml"))
    _set_seeds(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA — this will be extremely slow. Run on Kaggle GPU.")

    ds_cfg = cfg["dataset"]
    mean, std = ds_cfg["mean"], ds_cfg["std"]
    at_eps = cfg["defense"]["at_eps"]
    base_lr_cfg = cfg["defense"]["lr"]
    base_lr = (base_lr_cfg[args.compression]
               if isinstance(base_lr_cfg, dict) else base_lr_cfg)

    print(f"=== AT ceiling probe — dataset={ds_cfg['name']} "
          f"compression={args.compression} n={args.n} epochs={args.epochs} ===")
    print(f"    config baseline lr for {args.compression} = {base_lr:g}\n")

    # Build remapped datasets ONCE (labels -> ImageNet-1000 indices).
    train_full = build_remapped_folder(ds_cfg["train_dir"],
                                        build_train_transform(cfg), cfg)
    val_full = build_remapped_folder(ds_cfg["val_dir"],
                                     build_eval_transform(cfg), cfg)

    # Deterministic subsets (seed already set).
    train_idx = random.sample(range(len(train_full)), min(args.n, len(train_full)))
    val_idx = random.sample(range(len(val_full)), min(args.n, len(val_full)))
    train_loader = DataLoader(Subset(train_full, train_idx),
                              batch_size=cfg["defense"]["batch_size"],
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(val_full, val_idx),
                            batch_size=cfg["eval"]["batch_size"],
                            shuffle=False, num_workers=2)

    settings = [
        ("baseline",    base_lr, 4),
        ("higher-lr",   1e-4,    4),
        ("lr+capacity", 1e-4,    8),
    ]
    results = [
        _run_setting(name, lr, blocks, cfg, args.compression,
                     train_loader, val_loader, mean, std, at_eps,
                     args.epochs, device)
        for (name, lr, blocks) in settings
    ]

    # ---- Summary table ----
    IMAGENETTE_AT = 0.806   # imagenette fp32 AT fgsm — "what good looks like"
    FAILED_TINY = 0.140     # failed full tiny-imagenet fp32 AT fgsm
    print("\n" + "=" * 70)
    print("SUMMARY — FGSM robust_acc (subset, tiny-imagenet)")
    print("=" * 70)
    print(f"{'setting':<13} {'lr':>8} {'blk':>4} {'clean→':>8} "
          f"{'robust:before→after':>22}")
    for r in results:
        print(f"{r['name']:<13} {r['lr']:>8.0e} {r['blocks']:>4} "
              f"{r['clean_after']:>8.3f} "
              f"   {r['robust_before']:.3f} → {r['robust_after']:.3f}")
    print(f"\nreference: imagenette AT reached {IMAGENETTE_AT:.3f}; "
          f"failed tiny full run reached {FAILED_TINY:.3f}")

    best = max(results, key=lambda r: r["robust_after"])
    print("\n" + "=" * 70)
    if best["robust_after"] > 0.30:
        print(f"VERDICT: ✅ RECIPE WAS UNDER-POWERED. Best setting '{best['name']}' "
              f"(lr={best['lr']:g}, {best['blocks']} blocks) reached "
              f"fgsm_robust={best['robust_after']:.3f} on the subset — well above "
              f"the failed 0.14.")
        print("         -> Apply that recipe (base.yaml defense.lr and/or "
              "defense.num_unfrozen_blocks) and rerun Phase 2.")
    elif best["robust_after"] > FAILED_TINY + 0.05:
        print(f"VERDICT: ⚠️  PARTIAL. Best '{best['name']}' reached "
              f"{best['robust_after']:.3f} — better than 0.14 but modest. "
              "Consider an even stronger recipe (higher lr / more blocks / more "
              "epochs) before committing to a full rerun.")
    else:
        print(f"VERDICT: ❌ LOW-SIGNAL SETTING. Even the strongest probe "
              f"('{best['name']}') only reached {best['robust_after']:.3f}, near "
              f"the failed 0.14. Frozen-head tiny-imagenet appears fundamentally "
              "hard for this AT recipe.")
        print("         -> This is a PAPER-SCOPE decision, not a code fix. Options: "
              "retrain the head, report the low numbers as a finding, or keep "
              "imagenette primary. Do NOT spend GPU on a full Phase-2 rerun yet.")
    print("=" * 70)


if __name__ == "__main__":
    main()
