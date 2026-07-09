"""
scripts/diagnose_at_lr.py

CHEAP DIAGNOSTIC — is the AT learning rate the reason adversarial training
failed to recover robustness (and in some cells made it worse)?

Hypothesis
----------
The Phase-2 AT run barely moved FGSM robustness (fp32: 0.048 -> 0.140) and made
int8/int4 *worse*. The AT learning rates in configs/base.yaml (fp32=1e-5,
int8=5e-6, int4=1e-6), combined with a frozen backbone + only 7 epochs, are
~50-100x too small to actually learn a defense. This script tests that on the
clearest case (fp32, biggest LR) as cheaply as possible.

What it does (~10-15 min on a T4, NOT a full rerun)
---------------------------------------------------
  1. Load DeiT-S fp32.
  2. Take a 500-image train subset and a 500-image val subset (seed 42).
  3. Measure FGSM robust_acc BEFORE any AT.
  4. AT-train 2 epochs at a candidate LR (default 1e-4, override with --lr).
  5. Measure FGSM robust_acc AFTER.
  6. Print a verdict.

It writes NOTHING persistent — no checkpoints, no CSVs. Safe to run in any
Kaggle session without disturbing real results or resume state.

Interpreting the result
------------------------
  - Full 7-epoch AT at the OLD lr (1e-5) reached FGSM robust_acc ~0.140.
  - If 2 epochs at the NEW lr on just 500 images beats that comfortably
    (e.g. >0.20), the LR was the bottleneck -> bump base.yaml and rerun Phase 2.
  - If it barely moves, the problem is elsewhere (eval harness / attack scaling)
    and we investigate before spending 6h on a rerun.

Usage (Kaggle, after Cell 4 data-prep has produced the tiny_if dirs):
    python scripts/diagnose_at_lr.py                 # lr=1e-4, 2 epochs
    python scripts/diagnose_at_lr.py --lr 5e-5       # try a different LR
    python scripts/diagnose_at_lr.py --epochs 3 --n 800
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchattacks
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# Resolve project root so sibling packages import cleanly.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models.loader import load_config, load_model, resolve_data_path  # noqa: E402


class _LogitsWrapper(nn.Module):
    """Unwrap HuggingFace ImageClassifierOutput to a plain (N, C) tensor.

    Mirrors the wrapper used in defenses/adversarial_training.py so FGSM sees a
    plain tensor for both timm (fp32) and HuggingFace (int8/int4) models.
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
        model:  Model to evaluate (eval mode is set internally).
        loader: DataLoader of ImageNet-normalised (images, labels).
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Cheap AT learning-rate diagnostic.")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Candidate AT learning rate to test (default 1e-4).")
    ap.add_argument("--epochs", type=int, default=2,
                    help="AT epochs for the diagnostic (default 2).")
    ap.add_argument("--n", type=int, default=500,
                    help="Train/val subset size (default 500 each).")
    ap.add_argument("--compression", default="fp32",
                    choices=["fp32", "int8", "int4"],
                    help="Compression level to test (default fp32 — clearest signal).")
    args = ap.parse_args()

    cfg = load_config(str(_ROOT / "configs/base.yaml"))
    _set_seeds(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA — this will be extremely slow. Run on Kaggle GPU.")

    ds_cfg = cfg["dataset"]
    mean, std = ds_cfg["mean"], ds_cfg["std"]
    at_eps = cfg["defense"]["at_eps"]

    transform = T.Compose([
        T.Resize(ds_cfg["resize"]),
        T.CenterCrop(ds_cfg["image_size"]),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    print(f"=== AT LR diagnostic — {args.compression}, lr={args.lr}, "
          f"{args.epochs} epoch(s), n={args.n} ===\n")

    # Deterministic subsets (seed already set).
    train_full = ImageFolder(str(resolve_data_path(_ROOT, ds_cfg["train_dir"])),
                             transform=transform)
    val_full = ImageFolder(str(resolve_data_path(_ROOT, ds_cfg["val_dir"])),
                           transform=transform)
    train_idx = random.sample(range(len(train_full)), min(args.n, len(train_full)))
    val_idx = random.sample(range(len(val_full)), min(args.n, len(val_full)))

    train_loader = DataLoader(Subset(train_full, train_idx),
                              batch_size=cfg["defense"]["batch_size"],
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(val_full, val_idx),
                            batch_size=cfg["eval"]["batch_size"],
                            shuffle=False, num_workers=2)

    model = load_model("deit_small", args.compression, cfg, device=device)

    # FGSM for EVAL (bound to eval model) and one for TRAIN (bound to same model
    # via wrapper). Both use set_normalization_used so eps is applied in pixel space.
    fgsm_eval = torchattacks.FGSM(_LogitsWrapper(model), eps=at_eps)
    fgsm_eval.set_normalization_used(mean=mean, std=std)
    fgsm_eval.set_model_training_mode(model_training=False, batchnorm_training=False)

    # ---- BEFORE ----
    clean_b, robust_b = _fgsm_robust_acc(model, val_loader, fgsm_eval, device)
    print(f"[BEFORE AT]  clean_acc={clean_b:.4f}  fgsm_robust_acc={robust_b:.4f}\n")

    # ---- AT train (reuse the real training routine, overriding lr + epochs) ----
    from defenses.adversarial_training import adversarial_train

    diag_cfg = {
        **cfg,
        "defense": {
            **cfg["defense"],
            "epochs": args.epochs,
            # Force a scalar lr so adversarial_train uses it directly for any level.
            "lr": args.lr,
            "save_every_epoch": False,   # write nothing to disk
        },
    }
    model = adversarial_train(model, train_loader, diag_cfg,
                              compression=args.compression)

    # ---- AFTER ----
    clean_a, robust_a = _fgsm_robust_acc(model, val_loader, fgsm_eval, device)
    print(f"\n[AFTER AT]   clean_acc={clean_a:.4f}  fgsm_robust_acc={robust_a:.4f}")

    # ---- Verdict ----
    OLD_LR_FULL_RUN = 0.140   # what full 7-epoch AT at old lr reached (fp32 fgsm)
    delta = robust_a - robust_b
    print("\n" + "=" * 60)
    print(f"FGSM robust_acc:  before={robust_b:.4f}  ->  after={robust_a:.4f}"
          f"  (Δ={delta:+.4f})")
    print(f"Reference: full 7-epoch AT at OLD lr reached ~{OLD_LR_FULL_RUN:.3f}")
    if robust_a > OLD_LR_FULL_RUN + 0.05:
        print("VERDICT: ✅ LR was the bottleneck. Just 2 epochs on "
              f"{args.n} images at lr={args.lr} already beat the full old-lr run.")
        print("         -> Bump defense.lr in base.yaml and rerun Phase 2.")
    elif delta > 0.03:
        print("VERDICT: ⚠️  Higher LR helps but not dramatically. Consider a "
              "larger LR or more epochs before a full rerun.")
    else:
        print("VERDICT: ❌ LR does NOT appear to be the bottleneck. Robustness "
              "barely moved. Investigate the eval harness / attack scaling "
              "before spending GPU on a full Phase 2 rerun.")
    print("=" * 60)


if __name__ == "__main__":
    main()
