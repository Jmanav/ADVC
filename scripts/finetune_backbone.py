"""
scripts/finetune_backbone.py

Supervised (NON-adversarial) fine-tune of DeiT-S with a fresh K-way head on the
active dataset, producing the `finetuned_base` checkpoint that the rest of the
pipeline (compress → defend → attack) loads via models/loader.py.

Why this exists
---------------
DeiT-S's frozen 1000-way ImageNet head fails on tiny-imagenet (64px→224, 200
classes): clean_acc ~0.48 and adversarial training cannot recover robustness. A
ceiling probe confirmed no AT recipe fixes it. The real fix is to give the model a
genuine 200-way head fine-tuned on tiny-imagenet, THEN run the robustness pipeline.

This script is dataset-driven by `configs/base.yaml`:
  - head width  = cfg["dataset"]["num_classes"]   (tiny-imagenet: 200)
  - label mode  = cfg["dataset"]["label_mode"]     (tiny-imagenet: raw 0..K-1)
  - output path = dataset-scoped cfg["dataset"]["finetuned_base"]

It only makes sense for datasets with `label_mode: raw` + a sub-1000 num_classes.
Running it for imagenette (label_mode: imagenet, 1000-way frozen head) is a no-op
by design and is refused.

Usage (Kaggle, after data-prep has produced the ImageFolder dirs):
    python scripts/finetune_backbone.py                 # full fine-tune (config epochs)
    python scripts/finetune_backbone.py --epochs 1 --n 4000   # cheap sanity run
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models.loader import load_config, resolve_data_path  # noqa: E402
from utils.datasets import (  # noqa: E402
    build_eval_transform,
    build_train_transform,
    build_remapped_folder,
)


def _set_seeds(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scoped_finetuned_path(cfg: dict) -> Path:
    """Dataset-scoped output path for the fine-tuned base checkpoint.

    Mirrors models/loader.py:_resolve_finetuned_path scoping:
      results/checkpoints/base/deit_small_ft.pt
        -> results/checkpoints/<name>/base/deit_small_ft.pt
    """
    ds_cfg = cfg["dataset"]
    ft_rel = ds_cfg["finetuned_base"]
    p = Path(ft_rel)
    scoped = p.parent.parent / ds_cfg["name"] / p.parent.name / p.name
    return _ROOT / scoped


@torch.no_grad()
def _clean_acc(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Top-1 accuracy on the loader (model set to eval, restored after)."""
    was_training = model.training
    model.eval()
    correct = total = 0
    for images, labels in tqdm(loader, desc="val", leave=False):
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        logits = out.logits if hasattr(out, "logits") else out
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    if was_training:
        model.train()
    return correct / total if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Supervised fine-tune of DeiT-S K-way head.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epoch count (default: config finetune.epochs or 5).")
    ap.add_argument("--n", type=int, default=None,
                    help="Optional train-subset size for a cheap sanity run.")
    ap.add_argument("--model", default="deit_small")
    args = ap.parse_args()

    cfg = load_config(str(_ROOT / "configs/base.yaml"))
    _set_seeds(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA — fine-tuning on CPU is impractically slow.")

    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    num_classes = int(ds_cfg.get("num_classes", 1000))
    label_mode = ds_cfg.get("label_mode", "imagenet")

    # Guard: this script is only meaningful for a fine-tuned K-way head.
    if label_mode != "raw" or num_classes >= 1000:
        raise SystemExit(
            f"[finetune] dataset '{name}' uses label_mode={label_mode!r}, "
            f"num_classes={num_classes} — it relies on the frozen 1000-way ImageNet "
            "head and does NOT need fine-tuning. Nothing to do."
        )

    timm_name = cfg["models"][args.model]["timm_name"]
    ft_cfg = cfg.get("finetune", {})
    epochs = args.epochs if args.epochs is not None else int(ft_cfg.get("epochs", 5))
    lr = float(ft_cfg.get("lr", 1e-4))
    weight_decay = float(ft_cfg.get("weight_decay", 0.05))
    batch_size = int(ft_cfg.get("batch_size", 64))

    print(f"=== Fine-tune {args.model} on {name} — {num_classes}-way head ===")
    print(f"    epochs={epochs} lr={lr:g} wd={weight_decay:g} batch={batch_size} "
          f"device={device}\n")

    # Datasets (build_remapped_folder no-ops the remap under label_mode: raw,
    # so labels are raw 0..K-1 ImageFolder order — matching the K-way head).
    train_ds = build_remapped_folder(ds_cfg["train_dir"], build_train_transform(cfg), cfg)
    val_ds = build_remapped_folder(ds_cfg["val_dir"], build_eval_transform(cfg), cfg)
    assert len(train_ds.classes) == num_classes, (
        f"train folder has {len(train_ds.classes)} classes but num_classes="
        f"{num_classes}"
    )

    if args.n is not None:
        idx = random.sample(range(len(train_ds)), min(args.n, len(train_ds)))
        train_ds = Subset(train_ds, idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=cfg["eval"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                            num_workers=cfg["eval"]["num_workers"], pin_memory=True)

    # Full fine-tune (all params) — the head is fresh and the backbone adapts to
    # the 64px→224 domain. Standard AdamW + cosine schedule.
    model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    acc0 = _clean_acc(model, val_loader, device)
    print(f"[finetune] val clean_acc BEFORE fine-tune (fresh head): {acc0:.4f}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running = correct = total = 0.0
        loop = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", dynamic_ncols=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            logits = out.logits if hasattr(out, "logits") else out
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            bs = labels.size(0)
            running += loss.item() * bs
            correct += (logits.argmax(1) == labels).sum().item()
            total += bs
            loop.set_postfix(loss=f"{running/total:.3f}", acc=f"{correct/total:.3f}")
        scheduler.step()
        val_acc = _clean_acc(model, val_loader, device)
        print(f"[finetune] epoch {epoch}: train_acc={correct/total:.4f} "
              f"val_clean_acc={val_acc:.4f}")

    out_path = _scoped_finetuned_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))
    final_acc = _clean_acc(model, val_loader, device)
    print(f"\n[finetune] DONE. val clean_acc={final_acc:.4f} "
          f"(was {acc0:.4f} with the frozen head).")
    print(f"[finetune] Saved fine-tuned base → {out_path}")
    if final_acc < 0.55:
        print("[finetune] WARNING: clean_acc < 0.55 — below the ~0.7 target. "
              "Fine-tune may need more epochs / higher LR before running the matrix.")


if __name__ == "__main__":
    main()
