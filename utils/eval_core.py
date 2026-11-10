"""
utils/eval_core.py

Seeded, per-image clean / FGSM / PGD-k evaluation shared by run.py.

The algorithm matches the Q4 sweep
(scratchpad/checks/pgd_epoch_sweep.py:36-96) so that numbers stay comparable:
- the same stratified subset rule
- the same Logits wrapper
- a seed before every attack call, seed = attack_seed(seed, b) = seed + b
- the order clean -> fgsm -> pgd within each batch

The attacks come from attacks.fgsm.build_attack / attacks.pgd.build_attack.
Those wrap torchattacks with set_normalization_used(mean, std), so their
inputs and outputs are ImageNet-normalised tensors.

torch is imported at module level (needed for nn.Module). The attack modules
(torchattacks) are imported lazily, so CPU unit tests can import
stratified_indices without torchattacks installed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

from utils.seeding import attack_seed

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **_kw):  # type: ignore[no-redef]
        return it


class Logits(nn.Module):
    """Return plain logits whether the wrapped model returns a tensor or an
    object with `.logits` (HF-style); identical to the sweep's `Logits`."""

    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.m(x)
        return out.logits if hasattr(out, "logits") else out


def wrap_logits(model: nn.Module) -> nn.Module:
    """Wrap in Logits unless already wrapped; returns the module in eval mode."""
    return (model if isinstance(model, Logits) else Logits(model)).eval()


def stratified_indices(targets: Sequence[int], per_class: int | str | None, seed: int) -> list[int]:
    """Class-stratified subset: for each class (sorted), the first `per_class`
    entries of a torch.randperm over that class's members, using ONE
    torch.Generator seeded with `seed` and consumed class by class.

    Identical to pgd_epoch_sweep.py:46-56 (which iterated ds.samples; targets
    are the same labels in the same order). per_class in (None, "all", <=0)
    returns all indices in dataset order. A class with fewer members than
    per_class contributes all its members.
    """
    if per_class is None or (isinstance(per_class, str) and per_class.lower() == "all"):
        return list(range(len(targets)))
    per_class = int(per_class)
    if per_class <= 0:
        return list(range(len(targets)))
    g = torch.Generator().manual_seed(int(seed))
    by_class: dict = {}
    for i, y in enumerate(targets):
        by_class.setdefault(int(y), []).append(i)
    idx: list[int] = []
    for y in sorted(by_class):
        members = by_class[y]
        pick = torch.randperm(len(members), generator=g)[:per_class].tolist()
        idx.extend(members[j] for j in pick)
    return idx


def subset_signature(dataset_name: str, indices: Sequence[int], labels: Sequence[int]) -> str:
    """Short sha256 identifying an evaluation subset (dataset, indices, labels).
    Including labels makes the signature differ if a cache orders images
    differently from ImageFolder."""
    payload = json.dumps({"dataset": dataset_name, "indices": [int(i) for i in indices],
                          "labels": [int(y) for y in labels]}, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


_PGD_RE = re.compile(r"^pgd(\d+)?$")


def attack_params(cfg: dict, name: str) -> dict:
    """Parameters actually used for attack `name` ('clean', 'fgsm', 'pgd', 'pgdK')."""
    if name == "clean":
        return {"name": "clean"}
    if name == "fgsm":
        return {"name": "fgsm", "eps": float(cfg["fgsm"]["eps"]), "lib": "torchattacks.FGSM"}
    m = _PGD_RE.match(name)
    if m:
        steps = int(m.group(1)) if m.group(1) else int(cfg["pgd"]["steps"])
        return {"name": name, "eps": float(cfg["pgd"]["eps"]), "alpha": float(cfg["pgd"]["alpha"]),
                "steps": steps, "random_start": True, "lib": "torchattacks.PGD"}
    raise ValueError(f"Unknown attack {name!r}; use clean, fgsm, pgd or pgd<K> (e.g. pgd20)")


def attack_signature(cfg: dict, name: str) -> str:
    """Short sha256 of attack_params plus normalisation constants."""
    payload = {"p": attack_params(cfg, name), "mean": cfg["dataset"]["mean"], "std": cfg["dataset"]["std"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_attack_fn(model: nn.Module, cfg: dict, name: str):
    """Return a callable (images, labels) -> adv_images for `name`, or None for 'clean'.

    pgdK deep-copies cfg with pgd.steps = K so base.yaml is never mutated.
    """
    p = attack_params(cfg, name)
    if name == "clean":
        return None
    if name == "fgsm":
        import attacks.fgsm as fgsm_mod

        return fgsm_mod.build_attack(model, cfg)
    import attacks.pgd as pgd_mod

    c = copy.deepcopy(cfg)
    c["pgd"]["steps"] = p["steps"]
    return pgd_mod.build_attack(model, c)


def pixel_linf(adv: torch.Tensor, x: torch.Tensor, cfg: dict) -> float:
    """L-inf distance in [0,1] pixel space between two normalised batches."""
    std = torch.tensor(cfg["dataset"]["std"], device=x.device, dtype=torch.float32).view(1, -1, 1, 1)
    return float(((adv.float() - x.float()) * std).abs().max().item())


@torch.no_grad()
def _pred_correct(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (model(x).argmax(1) == y).to(torch.uint8).cpu()


def evaluate(
    model: nn.Module,
    loader: Iterable,
    cfg: dict,
    attacks: Sequence[str] = ("clean", "fgsm", "pgd20"),
    seed: int = 0,
    runlog: Any = None,
    device: str = "cuda",
    indices: Sequence[int] | None = None,
    tag: str = "",
) -> dict:
    """Per-image correctness of `model` under each attack.

    For batch b, `torch.manual_seed(attack_seed(seed, b))` is called right
    before each attack call (so FGSM and PGD see the same RNG state as in the
    sweep). An exception inside an attack is re-raised as RuntimeError naming
    the attack and batch (the original is chained, so the full traceback is
    kept by the RunLog Step that wraps this call).

    Args:
        model: Classifier (wrapped in Logits here if needed).
        loader: Iterable of (normalised images, labels), shuffle=False.
        cfg: Resolved base config.
        attacks: Subset of 'clean', 'fgsm', 'pgd', 'pgdK'.
        seed: Run seed for attack_seed.
        runlog: Optional RunLog; per-attack timings are added as '<tag>/<attack>'.
        device: CUDA device string.
        indices: Dataset indices in loader order (stored per image).
        tag: Label for timings / progress bar.

    Returns:
        {"n", "dataset_indices", "labels", "correct": {attack: [0/1 ...]},
         "seconds": {attack: s}, "max_linf": {attack: float}}
    """
    model = wrap_logits(model)
    attacks = list(attacks)
    fns = {a: build_attack_fn(model, cfg, a) for a in attacks}
    correct: dict[str, list[int]] = {a: [] for a in attacks}
    secs: dict[str, float] = {a: 0.0 for a in attacks}
    max_linf: dict[str, float] = {a: 0.0 for a in attacks if a != "clean"}
    labels: list[int] = []
    use_cuda = str(device).startswith("cuda")

    for b, (x, y) in enumerate(tqdm(loader, desc=tag or "eval", leave=False)):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        labels.extend(int(v) for v in y.tolist())
        for a in attacks:
            t0 = time.perf_counter()
            try:
                if a == "clean":
                    flags = _pred_correct(model, x, y)
                else:
                    torch.manual_seed(attack_seed(seed, b))
                    adv = fns[a](x, y)
                    flags = _pred_correct(model, adv, y)
                    max_linf[a] = max(max_linf[a], pixel_linf(adv, x, cfg))
                    del adv
            except Exception as exc:
                raise RuntimeError(f"[{tag}] attack {a!r} failed at batch {b} "
                                   f"(x {tuple(x.shape)} {x.dtype}, device {x.device})") from exc
            if use_cuda:
                torch.cuda.synchronize()
            secs[a] += time.perf_counter() - t0
            correct[a].extend(int(v) for v in flags.tolist())

    n = len(labels)
    if indices is not None and len(indices) != n:
        raise RuntimeError(f"loader yielded {n} images but {len(indices)} indices were given")
    if runlog is not None:
        for a in attacks:
            runlog.add_timing(f"{tag}/{a}" if tag else a, secs[a], True)
    return {
        "n": n,
        "dataset_indices": [int(i) for i in indices] if indices is not None else None,
        "labels": labels,
        "correct": correct,
        "seconds": {a: round(s, 3) for a, s in secs.items()},
        "max_linf": max_linf,
    }
