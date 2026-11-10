"""
utils/seeding.py

Seeding and evaluation-determinism helpers for the C-01 harness (run.py).

Why this exists: before C-01 the evaluation scripts seeded only the subset
generator (experiments/eval_phase1.py:109-112); PGD random start and patch
placement used the unseeded global RNG (REVISION_PLAN.md §1.4). Every harness
attack call is now preceded by torch.manual_seed(attack_seed(seed, batch_idx)).
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy (if installed) and torch (CPU + all CUDA devices).

    Args:
        seed: Integer seed.
    """
    seed = int(seed)
    random.seed(seed)
    # Only affects subprocesses started after this point (hash randomisation of
    # the current interpreter is fixed at start-up); recorded for completeness.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)  # also seeds every CUDA device's generator
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_eval_determinism(tf32: bool) -> dict:
    """Configure cuDNN / TF32 flags for evaluation and return the applied state.

    - cudnn.deterministic = True, cudnn.benchmark = False (DeiT-S has one Conv2d,
      the patch embedding, so cuDNN algorithm choice does matter).
    - TF32 is set EXPLICITLY for both cuBLAS matmuls and cuDNN convolutions, so
      the run log records exactly what was used instead of relying on PyTorch's
      version-dependent defaults.

    We deliberately do NOT call torch.use_deterministic_algorithms(True): the
    bitsandbytes INT8 (Linear8bitLt) / NF4 (Linear4bit) kernels are custom CUDA
    ops that may have no deterministic implementation registered, in which case
    PyTorch raises at the first INT8/INT4 forward. Determinism of the quantised
    paths is instead checked empirically (C-01 "done when": the same cell run
    twice yields identical JSON).

    Args:
        tf32: Allow TF32 tensor-core math for fp32 matmuls and convolutions.

    Returns:
        Dict of the flags as now set (for the run log).
    """
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    return {
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "use_deterministic_algorithms": False,
    }


def attack_seed(base: int, batch_idx: int) -> int:
    """Seed used immediately before the attack call on batch `batch_idx`.

    Kept identical to the Q4 sweep (`torch.manual_seed(seed + b)`,
    scratchpad/checks/pgd_epoch_sweep.py:92-94) so harness numbers are
    comparable with the A-13 sweep. Note this means (seed=0, batch=1) and
    (seed=1, batch=0) share a stream; multi-seed analyses should therefore use
    seeds that differ by more than the number of batches (e.g. 0, 1000, 2000).

    Args:
        base: Run-level seed.
        batch_idx: 0-based batch index within the evaluation loader.

    Returns:
        Integer seed.
    """
    return int(base) + int(batch_idx)
