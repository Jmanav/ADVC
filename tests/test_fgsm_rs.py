"""
tests/test_fgsm_rs.py

CPU-only unit tests for defenses.common.fgsm_rs (revision C-03).
No downloads, no dataset, no timm / bitsandbytes.

Run from the repo root:
    python -m pytest tests/ -q
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# The repo has no package setup (no setup.py / pyproject); make the repo root
# importable the same way defenses/*.py do.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from defenses.common import fgsm_rs  # noqa: E402

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
EPS = 8 / 255
ALPHA = 1.25 * EPS
TOL = 1e-6


class _TinyNet(nn.Module):
    """Tiny conv classifier with dropout so train/eval mode is observable."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(4 * 16 * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        return self.fc(self.drop(h.flatten(1)))


def _make_batch() -> tuple:
    """Return (model, normalised images, labels, mean_t, std_t)."""
    model = _TinyNet()
    mean_t = torch.tensor(MEAN).view(1, 3, 1, 1)
    std_t = torch.tensor(STD).view(1, 3, 1, 1)
    px = torch.rand(2, 3, 16, 16)
    images_norm = (px - mean_t) / std_t
    labels = torch.tensor([0, 3])
    return model, images_norm, labels, mean_t, std_t


def _denorm(x: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    """Normalised -> [0, 1] pixel space (no clamp, to expose any overflow)."""
    return x * std_t + mean_t


def test_projected_linf_within_eps_and_in_range() -> None:
    """(a) project=True keeps ||x1 - x||_inf <= eps and pixels in [0, 1]."""
    torch.manual_seed(42)
    model, x, y, mean_t, std_t = _make_batch()
    adv = fgsm_rs(model, x, y, eps=EPS, alpha=ALPHA, mean=MEAN, std=STD, project=True)

    adv_px = _denorm(adv, mean_t, std_t)
    clean_px = _denorm(x, mean_t, std_t).clamp(0.0, 1.0)
    linf = (adv_px - clean_px).abs().max().item()
    assert linf <= EPS + TOL, f"L-inf {linf} exceeds eps {EPS}"
    assert adv_px.min().item() >= -TOL
    assert adv_px.max().item() <= 1.0 + TOL


def test_legacy_unprojected_can_exceed_eps() -> None:
    """(b) project=False, alpha=eps reproduces the old bug: L-inf in (eps, 2eps]."""
    torch.manual_seed(42)
    model, x, y, mean_t, std_t = _make_batch()
    adv = fgsm_rs(model, x, y, eps=EPS, alpha=EPS, mean=MEAN, std=STD, project=False)

    adv_px = _denorm(adv, mean_t, std_t)
    clean_px = _denorm(x, mean_t, std_t).clamp(0.0, 1.0)
    linf = (adv_px - clean_px).abs().max().item()
    assert linf <= 2 * EPS + TOL, f"L-inf {linf} exceeds 2*eps"
    # With 1536 random-start draws, some pixel's uniform start and gradient
    # sign agree in direction, so the unprojected step overshoots eps.
    assert linf > EPS + TOL, f"expected legacy path to exceed eps, got {linf}"


@pytest.mark.parametrize("start_training", [True, False])
def test_model_mode_restored(start_training: bool) -> None:
    """(c) model's train/eval mode on exit equals its mode on entry."""
    torch.manual_seed(42)
    model, x, y, _, _ = _make_batch()
    model.train(start_training)
    fgsm_rs(model, x, y, eps=EPS, alpha=ALPHA, mean=MEAN, std=STD)
    assert model.training is start_training
    assert all(m.training is start_training for m in model.modules())


def test_eval_mode_during_generation() -> None:
    """(c') gradient is computed with the model in eval mode (torchattacks default)."""
    torch.manual_seed(42)
    model, x, y, _, _ = _make_batch()
    seen = []
    model.register_forward_pre_hook(lambda m, inp: seen.append(m.training))
    model.train()
    fgsm_rs(model, x, y, eps=EPS, alpha=ALPHA, mean=MEAN, std=STD)
    assert seen == [False]


def test_no_param_grad_accumulated() -> None:
    """(d) parameters' .grad stays None after the call."""
    torch.manual_seed(42)
    model, x, y, _, _ = _make_batch()
    model.train()
    fgsm_rs(model, x, y, eps=EPS, alpha=ALPHA, mean=MEAN, std=STD)
    assert all(p.grad is None for p in model.parameters())


def test_output_dtype_shape_and_detached() -> None:
    """(e) output shape/dtype/device match input; output is detached."""
    torch.manual_seed(42)
    model, x, y, mean_t, std_t = _make_batch()
    # mean/std given as (1,3,1,1) tensors, as the defense files pass them.
    adv = fgsm_rs(model, x, y, eps=EPS, alpha=ALPHA, mean=mean_t, std=std_t)
    assert adv.shape == x.shape
    assert adv.dtype == x.dtype
    assert adv.device == x.device
    assert not adv.requires_grad


def test_hf_style_logits_output_unwrapped() -> None:
    """Models returning an object with .logits (HF bnb INT8/INT4) are supported."""

    class _Out:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits

    class _HFNet(_TinyNet):
        def forward(self, x: torch.Tensor):  # type: ignore[override]
            return _Out(super().forward(x))

    torch.manual_seed(42)
    _, x, y, mean_t, std_t = _make_batch()
    adv = fgsm_rs(_HFNet(), x, y, eps=EPS, alpha=ALPHA, mean=MEAN, std=STD)
    linf = (_denorm(adv, mean_t, std_t) - _denorm(x, mean_t, std_t).clamp(0, 1)).abs().max()
    assert linf.item() <= EPS + TOL
