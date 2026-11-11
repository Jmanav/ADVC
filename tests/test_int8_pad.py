"""Tests for the INT8 out_features padding (models/loader.py, RESULTS_LOG R-002).

CPU tests: pad_linear_out_features / pad_int8_state_dict shapes and copy rule.
GPU tests (skipped without CUDA + bitsandbytes): padded Linear8bitLt vs unpadded
at batch 16 (both work there): forward bit-identical, input gradient within a
tight relative tolerance (the fp16 backward GEMM reduces over a longer N, so
bit-equality is not assumed), and batch 32 now runs.

    python -m pytest -q tests/test_int8_pad.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
pytest.importorskip("timm")   # models.loader imports timm at module level
pytest.importorskip("yaml")

from models.loader import (  # noqa: E402
    install_out_slice_hooks,
    pad_int8_state_dict,
    pad_linear_out_features,
)


def _net(n_out: int = 200) -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(384, 384), torch.nn.Linear(384, n_out))


def test_pad_linear_shapes_and_copies():
    m = _net(200)
    w0, b0 = m[1].weight.detach().clone(), m[1].bias.detach().clone()
    padded = pad_linear_out_features(m, 16)
    assert padded == {"1": (200, 208)}
    assert m[1].out_features == 208 and m[0].out_features == 384
    assert torch.equal(m[1].weight[:200], w0) and torch.equal(m[1].bias[:200], b0)
    assert torch.equal(m[1].weight[200:], w0[:8]) and torch.equal(m[1].bias[200:], b0[:8])


def test_pad_noop_for_multiples():
    m = _net(1024)
    assert pad_linear_out_features(m, 16) == {}


def test_slice_hook_fp32():
    m = _net(1000)
    x = torch.randn(4, 384)
    ref = m(x)
    padded = pad_linear_out_features(m, 16)
    assert padded == {"1": (1000, 1008)}
    install_out_slice_hooks(m, padded)
    install_out_slice_hooks(m, padded)  # idempotent: still one hook
    out = m(x)
    assert out.shape == (4, 1000)
    # fp32 CPU GEMM blocking may differ with N, so allow float rounding only.
    assert torch.allclose(out, ref, rtol=1e-6, atol=1e-6)


def test_pad_state_dict_old_and_new_lengths():
    padded = {"head": (200, 208)}
    w = torch.randint(-127, 127, (200, 384), dtype=torch.int8)
    scb = torch.rand(200)
    b = torch.randn(200)
    out = pad_int8_state_dict({"head.weight": w, "head.SCB": scb, "head.bias": b,
                               "head.weight_format": torch.tensor(0), "blocks.0.x": torch.ones(3)}, padded)
    assert out["head.weight"].shape == (208, 384) and torch.equal(out["head.weight"][200:], w[:8])
    assert out["head.SCB"].shape == (208,) and torch.equal(out["head.SCB"][200:], scb[:8])
    assert out["head.bias"].shape == (208,)
    assert out["head.weight_format"].dim() == 0 and torch.equal(out["blocks.0.x"], torch.ones(3))
    again = pad_int8_state_dict(out, padded)          # already padded -> unchanged
    assert torch.equal(again["head.weight"], out["head.weight"])
    with pytest.raises(ValueError):
        pad_int8_state_dict({"head.bias": torch.randn(150)}, padded)


# ── GPU ──────────────────────────────────────────────────────────────────────

def _gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return pytest.importorskip("bitsandbytes")


def _int8(n_out: int, pad: bool) -> torch.nn.Module:
    from models.loader import replace_linear_int8
    m = _net(n_out)
    padded = pad_linear_out_features(m, 16) if pad else {}
    replace_linear_int8(m)
    install_out_slice_hooks(m, padded)
    return m.cuda()


def test_gpu_padded_equals_unpadded_at_bs16():
    _gpu()
    a, b = _int8(200, pad=False), _int8(200, pad=True)
    torch.manual_seed(1)
    x = torch.randn(16, 384, device="cuda")
    xa = x.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)
    ya, yb = a(xa), b(xb)
    assert yb.shape == ya.shape == (16, 200)
    assert torch.equal(ya, yb)
    ga, = torch.autograd.grad(ya.float().square().sum(), xa)
    gb, = torch.autograd.grad(yb.float().square().sum(), xb)
    rel = ((ga - gb).abs().max() / ga.abs().max().clamp_min(1e-12)).item()
    assert rel <= 1e-3, rel


def test_gpu_padded_runs_at_bs32_and_33():
    _gpu()
    b = _int8(200, pad=True)
    for bs in (32, 33, 64):
        x = torch.randn(bs, 384, device="cuda", requires_grad=True)
        y = b(x)
        g, = torch.autograd.grad(y.float().sum(), x)
        assert y.shape == (bs, 200) and torch.isfinite(g).all()
