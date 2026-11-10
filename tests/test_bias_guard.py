"""
tests/test_bias_guard.py

fp32-bias guard for bitsandbytes layers (REVISION_PLAN C-04 / Q12).

bnb 0.50.2 Linear4bit.forward overwrites the stored bias with an fp16 copy on
the first forward pass (bitsandbytes/nn/modules.py:626-635). models/loader.py
installs forward hooks that keep the stored bias fp32 without changing the
values that reach the GEMM. These tests build the same bnb layers the loader
builds (via the loader's own replace_linear_int8 / replace_linear_int4) on a
tiny nn.Sequential and check:
  * all bnb biases are fp32 before forward, after forward, after backward;
  * bias gradients are fp32;
  * the guarded output is bit-identical to an unguarded copy (numerics unchanged).

Skips cleanly without CUDA or bitsandbytes. Run on a GPU box:
    python -m pytest tests/test_bias_guard.py -q
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")   # models.loader imports timm at module level
pytest.importorskip("yaml")
bnb = pytest.importorskip("bitsandbytes")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.loader import (  # noqa: E402
    assert_fp32_biases,
    enforce_fp32_biases,
    install_fp32_bias_guard,
    replace_linear_int4,
    replace_linear_int8,
)

DEVICE = "cuda"
# Mirrors configs/base.yaml compression.int4 (nf4, float16).
INT4_COMPUTE_DTYPE = torch.float16
INT4_QUANT_TYPE = "nf4"


def _fp32_source(seed: int = 0) -> torch.nn.Sequential:
    """Tiny fp32 MLP; deterministic so two conversions are identical."""
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 64),
    )


def _quantised(compression: str, seed: int = 0) -> torch.nn.Module:
    """Convert the tiny MLP exactly as models/loader.py does for `compression`."""
    model = _fp32_source(seed)
    if compression == "int8":
        replace_linear_int8(model)
        model = model.to(DEVICE)
        with torch.no_grad():  # loader triggers CB population the same way
            model(torch.zeros(1, 64, device=DEVICE))
    else:
        model = model.to(DEVICE)
        replace_linear_int4(model, INT4_COMPUTE_DTYPE, INT4_QUANT_TYPE, DEVICE)
    return model.eval()


def _bnb_biases(model: torch.nn.Module) -> list:
    return [m.bias for m in model.modules()
            if isinstance(m, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)) and m.bias is not None]


@pytest.mark.parametrize("compression", ["int8", "int4"])
def test_biases_stay_fp32_through_forward_and_backward(compression: str) -> None:
    model = _quantised(compression)
    report = install_fp32_bias_guard(model)
    assert report["n_bnb_layers"] == 2
    assert report["n_hooked"] == 2
    biases = _bnb_biases(model)
    ids_before = [id(b) for b in biases]
    assert all(b.dtype == torch.float32 for b in biases)
    assert_fp32_biases(model)

    # Make biases trainable (as the defenses do) so their grad dtype is checked.
    for b in biases:
        b.requires_grad_(True)

    # 3-D input like a ViT token sequence (batch, tokens, dim).
    x = torch.rand(4, 7, 64, device=DEVICE, requires_grad=True)
    out = model(x)
    assert out.dtype == torch.float32
    assert_fp32_biases(model)

    out.float().pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert_fp32_biases(model)
    biases_after = _bnb_biases(model)
    assert [id(b) for b in biases_after] == ids_before, "Parameter identity changed"
    for b in biases_after:
        assert isinstance(b, torch.nn.Parameter)
        assert b.dtype == torch.float32
        assert b.grad is not None and b.grad.dtype == torch.float32

    # Second pass (eval, no grad) must also leave them fp32.
    with torch.no_grad():
        model(torch.rand(2, 7, 64, device=DEVICE))
    assert_fp32_biases(model)

    # Idempotent re-install.
    assert install_fp32_bias_guard(model)["n_hooked"] == 0


@pytest.mark.parametrize("compression", ["int8", "int4"])
def test_guard_does_not_change_forward_numerics(compression: str) -> None:
    guarded = _quantised(compression)
    plain = _quantised(compression)  # same seed -> identical quantised weights
    install_fp32_bias_guard(guarded)
    x = torch.rand(3, 5, 64, device=DEVICE)
    with torch.no_grad():
        y_guarded = guarded(x)
        y_plain = plain(x)  # unguarded: bnb 0.50.2 casts int4 biases to fp16 here
    assert y_guarded.dtype == y_plain.dtype
    assert torch.equal(y_guarded, y_plain)

    # Input-gradient (what the attacks use) must match too.
    xg = x.clone().requires_grad_(True)
    xp = x.clone().requires_grad_(True)
    guarded(xg).sum().backward()
    plain(xp).sum().backward()
    assert torch.equal(xg.grad, xp.grad)


def test_enforce_casts_back_and_keeps_identity() -> None:
    model = _quantised("int4")
    b = _bnb_biases(model)[0]
    b.data = b.data.half()  # simulate the unguarded 0.50.2 state
    b.requires_grad_(True)
    report = enforce_fp32_biases(model)
    assert report["n_cast"] == 1
    assert report["dtypes_before"].get("float16") == 1
    assert _bnb_biases(model)[0] is b
    assert b.dtype == torch.float32 and b.requires_grad


def test_assert_fp32_biases_detects_fp16() -> None:
    model = _quantised("int8")
    _bnb_biases(model)[0].data = _bnb_biases(model)[0].data.half()
    with pytest.raises(AssertionError):
        assert_fp32_biases(model)
