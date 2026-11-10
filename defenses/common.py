"""
defenses/common.py

Shared helpers for the AT and AT+KD defenses.

fgsm_rs() — FGSM with random start (Wong et al., 2020, "Fast is Better than
Free: Revisiting Adversarial Training", ICLR 2020, arXiv:2001.03994):

    x0 = clip_[0,1]( x + U(-eps, eps) )
    x1 = clip_[0,1]( Proj_{B_inf(x, eps)}( x0 + alpha * sign(grad_x0 CE(f(x0), y)) ) )

All arithmetic is in [0, 1] PIXEL space; the model is always fed
ImageNet-normalised tensors ((px - mean) / std), exactly as the torchattacks
FGSM call with set_normalization_used() did before this revision.

This module depends on torch only (no timm / bitsandbytes / models.loader), so
it can be unit-tested on CPU (tests/test_fgsm_rs.py).
"""

from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn

_MeanStd = Union[Sequence[float], torch.Tensor]


def _as_channel_tensor(v: _MeanStd, like: torch.Tensor) -> torch.Tensor:
    """Return mean/std as a (1, C, 1, 1) tensor on like's device and dtype.

    Args:
        v:    3-element list/tuple or tensor (any shape with C elements).
        like: Reference tensor (N, C, H, W) providing device, dtype and C.

    Returns:
        Tensor of shape (1, C, 1, 1).
    """
    t = torch.as_tensor(v, dtype=like.dtype, device=like.device)
    return t.reshape(1, like.size(1), 1, 1)


def fgsm_rs(
    model: nn.Module,
    images_norm: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    alpha: float,
    mean: _MeanStd,
    std: _MeanStd,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    project: bool = True,
) -> torch.Tensor:
    """Generate FGSM-RS adversarial examples (pixel-space eps, normalised I/O).

    Steps (pixel space, [0, 1]):
      px = clamp(images_norm * std + mean, 0, 1)
      x0 = clamp(px + U(-eps, eps), 0, 1)
      g  = d loss(model((x0 - mean) / std), labels) / d x0
      x1 = x0 + alpha * sign(g)
      if project: x1 = min(max(x1, px - eps), px + eps)     # onto B_inf(px, eps)
      x1 = clamp(x1, 0, 1)
      return (x1 - mean) / std   (detached)

    Model mode: mirrors torchattacks' default (Attack.__call__ /
    _change_model_mode / _recover_model_mode, torchattacks 3.5.1
    attack.py:497-516, 228-245): the model is put in eval() for the gradient
    computation, and train() is called afterwards only if it was in training
    mode on entry.

    Gradients: computed with torch.autograd.grad w.r.t. x0 only, so no .grad is
    accumulated on model parameters (same as torchattacks FGSM.forward,
    attacks/fgsm.py:57-59).

    Legacy (pre-revision, July 2026) recipe: project=False, alpha=eps. That
    reproduces the old "uniform start, then torchattacks FGSM with step eps,
    no projection" path, whose perturbation can reach 2*eps. Remaining
    differences vs the old code: the old path round-tripped x0 through
    normalise -> torchattacks.inverse_normalize before the forward pass
    (float rounding only, ~1e-7), and the old torchattacks wrapper always
    called train() on the underlying model afterwards (see the caller notes).

    Args:
        model:       Classifier; may return a tensor or an object with .logits.
        images_norm: Normalised clean batch, shape (N, C, H, W).
        labels:      Class indices, shape (N,).
        eps:         L-inf radius in [0, 1] pixel space (e.g. 8/255).
        alpha:       FGSM step size in pixel space (Wong et al.: 1.25 * eps).
        mean:        Per-channel normalisation mean (list or tensor, C values).
        std:         Per-channel normalisation std  (list or tensor, C values).
        loss_fn:     loss(logits, labels) -> scalar. Defaults to
                     nn.CrossEntropyLoss() (mean reduction, as torchattacks).
        project:     If True (default, correct FGSM-RS) project x1 onto the
                     eps-ball around the clean pixels. False = legacy recipe.

    Returns:
        Normalised adversarial batch, same shape/dtype/device as images_norm,
        detached from the graph.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    images_norm = images_norm.detach()
    labels = labels.detach()
    mean_t = _as_channel_tensor(mean, images_norm)
    std_t = _as_channel_tensor(std, images_norm)

    # Clean image in pixel space and random start inside the eps-ball.
    px = (images_norm * std_t + mean_t).clamp(0.0, 1.0)
    delta = torch.empty_like(px).uniform_(-eps, eps)
    x0 = (px + delta).clamp(0.0, 1.0).detach()

    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            x0.requires_grad_(True)
            out = model((x0 - mean_t) / std_t)
            logits = out.logits if hasattr(out, "logits") else out
            loss = loss_fn(logits, labels)
            grad = torch.autograd.grad(
                loss, x0, retain_graph=False, create_graph=False
            )[0]
    finally:
        if was_training:
            model.train()

    x1 = x0.detach() + alpha * grad.sign()
    if project:
        x1 = torch.min(torch.max(x1, px - eps), px + eps)
    x1 = x1.clamp(0.0, 1.0)

    return ((x1 - mean_t) / std_t).detach()
