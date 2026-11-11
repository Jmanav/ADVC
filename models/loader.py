"""
models/loader.py

Loads DeiT-S or DeiT-B at a specified compression level (fp32, int8, int4).
All parameters come from configs/base.yaml — never hardcode values here.

Usage:
    from models.loader import load_config, load_model
    cfg = load_config()
    model = load_model(model_name="deit_small", compression="int8", config=cfg)
"""

import os
import torch
import timm
import yaml
from pathlib import Path
from typing import Literal


CompressionLevel = Literal["fp32", "int8", "int4"]
ModelName = Literal["deit_small"]


def resolve_data_path(root: Path, rel_or_abs: str) -> Path:
    """Return Path as-is if absolute, otherwise join with project root."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else root / p


def dataset_results_path(results_file: str, cfg: dict) -> str:
    """Insert the active dataset name into a results CSV path.

    Keeps per-dataset results in separate files so runs on different datasets
    never collide or falsely resume from each other. E.g. with dataset.name
    "tiny-imagenet", "results/phase1_results.csv" becomes
    "results/tiny-imagenet/phase1_results.csv".

    Args:
        results_file: Base results path (e.g. "results/phase1_results.csv").
        cfg: Parsed base.yaml config with the active dataset flattened.

    Returns:
        Dataset-scoped results path as a string.
    """
    name = cfg["dataset"]["name"]
    p = Path(results_file)
    return str(p.parent / name / p.name)


def dataset_scoped_dir(base_dir: str, cfg: dict) -> str:
    """Insert the active dataset name into a directory path.

    Keeps per-dataset checkpoints and figures physically separated so runs on
    different datasets never overwrite each other (checkpoint/figure FILENAMES
    do not encode the dataset, so without this an int8 checkpoint or a figure
    from one dataset would silently clobber the other's).

    The dataset name is inserted as the parent of the final path component:
      "results/checkpoints/at"   -> "results/checkpoints/<name>/at"
      "results/checkpoints/atkd" -> "results/checkpoints/<name>/atkd"
      "results/figures"          -> "results/figures/<name>"

    Args:
        base_dir: Base directory path from config (e.g. "results/checkpoints/at").
        cfg: Parsed base.yaml config with the active dataset flattened.

    Returns:
        Dataset-scoped directory path as a string.
    """
    name = cfg["dataset"]["name"]
    p = Path(base_dir)
    return str(p.parent / name / p.name)


def load_config(config_path: str = "configs/base.yaml") -> dict:
    """
    Load the base YAML config.

    The `dataset` section holds per-dataset sub-blocks (`imagenette`,
    `tiny-imagenet`) keyed under the active `dataset.name`. This function
    flattens the active sub-block up into `dataset` so all downstream reads
    (`cfg["dataset"]["val_dir"]`, `["resize"]`, subset sizes, ...) resolve
    without every call site needing to know which dataset is active.

    Args:
        config_path: Path to base.yaml relative to project root.

    Returns:
        config: Parsed config as a dictionary, with the active dataset
        sub-block merged into `config["dataset"]`.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ds = config.get("dataset", {})
    active = ds.get("name")
    if active and isinstance(ds.get(active), dict):
        # Merge the active sub-block's keys up into `dataset` (shared keys like
        # image_size/mean/std/name stay; per-dataset keys like val_dir override).
        for key, value in ds[active].items():
            ds[key] = value

    return config


def load_model(
    model_name: ModelName,
    compression: CompressionLevel,
    config: dict,
    device: str = "cuda",
    int8_state_dict: dict | None = None,
    int8_strict: bool = True,
    int8_pad_multiple: int = 16,
) -> torch.nn.Module:
    """
    Load a DeiT model at the specified compression level.

    Args:
        model_name:  "deit_small"
        compression: "fp32", "int8", or "int4"
        config:      Parsed base.yaml config dict
        device:      "cuda" or "cpu"
        int8_state_dict: int8 only — a saved Linear8bitLt state_dict (e.g. a
            defended checkpoint) to load BEFORE the first forward. bnb 0.50.2
            refuses it afterwards ("Loading a quantized checkpoint into
            non-quantized Linear8bitLt is not supported"; confirmed on Kaggle
            2026-11-10, REVISION_PLAN A-13). The load result is left on
            `model._advc_int8_load` as {"missing": [...], "unexpected": [...]}.
        int8_strict: strict flag for that load_state_dict.
        int8_pad_multiple: int8 only — pad Linear out_features up to this
            multiple before quantising (0 disables). On a T4 with bnb 0.50.2,
            int8 cuBLASLt raises for batch > 16 when out_features ≡ 8 (mod 16),
            e.g. the 200- and 1000-way heads; padding is bit-exact (R-002).

    Returns:
        model: torch.nn.Module in eval mode, moved to device.
    """
    model_cfg = config["models"][model_name]
    timm_name = model_cfg["timm_name"]
    # Head width: prefer the active dataset's num_classes (e.g. tiny-imagenet=200
    # with a fine-tuned head), falling back to 1000 (frozen ImageNet head).
    num_classes = int(config["dataset"].get("num_classes", 1000))

    # Resolve+load the fine-tuned base state_dict once (None if not configured).
    # int8/int4 need it BEFORE quantisation, so it is passed into those loaders;
    # fp32 applies it after construction.
    finetuned_state = _load_finetuned_state(config, device)

    if compression == "fp32":
        model = _load_fp32(timm_name, device, num_classes)
        if finetuned_state is not None:
            missing, unexpected = model.load_state_dict(finetuned_state, strict=False)
            print(f"[loader] FP32: applied fine-tuned base "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
    elif compression == "int8":
        model = _load_int8(timm_name, device, num_classes, finetuned_state,
                           int8_state_dict=int8_state_dict, int8_strict=int8_strict,
                           pad_multiple=int8_pad_multiple)
        _print_bias_guard_report("INT8", install_fp32_bias_guard(model))
    elif compression == "int4":
        model = _load_int4(timm_name, config, device, num_classes, finetuned_state)
        _print_bias_guard_report("INT4", install_fp32_bias_guard(model))
    else:
        raise ValueError(
            f"Unknown compression level: {compression!r}. "
            "Choose from: fp32, int8, int4"
        )

    model.eval()
    return model


def _resolve_finetuned_path(config: dict) -> Path | None:
    """Resolve the dataset-scoped fine-tuned base checkpoint path, or None.

    `finetuned_base` in config (e.g. "results/checkpoints/base/deit_small_ft.pt")
    is scoped by dataset name to
    "results/checkpoints/<name>/base/deit_small_ft.pt", mirroring
    dataset_scoped_dir. Returns None when the key is absent (imagenette keeps the
    frozen ImageNet head — no fine-tuned base).

    Raises:
        FileNotFoundError: if the key is set but no checkpoint exists on disk.
    """
    ds_cfg = config["dataset"]
    ft_rel = ds_cfg.get("finetuned_base")
    if not ft_rel:
        return None

    p = Path(ft_rel)
    scoped = p.parent.parent / ds_cfg["name"] / p.parent.name / p.name
    project_root = Path(__file__).resolve().parent.parent
    candidates = [Path(os.getcwd()) / scoped, project_root / scoped, Path(scoped)]
    for c in candidates:
        if c.is_file():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"finetuned_base checkpoint not found. Tried:\n  {tried}\n"
        "Run scripts/finetune_backbone.py first to produce it."
    )


def _load_finetuned_state(config: dict, device: str) -> dict | None:
    """Load the fine-tuned base state_dict for the active dataset, or None."""
    path = _resolve_finetuned_path(config)
    if path is None:
        return None
    print(f"[loader] Loading fine-tuned base state_dict from {path}")
    return torch.load(str(path), map_location=device)


def _load_fp32(timm_name: str, device: str, num_classes: int = 1000) -> torch.nn.Module:
    """Load full-precision model via timm.

    num_classes controls the head width — 1000 keeps the pretrained ImageNet head
    (imagenette), a smaller value swaps in a fresh K-way head (tiny-imagenet=200).
    """
    model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    return model


def _load_int8(
    timm_name: str,
    device: str,
    num_classes: int = 1000,
    finetuned_state: dict | None = None,
    int8_state_dict: dict | None = None,
    int8_strict: bool = True,
    pad_multiple: int = 16,
) -> torch.nn.Module:
    """
    Load INT8 quantized model via bitsandbytes Linear8bitLt.

    Requires CUDA sm_70+ (T4 is sm_75 — supported). Linear8bitLt stores weights
    as true 8-bit integers and uses cuBLAS-LT for matrix multiply.

    The weight must be assigned as bnb.nn.Int8Params (not a plain nn.Parameter)
    so bitsandbytes can populate the .CB quantized buffer on the first forward
    pass. We build the layer on CPU, move to CUDA, then run a dummy forward pass
    to trigger quantization before any downstream code touches the weights.

    If `finetuned_state` is given, it is loaded into the FP32 model BEFORE Linear
    replacement, so quantisation is applied to the fine-tuned weights (loading it
    after quantisation would be lost). num_classes sets the head width first.
    """
    import bitsandbytes as bnb

    model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)

    # Load fine-tuned float weights BEFORE quantising, so int8 quantises the
    # fine-tuned model (not the raw pretrained one).
    if finetuned_state is not None:
        missing, unexpected = model.load_state_dict(finetuned_state, strict=False)
        print(f"[loader] INT8: applied fine-tuned base pre-quant "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # Pad Linear layers whose out_features is not a multiple of `pad_multiple`
    # (in DeiT-S only the head: 200 or 1000) BEFORE quantising; a forward hook
    # slices the extra outputs off. See pad_linear_out_features().
    padded = pad_linear_out_features(model, pad_multiple) if pad_multiple else {}
    replace_linear_int8(model)
    install_out_slice_hooks(model, padded)
    model = model.to(device)

    # A saved INT8 state_dict must be loaded after .to(cuda) (weights quantised,
    # weight.SCB set) but BEFORE the first forward, which moves SCB into
    # layer.state and sets weight.SCB=None (bnb 0.50.2 nn/modules.py:1142-1163).
    if int8_state_dict is not None:
        if padded:
            int8_state_dict = pad_int8_state_dict(int8_state_dict, padded)
        res = model.load_state_dict(int8_state_dict, strict=int8_strict)
        model._advc_int8_load = {"missing": list(res.missing_keys),
                                 "unexpected": list(res.unexpected_keys)}
        print(f"[loader] INT8: loaded saved int8 state_dict before first forward "
              f"(missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)})")

    # Trigger .CB buffer population for all Linear8bitLt layers.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        model(dummy)
        # Fail loudly at load time (not mid-run) if a padded width still hits the
        # cuBLASLt error: batch 33 was failing for N=200/1000 on T4 (R-002).
        mods = dict(model.named_modules())
        for name, (true_n, padded_n) in padded.items():
            layer = mods[name]
            out = layer(torch.randn(33, layer.in_features, device=device))
            assert out.shape[-1] == true_n, (name, tuple(out.shape), true_n)
        if padded:
            print(f"[loader] INT8: padded out_features for cuBLASLt (multiple of {pad_multiple}): "
                  + ", ".join(f"{n} {a}->{b}" for n, (a, b) in padded.items())
                  + "; batch-33 self-check OK, outputs sliced back")

    print(f"[loader] INT8 (bitsandbytes Linear8bitLt): quantized Linear weights to int8 on {device}")
    return model


def _load_int4(
    timm_name: str,
    config: dict,
    device: str,
    num_classes: int = 1000,
    finetuned_state: dict | None = None,
) -> torch.nn.Module:
    """
    Load INT4 (NF4) quantized model.

    Same strategy as INT8: load FP32 via timm then replace nn.Linear layers
    with bitsandbytes Linear4bit (NF4) in-place. If `finetuned_state` is given it
    is loaded into the FP32 model BEFORE Linear replacement so NF4 quantises the
    fine-tuned weights. num_classes sets the head width first.
    """
    import bitsandbytes as bnb

    int4_cfg = config["compression"]["int4"]
    compute_dtype = (
        torch.float16
        if int4_cfg["bnb_4bit_compute_dtype"] == "float16"
        else torch.bfloat16
    )
    quant_type = int4_cfg["bnb_4bit_quant_type"]  # "nf4"

    model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)

    # Load fine-tuned float weights BEFORE quantising (see _load_int8).
    if finetuned_state is not None:
        missing, unexpected = model.load_state_dict(finetuned_state, strict=False)
        print(f"[loader] INT4: applied fine-tuned base pre-quant "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    model = model.to(device)

    replace_linear_int4(model, compute_dtype, quant_type, device)

    # Explicitly disable gradients on all int4 weight tensors — Params4bit are
    # integer dtype and cannot hold gradients.
    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)

    print(f"[loader] INT4 (NF4): replaced Linear layers with bitsandbytes Int4 on {device}")
    return model


def replace_linear_int8(module: torch.nn.Module) -> None:
    """Swap every nn.Linear under `module` (in place, recursively) for a bnb
    Linear8bitLt with the exact arguments the INT8 loader uses.

    has_fp16_weights=False, default threshold (0.0). Weights become
    bnb.nn.Int8Params (quantised to int8 when the module is moved to CUDA);
    biases are copied as plain fp32 Parameters with requires_grad=False. The
    caller moves the module to the device afterwards (as `_load_int8` does).
    Factored out of `_load_int8` so tests can build the same layers.
    """
    import bitsandbytes as bnb

    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            new_layer = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,
            )
            # Must use Int8Params so bitsandbytes populates .CB on forward.
            new_layer.weight = bnb.nn.Int8Params(
                child.weight.data.clone(),
                requires_grad=False,
                has_fp16_weights=False,
            )
            if child.bias is not None:
                new_layer.bias = torch.nn.Parameter(
                    child.bias.data.clone(), requires_grad=False
                )
            setattr(module, name, new_layer)
        else:
            replace_linear_int8(child)


# ── INT8 out_features padding (cuBLASLt shape workaround, RESULTS_LOG R-002) ──
#
# On a T4 (sm_75) with bnb 0.50.2, int8 cuBLASLt raises "cublasLt ran into an
# error!" for batch > 16 whenever out_features ≡ 8 (mod 16) — N = 200, 216, 232,
# 1000 failed; every tested multiple of 16 worked. LLM.int8 quantises weights per
# output row and activations per input row, so adding extra output rows and
# slicing them off changes nothing: R-002 measured bit-identical int8 weights,
# per-row scales and logits (max abs diff 0.0). Extra rows are COPIES of the
# first real rows (not zeros) so no row has an all-zero absmax scale.

def pad_linear_out_features(module: torch.nn.Module, multiple: int, prefix: str = "") -> dict:
    """Widen (in place, recursively) every nn.Linear whose out_features is not a
    multiple of `multiple`. Returns {qualified_name: (true_out, padded_out)}.
    Call on the fp32 model BEFORE replace_linear_int8; then call
    install_out_slice_hooks() on the quantised model."""
    padded: dict = {}
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear):
            n = child.out_features
            if n % multiple:
                p = -(-n // multiple) * multiple
                extra = p - n
                if extra > n:
                    raise ValueError(f"{full}: cannot pad {n} -> {p} with copies of real rows")
                new = torch.nn.Linear(child.in_features, p, bias=child.bias is not None,
                                      device=child.weight.device, dtype=child.weight.dtype)
                with torch.no_grad():
                    new.weight.copy_(torch.cat([child.weight, child.weight[:extra]]))
                    if child.bias is not None:
                        new.bias.copy_(torch.cat([child.bias, child.bias[:extra]]))
                setattr(module, name, new)
                padded[full] = (n, p)
        else:
            padded.update(pad_linear_out_features(child, multiple, full))
    return padded


def _slice_out_hook(module: torch.nn.Module, inputs, output):  # noqa: ANN001
    """Forward hook: drop the padding outputs (module-level so it pickles)."""
    n = module._advc_true_out
    return output[..., :n] if torch.is_tensor(output) else output


def install_out_slice_hooks(model: torch.nn.Module, padded: dict) -> None:
    """Register _slice_out_hook on every padded layer (idempotent)."""
    mods = dict(model.named_modules())
    for name, (true_n, _p) in padded.items():
        m = mods[name]
        m._advc_true_out = int(true_n)
        if not getattr(m, "_advc_slice_hook_installed", False):
            m.register_forward_hook(_slice_out_hook)
            m._advc_slice_hook_installed = True


def pad_int8_state_dict(sd: dict, padded: dict) -> dict:
    """Return a copy of `sd` whose padded layers' per-row tensors (weight, bias,
    SCB) are padded the same way pad_linear_out_features pads them. Tensors that
    already have the padded length (checkpoints saved from a padded model) are
    left as they are; any other length raises."""
    out = dict(sd)
    for name, (n, p) in padded.items():
        for key in (f"{name}.weight", f"{name}.bias", f"{name}.SCB", f"{name}.weight.SCB"):
            v = sd.get(key)
            if not torch.is_tensor(v) or v.dim() == 0:
                continue
            if v.shape[0] == n:
                out[key] = torch.cat([v, v[: p - n]])
            elif v.shape[0] != p:
                raise ValueError(f"{key}: expected {n} or {p} rows, got {tuple(v.shape)}")
    return out


def replace_linear_int4(
    module: torch.nn.Module,
    compute_dtype: torch.dtype,
    quant_type: str,
    device: str,
) -> None:
    """Swap every nn.Linear under `module` (in place, recursively) for a bnb
    Linear4bit with the exact arguments the INT4 loader uses.

    Each new layer gets Params4bit weights (quantised when moved to `device`)
    and a plain fp32 bias Parameter with requires_grad=False, and is moved to
    `device` individually (as `_load_int4` always did). Factored out of
    `_load_int4` so tests can build the same layers.
    """
    import bitsandbytes as bnb

    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            new_layer = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype,
                quant_type=quant_type,
            )
            new_layer.weight = bnb.nn.Params4bit(
                child.weight.data.clone(),
                requires_grad=False,
                quant_type=quant_type,
            )
            if child.bias is not None:
                new_layer.bias = torch.nn.Parameter(
                    child.bias.data.clone(), requires_grad=False
                )
            new_layer = new_layer.to(device)
            setattr(module, name, new_layer)
        else:
            replace_linear_int4(child, compute_dtype, quant_type, device)


# ── fp32-bias guard for bitsandbytes layers (REVISION_PLAN C-04, Q12) ─────────
#
# Facts from the bitsandbytes sources (wheels unpacked read-only, nothing run):
#
# * Linear4bit, bnb 0.50.2 (nn/modules.py:626-635): forward first casts the
#   input to compute_dtype (fp16 here), THEN does
#   `if bias.dtype != x.dtype: bias.data = bias.data.to(x.dtype)` -> the STORED
#   bias Parameter is overwritten in place with an fp16 copy on the first
#   forward pass. All later updates by the optimiser then happen in fp16.
# * Linear4bit, bnb 0.41.0 (nn/modules.py:232-248) and 0.43.0
#   (nn/modules.py:405-433): the same in-place cast runs BEFORE the input is
#   cast, i.e. against the caller's fp32 input, so the stored bias stays fp32;
#   bnb then uses a differentiable fp16 copy `self.bias.to(compute_dtype)`.
#   (Caveat: in these versions `compute_type_is_set` starts False
#   (0.41:209, 0.43:374) and set_compute_type() switches compute_dtype to the
#   input dtype for fp32 input (0.41:212-215, 0.43:378-382), so the configured
#   float16 compute dtype was overridden to float32. 0.50.2 respects it
#   (`compute_type_is_set = compute_dtype is not None`, nn/modules.py:569-570).)
# * Linear8bitLt, all three versions (0.41:437-439, 0.43:683-685,
#   0.50.2:1185-1187): `if bias.dtype != x.dtype: bias.data = ...to(x.dtype)`
#   against the module's INPUT dtype. Our inputs are fp32 (no autocast), so the
#   int8 biases stay fp32; the fp16 cast of the activations happens inside
#   MatMul8bitLt (0.50.2 autograd/_functions.py:127-140) and does not touch the
#   bias.
# * The July 2026 checkpoints were trained with all biases float32
#   (REVISION_PLAN A-3, runtime check of 2026-11-09).
#
# Because the 0.50.2 mutation happens INSIDE forward, casting once at load time
# is not enough. The guard below is a pair of forward hooks on every bnb layer:
# the pre-hook hands bnb a temporary, differentiable copy of the fp32 bias in
# the dtype bnb is about to compare against (so bnb's in-place cast becomes a
# no-op), and, for Linear4bit, pre-casts the input to compute_dtype exactly as
# bnb itself would; the post-hook puts the fp32 Parameter back and restores the
# output dtype bnb would have returned. Per the 0.41/0.43/0.50.2 sources the
# tensors reaching the GEMM are bit-identical to the unguarded forward pass
# (on <=0.43 the hooks never act: compute_dtype becomes fp32 = bias dtype; on
# 0.50.2 bnb itself would feed fp16(bias) and fp16(x)); the only difference is
# that the stored/trained bias stays fp32 (the July behaviour) and receives an
# fp32 gradient through the copy. Versions 0.44-0.49 were not audited.
# NOTE: the guard does NOT reproduce the <=0.43 fp32 compute override; on
# 0.50.2 INT4 computes in the configured float16. Hooks are used
# (not a subclass or an instance-level forward override) so the layers stay
# genuine bnb classes and full-model pickles (`torch.save(model)`, INT4
# checkpoints) remain picklable: the hook functions are module-level.

_BIAS_MASTER = "_advc_bias_master"     # fp32 Parameter stashed during forward
_BIAS_OUT_DTYPE = "_advc_out_dtype"    # output dtype to restore (Linear4bit only)
_BIAS_GUARD_FLAG = "_advc_fp32_bias_guard"


def _bnb_linear_classes() -> tuple:
    """Return the bitsandbytes Linear classes present, or () if bnb is absent."""
    try:
        import bitsandbytes as bnb
    except Exception:
        return ()
    classes = []
    for name in ("Linear8bitLt", "Linear4bit"):
        cls = getattr(bnb.nn, name, None)
        if cls is not None:
            classes.append(cls)
    return tuple(classes)


def _iter_bnb_layers(model: torch.nn.Module):
    """Yield (name, module) for every bitsandbytes Linear8bitLt/Linear4bit."""
    classes = _bnb_linear_classes()
    if not classes:
        return
    for name, module in model.named_modules():
        if isinstance(module, classes):
            yield name, module


def _is_linear4bit(module: torch.nn.Module) -> bool:
    """True for bnb Linear4bit (and subclasses such as LinearNF4)."""
    try:
        import bitsandbytes as bnb
    except Exception:
        return False
    return isinstance(module, bnb.nn.Linear4bit)


def _restore_master_bias(module: torch.nn.Module) -> None:
    """Put the stashed fp32 bias Parameter back into module._parameters."""
    master = module.__dict__.get(_BIAS_MASTER)
    if master is not None:
        module._parameters["bias"] = master
        module.__dict__[_BIAS_MASTER] = None


def _bias_target_dtype(module: torch.nn.Module, x: torch.Tensor) -> torch.dtype | None:
    """dtype the bnb forward compares the bias against, or None if bnb would
    not touch the bias / the compute dtype is not yet fixed."""
    if _is_linear4bit(module):
        # Before the first forward on bnb <=0.43, compute_type_is_set is False and
        # bnb may still switch compute_dtype to the input dtype: do not interfere.
        if not getattr(module, "compute_type_is_set", False):
            return None
        return getattr(module, "compute_dtype", None)
    return x.dtype  # Linear8bitLt: bias is cast to the module-input dtype


def _bnb_bias_pre_hook(module: torch.nn.Module, args: tuple):
    """Forward pre-hook: keep the stored bias fp32 (see block comment above)."""
    _restore_master_bias(module)  # self-heal if a previous forward raised
    bias = module._parameters.get("bias")
    if bias is None or not args or not torch.is_tensor(args[0]):
        return None
    x = args[0]
    target = _bias_target_dtype(module, x)
    if target is None or target == bias.dtype:
        return None
    module.__dict__[_BIAS_MASTER] = bias
    # Differentiable copy: grads flow back to the fp32 Parameter as fp32.
    # requires_grad of the copy equals the Parameter's, so bnb picks the same
    # (grad / no-grad) kernel path as it would unguarded.
    module._parameters["bias"] = bias.to(target)
    if _is_linear4bit(module) and x.dtype != target:
        # Linear4bit: cast the input to compute_dtype here, exactly as bnb does,
        # so bnb's own bias check (against x.dtype, before or after its cast
        # depending on version) sees matching dtypes and does nothing.
        module.__dict__[_BIAS_OUT_DTYPE] = x.dtype
        return (x.to(target),) + tuple(args[1:])
    return None


def _bnb_bias_post_hook(module: torch.nn.Module, args: tuple, output):
    """Forward hook: restore the fp32 bias Parameter and the bnb output dtype."""
    _restore_master_bias(module)
    out_dtype = module.__dict__.get(_BIAS_OUT_DTYPE)
    module.__dict__[_BIAS_OUT_DTYPE] = None
    if out_dtype is not None and torch.is_tensor(output) and output.dtype != out_dtype:
        # bnb returns `out.to(inp_dtype)`; inp_dtype was the caller's dtype before
        # the pre-hook cast, so restore exactly that cast.
        return output.to(out_dtype)
    return None


def enforce_fp32_biases(model: torch.nn.Module) -> dict:
    """Cast any non-fp32 bias of bitsandbytes Linear layers back to fp32.

    Keeps Parameter identity and requires_grad (only `.data` is replaced), so
    optimisers holding the Parameter keep working.

    Args:
        model: Any module; non-bnb layers are ignored.

    Returns:
        {"n_bnb_layers": int, "n_cast": int, "dtypes_before": {dtype_str: count}}
    """
    n_layers = n_cast = 0
    dtypes_before: dict = {}
    for _, module in _iter_bnb_layers(model):
        n_layers += 1
        _restore_master_bias(module)
        bias = module._parameters.get("bias")
        if bias is None:
            continue
        key = str(bias.dtype).replace("torch.", "")
        dtypes_before[key] = dtypes_before.get(key, 0) + 1
        if bias.dtype != torch.float32:
            bias.data = bias.data.to(torch.float32)
            n_cast += 1
    return {"n_bnb_layers": n_layers, "n_cast": n_cast, "dtypes_before": dtypes_before}


def install_fp32_bias_guard(model: torch.nn.Module) -> dict:
    """Cast bnb biases to fp32 now and install the forward hooks that keep them
    fp32 across every later forward/backward (idempotent).

    Returns the enforce_fp32_biases() report plus "n_hooked" (layers that got
    hooks in this call; 0 when already guarded).
    """
    report = enforce_fp32_biases(model)
    n_hooked = 0
    for _, module in _iter_bnb_layers(model):
        if module.__dict__.get(_BIAS_GUARD_FLAG):
            continue
        module.register_forward_pre_hook(_bnb_bias_pre_hook)
        try:
            # always_call: restore the Parameter even if forward raises (torch>=2.0).
            module.register_forward_hook(_bnb_bias_post_hook, always_call=True)
        except TypeError:
            module.register_forward_hook(_bnb_bias_post_hook)
        module.__dict__[_BIAS_GUARD_FLAG] = True
        n_hooked += 1
    report["n_hooked"] = n_hooked
    return report


def assert_fp32_biases(model: torch.nn.Module) -> None:
    """Raise AssertionError if any bnb layer's bias is not an fp32 Parameter
    (or its gradient, if present, is not fp32). Use after loading, training
    or a forward/backward pass."""
    bad = []
    for name, module in _iter_bnb_layers(model):
        bias = module._parameters.get("bias")
        if bias is None:
            continue
        if not isinstance(bias, torch.nn.Parameter):
            bad.append(f"{name}.bias is {type(bias).__name__}, not Parameter")
        elif bias.dtype != torch.float32:
            bad.append(f"{name}.bias dtype={bias.dtype}")
        elif bias.grad is not None and bias.grad.dtype != torch.float32:
            bad.append(f"{name}.bias.grad dtype={bias.grad.dtype}")
    if bad:  # explicit raise: must survive `python -O`
        raise AssertionError("non-fp32 bnb biases: " + "; ".join(bad[:10]) + (
            f" (+{len(bad) - 10} more)" if len(bad) > 10 else ""))


def _print_bias_guard_report(tag: str, report: dict) -> None:
    """Print the one-line [loader] summary of install_fp32_bias_guard()."""
    print(f"[loader] {tag}: fp32-bias guard — bnb_layers={report['n_bnb_layers']}, "
          f"cast={report['n_cast']}, dtypes_before={report['dtypes_before']}, "
          f"hooked={report['n_hooked']}")


def _get_hf_name(timm_name: str, config: dict) -> str:
    """Resolve a timm model name to its HuggingFace repo name via config."""
    for _, model_cfg in config["models"].items():
        if model_cfg["timm_name"] == timm_name:
            return model_cfg["hf_name"]
    raise ValueError(f"No HuggingFace name found for timm model: {timm_name!r}")


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Return model parameter size in MB.

    Args:
        model: Any torch.nn.Module.

    Returns:
        Size in megabytes, rounded to 2 decimal places.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(total_bytes / (1024 ** 2), 2)


def print_model_info(
    model: torch.nn.Module,
    model_name: str,
    compression: str,
) -> None:
    """Print a quick summary of the loaded model."""
    size_mb = get_model_size_mb(model)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    device = next(model.parameters()).device
    print(f"[loader] {model_name} @ {compression}")
    print(f"         Params : {num_params:.1f}M")
    print(f"         Size   : {size_mb} MB")
    print(f"         Device : {device}")


# Sanity check — run directly to verify everything loads
if __name__ == "__main__":
    cfg = load_config()
    print("=== Sanity check: DeiT-S at all compression levels ===\n")

    for level in ["fp32", "int8", "int4"]:
        try:
            model = load_model("deit_small", level, cfg)
            print_model_info(model, "deit_small", level)

            dummy = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
            with torch.no_grad():
                out = model(dummy)
            print(f"         Output : {out.logits.shape if hasattr(out, 'logits') else out.shape}\n")

        except Exception as e:
            print(f"[loader] {level} failed: {e}\n")
