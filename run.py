#!/usr/bin/env python
"""
run.py — config-driven entry point of the C-01 reproducible harness.

    python run.py --exp configs/exp/<name>.yaml [--set key.sub=value ...]
                  [--precisions fp32,int8,int4] [--out-root DIR] [--force]

Every run writes ONE JSON to <out-root>/<dataset>/runs/<run_id>.json (see
utils/runlog.py for the schema), plus <run_id>.traceback.txt if any step fails.
Select the GPU with CUDA_VISIBLE_DEVICES. CUDA is required (no CPU fallback).

Experiment types (the `type:` key of the YAML)
----------------------------------------------
smoke        Per precision: load -> forward -> backward-to-input -> FGSM (1 batch)
             -> PGD-20 (1 batch) -> clean/FGSM/PGD correctness. Each step is
             isolated; a failure records the full traceback and the run moves
             on. This is the W1 INT8 attack smoke test (REVISION_PLAN A-13).
               python run.py --exp configs/exp/smoke_tiny.yaml
               python run.py --exp configs/exp/smoke_imagenette.yaml \
                   --set data.eval_cache=/path/imagenette_val.pt
calibrate    Throughput per precision x TF32: forward-only and forward+backward
             to the input, as image-passes/s (§5.0 unit). Then a TF32 validation:
             fp32 PGD-20 on a 200-image stratified subset, TF32 off vs on.
               python run.py --exp configs/exp/calibrate_imagenette.yaml
eval         precisions x checkpoints x attacks on a stratified subset, with
             per-image JSON. Resumable: combos already in runs/*.json for the
             same subset/attack/seed/tf32 are skipped unless --force is given.
               python run.py --exp configs/exp/eval_example.yaml --precisions int8
repro_check  TinyImageNet only. Undefended fp32/int8/int4 on 1,000 stratified
             images (5/class, seed 0): clean/FGSM/PGD-20, compared with the
             July rows of results/tiny-imagenet/phase1_results.csv.
               python run.py --exp configs/exp/repro_check_tiny.yaml

Expected runtimes (ESTIMATES, not measurements)
-----------------------------------------------
These use the §5.0 throughput model of REVISION_PLAN.md. Measured Kaggle T4
PGD rates: FP32 ≈ 133-145, INT8 ≈ 138-150, INT4 ≈ 221-231 passes/s. The H100
slice planning value is R ≈ 200. FGSM + PGD-20 costs ≈ 22 passes per image.
Each model load takes roughly 10-60 s; the first-ever load also downloads the
timm weights.
  smoke        ≈ 1-3 min      3 loads + ≈ 32 x 24 passes per precision
  calibrate    ≈ 3-5 min      ≈ 4 x 2 timing loops of 13 batches, plus
                              2 x 200 x 21 PGD passes at fp32 (≈ 1 min on a T4)
  repro_check  ≈ 8-12 min     1,000 x 22 passes per precision ≈ 160 / 155 / 100 s
                              for fp32 / int8 / int4 on a T4, plus loads
  eval         ≈ n_images x 22 / R seconds per (precision, checkpoint)
               e.g. Imagenette full val (3,925) ≈ 7 min per precision at R = 200
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from utils.exp_config import apply_sets, config_for_dataset, deep_merge  # noqa: E402
from utils.runlog import (  # noqa: E402
    RunLog, load_completed_results, subsample_fpc_check, summarize_flags,
    two_proportion_check, wilson_interval,
)
from utils.seeding import attack_seed, seed_everything, set_eval_determinism  # noqa: E402
from utils.eval_core import (  # noqa: E402
    attack_params, attack_signature, build_attack_fn, evaluate, pixel_linf,
    stratified_indices, subset_signature, wrap_logits,
)

EXP_DEFAULTS: dict = {
    "name": None,
    "type": None,
    "dataset": None,          # imagenette | tiny-imagenet (None = base.yaml's active one)
    "model": "deit_small",
    "seed": 0,
    "tf32": False,
    "precisions": ["fp32", "int8", "int4"],
    "attacks": ["clean", "fgsm", "pgd20"],
    "checkpoints": ["none"],
    "subset": {"per_class": 5, "seed": 0},
    "data": {"eval_cache": None, "batch_size": None, "num_workers": None},
    "cfg_overrides": {},
}


# ── config ────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="configs/exp/<name>.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY.SUB=VALUE",
                    help="override an experiment-YAML key (repeatable); base.yaml keys via cfg_overrides.<...>")
    ap.add_argument("--precisions", default=None, help="comma list overriding the YAML, e.g. int8")
    ap.add_argument("--out-root", default=None, help="results root (default <repo>/results)")
    ap.add_argument("--base-config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    ap.add_argument("--force", action="store_true", help="eval/repro_check: recompute completed combos")
    return ap.parse_args(argv)


def load_exp(args: argparse.Namespace) -> dict:
    """Experiment YAML merged over EXP_DEFAULTS, then --set and --precisions."""
    import yaml

    exp = copy.deepcopy(EXP_DEFAULTS)
    with open(args.exp, "r", encoding="utf-8") as f:
        deep_merge(exp, yaml.safe_load(f) or {})
    exp["_sets"] = [list(p) for p in apply_sets(exp, args.set)]
    if args.precisions:
        exp["precisions"] = [p.strip() for p in args.precisions.split(",") if p.strip()]
    if not exp.get("name"):
        exp["name"] = Path(args.exp).stem
    if exp.get("type") not in DISPATCH:
        raise SystemExit(f"exp type must be one of {sorted(DISPATCH)}, got {exp.get('type')!r}")
    bad = [p for p in exp["precisions"] if p not in ("fp32", "int8", "int4")]
    if bad:
        raise SystemExit(f"unknown precisions {bad}")
    exp["checkpoints"] = [normalize_ckpt(c) for c in exp.get("checkpoints") or ["none"]]
    return exp


def build_cfg(exp: dict, base_config: str) -> dict:
    """Base config with the experiment's dataset active and cfg_overrides applied."""
    cfg = config_for_dataset(base_config, exp.get("dataset"))
    deep_merge(cfg, exp.get("cfg_overrides") or {})
    if exp.get("dataset") and cfg["dataset"]["name"] != exp["dataset"]:
        raise SystemExit(f"dataset activation failed: {cfg['dataset']['name']} != {exp['dataset']}")
    exp["dataset"] = cfg["dataset"]["name"]
    return cfg


def normalize_ckpt(c) -> dict:
    """'none' | {name, path | path_by_precision, format, strict} -> dict."""
    if isinstance(c, str):
        if c != "none":
            raise SystemExit(f"checkpoint {c!r}: give a dict with name + path/path_by_precision")
        return {"name": "none"}
    if "name" not in c:
        raise SystemExit(f"checkpoint {c!r} needs a name")
    return dict(c)


def _resolve_path(p: str) -> Path:
    q = Path(p)
    if q.is_absolute():
        return q
    for base in (Path.cwd(), REPO_ROOT):
        if (base / q).exists():
            return base / q
    return REPO_ROOT / q


# ── data / model ──────────────────────────────────────────────────────────────

def build_eval_dataset(exp: dict, cfg: dict):
    """Eval dataset (tensor cache if data.eval_cache, else ImageFolder), its
    integer targets, and a JSON description."""
    cache = (exp.get("data") or {}).get("eval_cache")
    if cache:
        from utils.tensor_cache import load_eval_cache  # lazy: optional module

        ds = load_eval_cache(str(_resolve_path(cache)), cfg)
        source = {"kind": "tensor_cache", "path": str(cache)}
    else:
        from utils.datasets import build_eval_transform, build_remapped_folder

        ds = build_remapped_folder(cfg["dataset"]["val_dir"], build_eval_transform(cfg), cfg)
        source = {"kind": "image_folder", "root": cfg["dataset"]["val_dir"]}
    targets = getattr(ds, "targets", None)
    if targets is None:
        targets = [y for _, y in ds.samples]
    targets = [int(t) for t in targets]
    source.update(n=len(targets), n_classes=len(set(targets)))
    return ds, targets, source


def make_loader(ds, idx: list[int], exp: dict, cfg: dict, batch_size: int | None = None) -> DataLoader:
    """Ordered (shuffle=False) loader over ds[idx]."""
    data = exp.get("data") or {}
    bs = int(batch_size or data.get("batch_size") or cfg["eval"]["batch_size"])
    nw = data.get("num_workers")
    if nw is None:
        nw = 0 if data.get("eval_cache") else int(cfg["eval"]["num_workers"])
    return DataLoader(Subset(ds, idx), batch_size=bs, shuffle=False, num_workers=int(nw), pin_memory=True)


def fixed_batch(ds, idx: list[int], n: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """First n items of idx (cycled if shorter) stacked into one CPU batch."""
    take = [idx[i % len(idx)] for i in range(n)]
    xs, ys = zip(*(ds[i] for i in take))
    return torch.stack(list(xs)), torch.tensor([int(y) for y in ys], dtype=torch.long), take


def _fill_missing_attrs(loaded: torch.nn.Module, fresh: torch.nn.Module) -> list[str]:
    """Copy simple attributes present on the fresh model's modules but absent on
    an old pickled model (e.g. timm Attention.gate; REVISION_PLAN A-13
    compatibility note). Only None/bool/int/float/str/tuple values are copied."""
    fresh_mods = dict(fresh.named_modules())
    filled = []
    for name, m in loaded.named_modules():
        f = fresh_mods.get(name)
        if f is None or type(f) is not type(m):
            continue
        for k, v in vars(f).items():
            if k.startswith("_") or k in vars(m):
                continue
            if isinstance(v, (type(None), bool, int, float, str, tuple)):
                setattr(m, k, v)
                filled.append(f"{name}.{k}" if name else k)
    return filled


def _ckpt_path(cfg: dict, precision: str, ckpt: dict) -> Path | None:
    """Resolved checkpoint file for (precision, ckpt spec), or None if the spec
    gives no path for this precision. Shared by loading and resume matching."""
    tmpl = (ckpt.get("path_by_precision") or {}).get(precision) or ckpt.get("path")
    if not tmpl:
        return None
    return _resolve_path(tmpl.format(precision=precision, dataset=cfg["dataset"]["name"]))


def load_eval_model(exp: dict, cfg: dict, precision: str, ckpt: dict) -> tuple[torch.nn.Module, dict]:
    """models.loader.load_model (fine-tuned TI base applied there), then an
    optional defended checkpoint. Returns (model in eval mode, info dict)."""
    from models.loader import load_model

    model = load_model(exp["model"], precision, cfg, device="cuda")
    info: dict = {"checkpoint": ckpt["name"]}
    if ckpt["name"] == "none":
        return model.eval(), info
    path = _ckpt_path(cfg, precision, ckpt)
    if path is None:
        raise FileNotFoundError(f"checkpoint {ckpt['name']}: no path for precision {precision}")
    fmt = ckpt.get("format", "auto")
    if fmt == "auto":
        fmt = "full_model" if path.name.endswith("_full_model.pt") else "state_dict"
    info.update(path=str(path), format=fmt)
    if fmt == "full_model":
        try:
            loaded = torch.load(str(path), map_location="cuda", weights_only=False)
        except Exception as exc:
            print(f"[run] cuda map_location failed ({exc!r}); retrying via cpu", flush=True)
            loaded = torch.load(str(path), map_location="cpu", weights_only=False).cuda()
        info["filled_attrs"] = _fill_missing_attrs(loaded, model)
        del model
        if precision == "int4":
            # Pickled INT4 models (e.g. July checkpoints) carry no fp32-bias hooks;
            # load_model() guarded the fresh model we just discarded.
            from models.loader import install_fp32_bias_guard
            install_fp32_bias_guard(loaded)
            info["bias_guard_installed"] = True
        return loaded.eval(), info
    sd = torch.load(str(path), map_location="cuda")
    strict = bool(ckpt.get("strict", True))
    if precision == "int8":
        # bnb 0.50.2 only accepts an INT8 state_dict before the first forward
        # (A-13, confirmed on Kaggle 2026-11-10), so rebuild with it applied.
        del model
        model = load_model(exp["model"], precision, cfg, device="cuda", int8_state_dict=sd, int8_strict=strict)
        info.update(model.__dict__.pop("_advc_int8_load", {}))
        return model.eval(), info
    res = model.load_state_dict(sd, strict=strict)
    info.update(missing=list(res.missing_keys), unexpected=list(res.unexpected_keys))
    return model.eval(), info


def _free() -> None:
    """Collect garbage and release cached CUDA memory. Callers must drop their
    own references to the model first (per-precision work lives in helper
    functions so their locals die on return)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _table(rows: list[dict], cols: list[str]) -> str:
    w = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) if rows else len(c) for c in cols}
    line = "  ".join(c.ljust(w[c]) for c in cols)
    out = [line, "  ".join("-" * w[c] for c in cols)]
    for r in rows:
        out.append("  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))
    return "\n".join(out)


def _fmt(x, nd: int = 4) -> str:
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


# ── smoke ─────────────────────────────────────────────────────────────────────

def run_smoke(exp: dict, cfg: dict, log: RunLog, args: argparse.Namespace) -> int:
    """Per-precision step-isolated smoke test on one batch (W1 INT8 check)."""
    sm = exp.get("smoke") or {}
    bs = int(sm.get("batch_size") or cfg["eval"]["batch_size"])
    seed = int(exp["seed"])
    tol = float(sm.get("linf_tol", 1e-4))
    with log.step("data/build") as st:
        ds, targets, source = build_eval_dataset(exp, cfg)
        idx = stratified_indices(targets, sm.get("per_class", 1), int(sm.get("subset_seed", 0)))
        x_cpu, y_cpu, take = fixed_batch(ds, idx, bs)
        log.set_section("data", {"source": source, "batch_indices": take})
    if not st.ok:
        return 1

    rows, details = [], {}
    for prec in exp["precisions"]:
        row, det = _smoke_one(exp, cfg, log, prec, x_cpu, y_cpu, take, seed, tol)
        rows.append(row)
        details[prec] = det
        log.set_section("smoke", {"rows": rows, "details": details})
        log.save()
        _free()

    log.set_section("smoke", {"rows": rows, "details": details})
    print("\n=== smoke summary ===")
    print(_table(rows, ["precision", "load", "forward", "backward", "fgsm", "pgd20", "clean/fgsm/pgd20 correct"]))
    if log.data["warnings"]:
        print("warnings counted:", {k[:80]: v["count"] for k, v in log.data["warnings"].items()})
    return 0 if not log.errors else 1


def _smoke_one(exp, cfg, log, prec, x_cpu, y_cpu, take, seed, tol) -> tuple[dict, dict]:
    """All smoke steps for one precision; every step isolated by log.step."""
    row: dict = {"precision": prec}
    det: dict = {}
    model = None
    torch.cuda.reset_peak_memory_stats()
    with log.step(f"{prec}/load") as st:
        m, info = load_eval_model(exp, cfg, prec, {"name": "none"})
        model = wrap_logits(m)
        del m
        det["load"] = info
    row["load"] = "ok" if st.ok else "FAIL"
    if not st.ok:
        for c in ("forward", "backward", "fgsm", "pgd20"):
            row[c] = "skip"
        return row, det
    x, y = x_cpu.cuda(), y_cpu.cuda()
    flags: dict = {}

    with log.step(f"{prec}/forward") as st:
        with torch.no_grad():
            out = model(x)
        det["forward"] = {"shape": list(out.shape), "dtype": str(out.dtype),
                          "finite": bool(torch.isfinite(out).all())}
        flags["clean"] = (out.argmax(1) == y).to(torch.uint8).cpu().tolist()
        del out
    row["forward"] = "ok" if st.ok else "FAIL"

    with log.step(f"{prec}/backward_to_input") as st:
        xg = x.detach().clone().requires_grad_(True)
        loss = F.cross_entropy(model(xg).float(), y)
        (g,) = torch.autograd.grad(loss, xg)
        det["backward"] = {"loss": float(loss.item()), "grad_norm": float(g.float().norm().item()),
                           "grad_dtype": str(g.dtype), "finite": bool(torch.isfinite(g).all())}
        del xg, loss, g
    row["backward"] = f"ok |g|={det['backward']['grad_norm']:.3g}" if st.ok else "FAIL"

    eps = float(cfg["fgsm"]["eps"])
    for atk in ("fgsm", "pgd20"):
        with log.step(f"{prec}/{atk}") as st:
            fn = build_attack_fn(model, cfg, atk)
            torch.manual_seed(attack_seed(seed, 0))
            adv = fn(x, y)
            linf = pixel_linf(adv, x, cfg)
            with torch.no_grad():
                flags[atk] = (model(adv).argmax(1) == y).to(torch.uint8).cpu().tolist()
            det[atk] = {"params": attack_params(cfg, atk), "adv_shape": list(adv.shape),
                        "adv_dtype": str(adv.dtype), "linf_pixel": linf,
                        "linf_within_eps": bool(linf <= eps + tol)}
            del adv, fn
        row[atk] = (f"ok linf={det[atk]['linf_pixel']:.4f}" if st.ok else "FAIL")
    det["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    row["clean/fgsm/pgd20 correct"] = "/".join(
        f"{sum(flags[a])}" if a in flags else "-" for a in ("clean", "fgsm", "pgd20")) + f" of {len(take)}"
    log.set_per_image(f"{prec}|none|smoke", {"dataset_indices": take, "labels": y_cpu.tolist(), "correct": flags})
    return row, det


# ── calibrate ─────────────────────────────────────────────────────────────────

def _time_loop(fn, warmup: int, timed: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(timed):
        fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_calibrate(exp: dict, cfg: dict, log: RunLog, args: argparse.Namespace) -> int:
    """Throughput (image-passes/s) per precision x TF32 x batch size, loader
    throughput, and fp32 PGD-20 TF32 off/on agreement on a fixed subset."""
    cal = exp.get("calibrate") or {}
    bss = [int(b) for b in (cal.get("batch_sizes") or [cfg["eval"]["batch_size"]])]
    warm, timed = int(cal.get("warmup_batches", 3)), int(cal.get("timed_batches", 10))
    modes_fp32 = [bool(t) for t in cal.get("tf32_modes", [False, True])]
    modes_q = [bool(t) for t in cal.get("tf32_modes_quantized", [False])]
    seed = int(exp["seed"])
    base_tf32 = bool(exp["tf32"])

    with log.step("data/build") as st:
        ds, targets, source = build_eval_dataset(exp, cfg)
        idx = stratified_indices(targets, cal.get("per_class", 1), int(cal.get("subset_seed", 0)))
        X, Y, take = fixed_batch(ds, idx, max(bss))
        log.set_section("data", {"source": source, "timing_batch_indices": take})
    if not st.ok:
        return 1

    calib: dict = {"definition": "image-pass = one forward+backward(to input) of one image (REVISION_PLAN §5.0); "
                                 "GPU-only (fixed in-memory batch, data loading excluded)",
                   "warmup_batches": warm, "timed_batches": timed, "throughput": [], "loader": None,
                   "tf32_validation": None}

    nlb = int(cal.get("loader_batches", 0))
    if nlb > 0:
        with log.step("loader_throughput") as st:
            perm = torch.randperm(len(targets), generator=torch.Generator().manual_seed(0)).tolist()
            ldr = make_loader(ds, perm[: nlb * bss[0]], exp, cfg, batch_size=bss[0])
            t0, n = time.perf_counter(), 0
            for xb, _ in ldr:
                n += xb.shape[0]
            dt = time.perf_counter() - t0
            calib["loader"] = {"images": n, "seconds": round(dt, 3), "images_per_s": round(n / dt, 1),
                               "num_workers": ldr.num_workers, "batch_size": bss[0], "source": source["kind"]}
            del ldr

    rows = []
    for prec in exp["precisions"]:
        _calibrate_one(exp, cfg, log, prec, cal, calib, rows, X, Y, bss, warm, timed,
                       modes_fp32 if prec == "fp32" else modes_q, ds, targets, seed)
        set_eval_determinism(base_tf32)
        _free()

    set_eval_determinism(base_tf32)
    log.set_section("calibration", calib)
    print("\n=== throughput (GPU-only, fixed batch) ===")
    print(_table(rows, ["precision", "tf32", "bs", "fwd img/s", "fwd+bwd passes/s", "peak GB"]))
    if calib["loader"]:
        print(f"loader: {calib['loader']['images_per_s']} img/s "
              f"({calib['loader']['source']}, num_workers={calib['loader']['num_workers']})")
    v = calib.get("tf32_validation")
    if v and v.get("ok"):
        print("\n=== TF32 validation (fp32) ===")
        print(_table(v["rows"], ["attack", "acc_tf32_off", "acc_tf32_on", "agreement", "off_only", "on_only", "n"]))
        for k, r in v["pgd_rate"].items():
            print(f"PGD-20 effective rate {k}: {r} passes/s (21 passes/image)")
    return 0 if not log.errors else 1


def _calibrate_one(exp, cfg, log, prec, cal, calib, rows, X, Y, bss, warm, timed, modes, ds, targets, seed) -> None:
    """Timing loops (and, for fp32, the TF32 validation) for one precision.
    Appends to calib["throughput"] and rows in place."""
    model = None
    with log.step(f"{prec}/load") as st:
        m, _ = load_eval_model(exp, cfg, prec, {"name": "none"})
        model = wrap_logits(m)
        del m
    if not st.ok:
        rows.append({"precision": prec, "tf32": "-", "bs": "-", "fwd img/s": "FAIL", "fwd+bwd passes/s": "FAIL"})
        return
    for tf32 in modes:
        set_eval_determinism(tf32)
        for bs in bss:
            x, y = X[:bs].cuda(), Y[:bs].cuda()
            ent = {"precision": prec, "tf32": tf32, "batch_size": bs}
            torch.cuda.reset_peak_memory_stats()
            with log.step(f"{prec}/tf32={tf32}/bs={bs}/forward") as st:
                def _fwd() -> None:
                    with torch.no_grad():
                        model(x)
                dt = _time_loop(_fwd, warm, timed)
                ent["forward_images_per_s"] = round(timed * bs / dt, 2)
            ent["forward_ok"] = st.ok
            with log.step(f"{prec}/tf32={tf32}/bs={bs}/fwd_bwd") as st:
                xg = x.detach().clone().requires_grad_(True)

                def _fb() -> None:
                    loss = F.cross_entropy(model(xg).float(), y)
                    torch.autograd.grad(loss, xg)
                dt = _time_loop(_fb, warm, timed)
                ent["fwd_bwd_passes_per_s"] = round(timed * bs / dt, 2)
            ent["fwd_bwd_ok"] = st.ok
            ent["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            calib["throughput"].append(ent)
            rows.append({"precision": prec, "tf32": tf32, "bs": bs,
                         "fwd img/s": _fmt(ent.get("forward_images_per_s"), 1) if ent["forward_ok"] else "FAIL",
                         "fwd+bwd passes/s": _fmt(ent.get("fwd_bwd_passes_per_s"), 1) if ent["fwd_bwd_ok"] else "FAIL",
                         "peak GB": ent["peak_mem_gb"]})
            log.set_section("calibration", calib)
            log.save()

    tv = cal.get("tf32_validation") or {}
    if prec == "fp32" and tv.get("enabled", True):
        calib["tf32_validation"] = _tf32_validation(exp, cfg, log, ds, targets, model, tv, seed)


def _tf32_validation(exp, cfg, log, ds, targets, model, tv: dict, seed: int) -> dict:
    """fp32 clean + PGD-20 on a fixed stratified subset with TF32 off vs on."""
    attacks = list(tv.get("attacks", ["clean", "pgd20"]))
    idx = stratified_indices(targets, tv.get("per_class", 1), int(tv.get("subset_seed", 0)))
    loader = make_loader(ds, idx, exp, cfg)
    res: dict = {"ok": False, "n": len(idx), "subset_sha": subset_signature(cfg["dataset"]["name"], idx, [targets[i] for i in idx])}
    out = {}
    for tf32 in (False, True):
        with log.step(f"tf32_validation/tf32={tf32}") as st:
            set_eval_determinism(tf32)
            out[tf32] = evaluate(model, loader, cfg, attacks, seed, log, indices=idx, tag=f"tf32val/tf32={tf32}")
            log.set_per_image(f"fp32|none|tf32val_tf32={tf32}",
                              {k: out[tf32][k] for k in ("dataset_indices", "labels", "correct")})
        if not st.ok:
            return res
    rows = []
    for a in attacks:
        off, on = out[False]["correct"][a], out[True]["correct"][a]
        n = len(off)
        rows.append({"attack": a, "n": n,
                     "acc_tf32_off": round(sum(off) / n, 4), "acc_tf32_on": round(sum(on) / n, 4),
                     "agreement": round(sum(int(p == q) for p, q in zip(off, on)) / n, 4),
                     "off_only": sum(int(p and not q) for p, q in zip(off, on)),
                     "on_only": sum(int(q and not p) for p, q in zip(off, on)),
                     "ci95_off": wilson_interval(sum(off), n), "ci95_on": wilson_interval(sum(on), n)})
    pgd_rate = {}
    for tf32 in (False, True):
        for a in attacks:
            if a.startswith("pgd"):
                steps = attack_params(cfg, a)["steps"]
                s = out[tf32]["seconds"][a]
                pgd_rate[f"tf32={tf32}"] = round(out[tf32]["n"] * (steps + 1) / s, 1) if s else None
    res.update(ok=True, rows=rows, pgd_rate=pgd_rate, seconds={str(k): v["seconds"] for k, v in out.items()},
               note="within-noise agreement is a judgement on rows[*]; off_only/on_only are the discordant counts")
    return res


# ── eval ──────────────────────────────────────────────────────────────────────

def _result_entry(exp, cfg, prec, ckpt_info, attack, flags, seconds, subset_sha, subset_spec, max_linf) -> dict:
    s = summarize_flags(flags)
    return {
        "status": "ok", "exp_type": exp["type"], "dataset": cfg["dataset"]["name"], "model": exp["model"],
        "precision": prec, "checkpoint": ckpt_info["checkpoint"], "checkpoint_path": ckpt_info.get("path"),
        "attack": attack, "attack_params": attack_params(cfg, attack), "attack_sig": attack_signature(cfg, attack),
        "seed": int(exp["seed"]), "tf32": bool(exp["tf32"]), "subset_sha": subset_sha, "subset_spec": subset_spec,
        "n": s["n"], "correct": s["correct"], "accuracy": s["accuracy"], "ci95_wilson": s["ci95_wilson"],
        "seconds": seconds, "images_per_s": round(s["n"] / seconds, 2) if seconds else None,
        "max_linf_pixel": max_linf,
    }


def do_eval(exp: dict, cfg: dict, log: RunLog, args: argparse.Namespace) -> dict:
    """Evaluate precisions x checkpoints x attacks; returns
    {(precision, attack, checkpoint): {"entry", "run_id", "source"}} including
    results reused from earlier runs."""
    sub = exp.get("subset") or {}
    attacks = list(exp["attacks"])
    for a in attacks:
        attack_params(cfg, a)  # validate names early
    with log.step("data/build", reraise=True):
        ds, targets, source = build_eval_dataset(exp, cfg)
        idx = stratified_indices(targets, sub.get("per_class", 5), int(sub.get("seed", 0)))
        labels = [targets[i] for i in idx]
        sha = subset_signature(cfg["dataset"]["name"], idx, labels)
        loader = make_loader(ds, idx, exp, cfg)
        log.set_section("data", {"source": source, "subset_spec": sub, "subset_sha": sha, "n": len(idx)})
    print(f"[run] subset: {len(idx)} images, sha {sha}", flush=True)

    prior = {} if args.force else load_completed_results(log.runs_dir, where={
        "subset_sha": sha, "seed": int(exp["seed"]), "tf32": bool(exp["tf32"]), "model": exp["model"],
        "dataset": cfg["dataset"]["name"]})
    collected: dict = {}
    for prec in exp["precisions"]:
        for ckpt in exp["checkpoints"]:
            todo = []
            cpath = None if ckpt["name"] == "none" else _ckpt_path(cfg, prec, ckpt)
            cpath = str(cpath) if cpath is not None else None
            for a in attacks:
                p = prior.get((prec, a, ckpt["name"]))
                # Reuse only if the attack AND the resolved checkpoint file match;
                # a YAML that keeps a name but points at a new file must re-run.
                if (p and p["entry"].get("attack_sig") == attack_signature(cfg, a)
                        and p["entry"].get("checkpoint_path") == cpath):
                    collected[(prec, a, ckpt["name"])] = {"entry": p["entry"], "run_id": p["run_id"], "source": "reused"}
                else:
                    todo.append(a)
            if not todo:
                print(f"[run] skip {prec}/{ckpt['name']}: all attacks already in {log.runs_dir}", flush=True)
                continue
            tag = f"{prec}/{ckpt['name']}"
            model = None
            with log.step(f"{tag}/load") as st:
                model, info = load_eval_model(exp, cfg, prec, ckpt)
            if not st.ok:
                continue
            with log.step(f"{tag}/evaluate[{','.join(todo)}]") as st:
                r = evaluate(model, loader, cfg, todo, int(exp["seed"]), log, indices=idx, tag=tag)
            if st.ok:
                for a in todo:
                    e = _result_entry(exp, cfg, prec, info, a, r["correct"][a], r["seconds"][a], sha, sub,
                                      r["max_linf"].get(a))
                    log.add_result(e)
                    collected[(prec, a, ckpt["name"])] = {"entry": e, "run_id": log.run_id, "source": "new"}
                    print(f"[run] {tag} {a:6s} acc={e['accuracy']:.4f} "
                          f"CI95=[{e['ci95_wilson'][0]:.4f},{e['ci95_wilson'][1]:.4f}] n={e['n']} "
                          f"({e['seconds']:.0f}s)", flush=True)
                log.set_per_image(f"{prec}|{ckpt['name']}",
                                  {"dataset_indices": r["dataset_indices"], "labels": r["labels"], "correct": r["correct"]})
                log.save()
            model = None  # drop the reference before freeing CUDA memory
            _free()
    return collected


def run_eval(exp: dict, cfg: dict, log: RunLog, args: argparse.Namespace) -> int:
    """`eval` experiment type."""
    res = do_eval(exp, cfg, log, args)
    rows = [{"precision": k[0], "checkpoint": k[2], "attack": k[1], "n": v["entry"]["n"],
             "acc": _fmt(v["entry"]["accuracy"]),
             "ci95": f"[{_fmt(v['entry']['ci95_wilson'][0])},{_fmt(v['entry']['ci95_wilson'][1])}]",
             "source": v["source"] if v["source"] == "new" else f"reused {v['run_id']}"}
            for k, v in sorted(res.items())]
    print("\n=== eval ===")
    print(_table(rows, ["precision", "checkpoint", "attack", "n", "acc", "ci95", "source"]))
    return 0 if not log.errors else 1


# ── repro_check ───────────────────────────────────────────────────────────────

# July CSV attack name -> harness attack name. Column semantics
# (results/tiny-imagenet/phase1_results.csv, header line 1):
# timestamp,dataset,model,compression,attack,clean_acc,robust_acc,asr,robustness_gap
_JULY_ATTACK = {"fgsm": "fgsm", "pgd": "pgd20"}


def read_july(csv_path: Path) -> tuple[dict, list[str]]:
    """{(compression, 'clean'|'fgsm'|'pgd20'): {p, csv_line, timestamp, rows_seen}}, notes."""
    out: dict = {}
    clean_seen: dict = {}
    notes: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            comp, att = row["compression"], row["attack"]
            clean_seen.setdefault(comp, set()).add(row["clean_acc"])
            if att in _JULY_ATTACK:
                key = (comp, _JULY_ATTACK[att])
                prev = out.get(key)
                out[key] = {"p": float(row["robust_acc"]), "csv_line": line_no, "timestamp": row.get("timestamp"),
                            "rows_seen": (prev["rows_seen"] + 1) if prev else 1}
                if att == "fgsm":  # clean_acc taken from the fgsm row (same 10k loader as fgsm/pgd)
                    prevc = out.get((comp, "clean"))
                    out[(comp, "clean")] = {"p": float(row["clean_acc"]), "csv_line": line_no,
                                            "timestamp": row.get("timestamp"),
                                            "rows_seen": (prevc["rows_seen"] + 1) if prevc else 1}
    for comp, vals in clean_seen.items():
        if len(vals) > 1:
            notes.append(f"{comp}: clean_acc differs across CSV rows {sorted(vals)}; using the fgsm row's value")
    for k, v in out.items():
        if v["rows_seen"] > 1:
            notes.append(f"{k}: {v['rows_seen']} CSV rows; using the LAST (line {v['csv_line']})")
    return out, notes


def run_repro_check(exp: dict, cfg: dict, log: RunLog, args: argparse.Namespace) -> int:
    """TinyImageNet undefended 1k-image check vs the July phase-1 rows."""
    if cfg["dataset"]["name"] != "tiny-imagenet":
        raise SystemExit("repro_check is TinyImageNet only (set dataset: tiny-imagenet)")
    rc = exp.get("repro") or {}
    z = float(rc.get("z", 1.96))
    n2 = int(rc.get("july_n", 10000))
    res = do_eval(exp, cfg, log, args)

    csv_path = _resolve_path(rc.get("july_csv", "results/tiny-imagenet/phase1_results.csv"))
    july, notes = read_july(csv_path)
    assumptions = [
        f"July n2 = {n2}: the CSV has no count column (header: timestamp,dataset,model,compression,attack,"
        "clean_acc,robust_acc,asr,robustness_gap). n2 is inferred from configs/base.yaml val_subset_size "
        "(10000 for tiny-imagenet) and experiments/eval_phase1.py:111 n = min(val_subset_size, len(val)), "
        "with Tiny val = 10,000 images. FGSM/PGD rows use that loader; patch rows used 500 and are not compared.",
        "July PGD steps assumed = 20 (configs/base.yaml pgd.steps); the CSV does not record steps.",
        "July 'pgd' maps to the harness 'pgd20'; July clean_acc is taken from the fgsm row.",
        "The independent-samples test is lenient here: the 1,000 images are a subset of the July 10,000 "
        "(the same val set), so the samples are positively correlated. The informational FPC column treats the "
        "1k set as a subsample of the July population, which is stricter.",
    ]
    rows = []
    for prec in exp["precisions"]:
        for metric in ("clean", "fgsm", "pgd20"):
            r = res.get((prec, metric, "none"))
            j = july.get((prec, metric))
            row = {"precision": prec, "metric": metric}
            if r is None or j is None:
                row.update(verdict="MISSING", why="no new result" if r is None else "no July row")
                rows.append(row)
                continue
            e = r["entry"]
            p1, n1, p2 = float(e["accuracy"]), int(e["n"]), float(j["p"])
            chk = two_proportion_check(p1, n1, p2, n2, z)
            fpc = subsample_fpc_check(p1, n1, p2, n2, z)
            lo, hi = e["ci95_wilson"]
            row.update(p1=round(p1, 4), n1=n1, p2=p2, n2=n2, diff=round(chk["diff"], 4),
                       threshold=round(chk["threshold"], 4), verdict=chk["verdict"], rule=chk["rule"],
                       fpc_verdict=fpc["verdict"], p2_in_wilson95_p1=bool(lo <= p2 <= hi),
                       july_csv_line=j["csv_line"], july_timestamp=j["timestamp"],
                       new_run_id=r["run_id"], source=r["source"])
            rows.append(row)
    overall = "PASS" if rows and all(r["verdict"] == "PASS" for r in rows) else "FAIL"
    int8_note = ("INT8 has no prior independent check: the Q4 sweep (A-13) failed on every INT8 row, so the "
                 "INT8 comparison here is the first test of the July INT8 numbers.")
    log.set_section("repro_check", {"rows": rows, "overall": overall, "z": z, "july_csv": str(csv_path),
                                    "assumptions": assumptions, "notes": notes + [int8_note]})
    print("\n=== repro_check vs July TI phase 1 ===")
    print(_table(rows, ["precision", "metric", "p1", "n1", "p2", "n2", "diff", "threshold", "verdict",
                        "fpc_verdict", "p2_in_wilson95_p1", "july_csv_line"]))
    print(f"\nOVERALL: {overall}")
    for s in assumptions + notes + [int8_note]:
        print(f"- {s}")
    return 0 if overall == "PASS" and not log.errors else 1


DISPATCH = {"smoke": run_smoke, "calibrate": run_calibrate, "eval": run_eval, "repro_check": run_repro_check}


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Parse, resolve config, set seeds/determinism, dispatch, save the run JSON."""
    args = parse_args(argv)
    exp = load_exp(args)
    cfg = build_cfg(exp, args.base_config)
    if exp["type"] == "repro_check":
        # Fixed before the config hash is taken: undefended only.
        exp["checkpoints"] = [{"name": "none"}]
        sub = exp.setdefault("subset", {})
        sub.setdefault("per_class", 5)
        sub.setdefault("seed", 0)
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "results"
    if not out_root.is_absolute():
        out_root = Path.cwd() / out_root

    if not torch.cuda.is_available():
        print("[run] CUDA is not available. This harness never falls back to CPU (CLAUDE.md).", flush=True)
        return 2

    seed_everything(int(exp["seed"]))
    det = set_eval_determinism(bool(exp["tf32"]))
    log = RunLog(out_root, cfg["dataset"]["name"], exp["name"], exp["type"],
                 {"exp": exp, "cfg": cfg}, int(exp["seed"]), bool(exp["tf32"]))
    log.set("determinism", det)
    log.set("cli", {"argv": sys.argv, "exp_file": str(args.exp), "out_root": str(out_root), "force": args.force,
                    "base_config": str(args.base_config)})
    log.install_warning_hook()
    log.save("running")
    print(f"[run] {exp['type']} '{exp['name']}' dataset={cfg['dataset']['name']} run_id={log.run_id}\n"
          f"[run] log: {log.path}", flush=True)
    rc = 1
    try:
        rc = DISPATCH[exp["type"]](exp, cfg, log, args)
    except BaseException as exc:
        import traceback

        log.record_error("run", exc, traceback.format_exc())
        log.save("failed")
        raise
    finally:
        log.restore_warning_hook()
    log.save("complete" if not log.errors else "complete_with_errors")
    print(f"[run] wrote {log.path}" + (f"\n[run] tracebacks: {log.traceback_path}" if log.errors else ""), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
