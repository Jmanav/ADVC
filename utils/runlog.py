"""
utils/runlog.py

Per-run JSON log for the C-01 harness (run.py).

One JSON file per run at  <out_root>/<dataset>/runs/<run_id>.json , rewritten
atomically (temp file in the same directory + os.replace) after every
completed unit of work, so a crash or a Kaggle session kill leaves the last
consistent state on disk. When any error is recorded, a human-readable
<run_id>.traceback.txt is written next to it.

Top-level JSON keys (schema_version 1):
    schema_version, run_id, status, created_utc, updated_utc,
    exp {name, type}, dataset, seed, tf32, determinism,
    config (fully resolved: {"exp": ..., "cfg": ...}), config_sha256,
    cli, git, versions, hardware, env_tags,
    timings      [ {step, seconds, ok, t_utc} ],
    warnings     { "<Category>: <message[:200]>": {count, first_location, suppressed} },
    results      [ result entries; see run.py:_result_entry ],
    per_image    { "<precision>|<checkpoint>[|tag]": {dataset_indices, labels, correct{attack: [0/1]}} },
    sections     { free-form: "smoke", "calibration", "repro_check", "data", ... },
    errors       [ {step, type, message, traceback, t_utc} ]

Pure-Python (torch is imported lazily) so the statistics helpers and the
writer are unit-testable on CPU without a GPU stack (tests/test_runlog.py).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1

# Warnings matched here are counted but not printed after their first sighting.
# bitsandbytes 0.50.2 Linear8bitLt emits this on every INT8 forward because the
# fp32 activations are cast to fp16 (REVISION_PLAN.md A-13 / Q5).
SUPPRESSED_WARNING_PATTERNS: list[str] = [r"MatMul8bitLt: inputs will be cast"]

# In bnb 0.50.2 that message goes through the `logging` module, not `warnings`
# (bitsandbytes/autograd/_functions.py:12 logger, :123-124 logger.warning), and
# bnb installs no handler, so Python's lastResort handler prints it per call.
# Older bnb versions used warnings.warn. We intercept both channels.
SUPPRESSED_LOGGERS: list[str] = ["bitsandbytes.autograd._functions", "bitsandbytes"]


class _CountingLogFilter(logging.Filter):
    """Counts log records matching SUPPRESSED_WARNING_PATTERNS and drops them
    (prints the first one once). Non-matching records pass through untouched."""

    def __init__(self, counts: dict) -> None:
        super().__init__()
        self.counts = counts

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            text = record.getMessage()
        except Exception:
            return True
        if not any(re.match(p, text) for p in SUPPRESSED_WARNING_PATTERNS):
            return True
        key = f"log.{record.levelname}: {text[:200]}"
        rec = self.counts.setdefault(key, {"count": 0, "first_location": f"{record.pathname}:{record.lineno}",
                                           "suppressed": True, "channel": "logging"})
        rec["count"] += 1
        if rec["count"] == 1:
            print(f"[runlog] counted+suppressed log warning: {text[:160]}", flush=True)
        return False


# ── JSON helpers ──────────────────────────────────────────────────────────────

def to_jsonable(obj: Any) -> Any:
    """Recursively convert tensors / numpy / Path / tuples into JSON-safe types.

    Non-finite floats become None (strict JSON has no NaN/Inf).
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [to_jsonable(v) for v in sorted(obj, key=repr)]
    if isinstance(obj, Path):
        return obj.as_posix()
    # torch.Tensor / numpy arrays and scalars (duck-typed: no hard import).
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        obj = obj.detach().cpu()
    if hasattr(obj, "tolist"):
        return to_jsonable(obj.tolist())
    if hasattr(obj, "item"):
        try:
            return to_jsonable(obj.item())
        except Exception:
            pass
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON dump (sorted keys, no whitespace, ASCII) for hashing."""
    return json.dumps(
        to_jsonable(obj), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    )


def config_sha256(obj: Any) -> str:
    """sha256 hex digest of canonical_json(obj)."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically (temp file in same dir + os.replace).

    os.replace is atomic on POSIX and on Windows/NTFS when source and target are
    on the same volume, which a same-directory temp file guarantees.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Statistics ────────────────────────────────────────────────────────────────

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion k/n.

    Returns (None, None) for n <= 0.
    """
    if n <= 0:
        return (None, None)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize_flags(flags: Iterable[int | bool], z: float = 1.96) -> dict:
    """n, correct, accuracy and 95 % Wilson CI for a list of 0/1 correctness flags."""
    flags = [int(bool(f)) for f in flags]
    n = len(flags)
    k = sum(flags)
    lo, hi = wilson_interval(k, n, z)
    return {"n": n, "correct": k, "accuracy": (k / n) if n else None, "ci95_wilson": [lo, hi]}


def two_proportion_check(p1: float, n1: int, p2: float, n2: int, z: float = 1.96) -> dict:
    """Independent-samples two-proportion agreement check (Wald SE of difference).

    PASS iff |p1 - p2| <= z * sqrt(p1(1-p1)/n1 + p2(1-p2)/n2). Both zero -> PASS
    (exact agreement). SE == 0 otherwise -> PASS only if the proportions are equal.
    """
    diff = abs(p1 - p2)
    if p1 == 0 and p2 == 0:
        return {"diff": 0.0, "se": 0.0, "threshold": 0.0, "verdict": "PASS", "rule": "both zero (exact)"}
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    thr = z * se
    if se == 0:
        verdict = "PASS" if diff == 0 else "FAIL"
        rule = "zero variance: exact equality required"
    else:
        verdict = "PASS" if diff <= thr else "FAIL"
        rule = f"|p1-p2| <= {z}*SE"
    return {"diff": diff, "se": se, "threshold": thr, "verdict": verdict, "rule": rule}


def subsample_fpc_check(p_sub: float, n_sub: int, p_pop: float, n_pop: int, z: float = 1.96) -> dict:
    """Informational: is p_sub consistent with an n_sub-subsample of a finite
    population of n_pop items whose proportion is p_pop (finite-population
    correction)? Valid only when the subsample's items are a subset of the
    population's items.
    """
    if n_pop <= 1 or n_sub <= 0:
        return {"verdict": "N/A", "rule": "degenerate sizes"}
    diff = abs(p_sub - p_pop)
    fpc = max(0.0, (n_pop - n_sub) / (n_pop - 1))
    se = math.sqrt(p_pop * (1 - p_pop) / n_sub * fpc)
    thr = z * se
    if se == 0:
        verdict = "PASS" if diff == 0 else "FAIL"
    else:
        verdict = "PASS" if diff <= thr else "FAIL"
    return {"diff": diff, "se": se, "threshold": thr, "verdict": verdict, "rule": f"FPC, |p_sub-p_pop| <= {z}*SE"}


# ── Environment capture ───────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 20) -> str:
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
        timeout=timeout, check=True,
    ).stdout.strip()


def git_info(repo: Path = REPO_ROOT) -> dict:
    """git SHA, branch and dirty flag (tracked changes only). Tolerates no git."""
    info: dict = {"sha": None, "branch": None, "dirty": None, "dirty_files": [], "untracked_files": None, "error": None}
    try:
        info["sha"] = _run(["git", "rev-parse", "HEAD"], repo)
        info["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)
        tracked = _run(["git", "status", "--porcelain", "--untracked-files=no"], repo)
        info["dirty_files"] = [ln[3:] for ln in tracked.splitlines() if ln.strip()]
        info["dirty"] = bool(info["dirty_files"])
        allst = _run(["git", "status", "--porcelain"], repo)
        info["untracked_files"] = sum(1 for ln in allst.splitlines() if ln.startswith("??"))
    except Exception as exc:  # git absent, not a repo, timeout ...
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def library_versions() -> dict:
    """Installed versions of the libraries that determine results."""
    from importlib import metadata

    out: dict = {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
    }
    for dist in ("torch", "torchvision", "timm", "bitsandbytes", "torchattacks", "numpy", "PyYAML"):
        try:
            out[dist] = metadata.version(dist)
        except Exception:
            out[dist] = None
    try:
        import torch

        out["torch_runtime"] = torch.__version__
        out["cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except Exception as exc:
        out["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    return out


def hardware_info() -> dict:
    """GPU name(s)/memory/capability, CPU counts, nvidia-smi summary."""
    hw: dict = {
        "cpu_count": os.cpu_count(),
        # Usable cores can be far below cpu_count in containers / MIG slices.
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": [],
    }
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                hw["gpus"].append({
                    "index": i,
                    "name": p.name,
                    "total_memory_gb": round(p.total_memory / 1e9, 3),
                    "capability": f"{p.major}.{p.minor}",
                    "multi_processor_count": p.multi_processor_count,
                })
    except Exception as exc:
        hw["torch_cuda_error"] = f"{type(exc).__name__}: {exc}"
    try:
        hw["nvidia_smi_L"] = _run(["nvidia-smi", "-L"])
        hw["nvidia_smi_query"] = _run([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader",
        ])
    except Exception as exc:
        hw["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return hw


def env_tags() -> dict:
    """Platform / account tags from the environment (set by the notebook)."""
    return {
        "ADVC_PLATFORM": os.environ.get("ADVC_PLATFORM"),
        "ADVC_ACCOUNT": os.environ.get("ADVC_ACCOUNT"),
        "kaggle_detected": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle")),
    }


# ── Step context ──────────────────────────────────────────────────────────────

class Step:
    """Context manager: times a block, records any exception with full traceback.

    Exceptions are SUPPRESSED (so a failing INT8 step does not stop the next
    precision) unless reraise=True. KeyboardInterrupt / SystemExit are recorded
    and always re-raised. Inspect `.ok` after the block.
    """

    def __init__(self, log: "RunLog", name: str, reraise: bool = False) -> None:
        self.log = log
        self.name = name
        self.reraise = reraise
        self.ok: bool | None = None
        self.seconds: float | None = None
        self.error: str | None = None

    def __enter__(self) -> "Step":
        print(f"[step] {self.name} ...", flush=True)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, et, ev, tb) -> bool:
        self.seconds = time.perf_counter() - self._t0
        if et is None:
            self.ok = True
            self.log.add_timing(self.name, self.seconds, True)
            print(f"[step] {self.name} ok ({self.seconds:.1f}s)", flush=True)
            return False
        self.ok = False
        tb_str = "".join(traceback.format_exception(et, ev, tb))
        self.error = f"{et.__name__}: {ev}"
        self.log.add_timing(self.name, self.seconds, False)
        self.log.record_error(self.name, ev, tb_str)
        print(f"[step] {self.name} FAILED ({self.error})\n{tb_str}", flush=True)
        try:
            self.log.save()
        except Exception as save_exc:  # never mask the original failure
            print(f"[runlog] save after error failed: {save_exc!r}", flush=True)
        if issubclass(et, (KeyboardInterrupt, SystemExit)) or self.reraise:
            return False
        return True


# ── RunLog ────────────────────────────────────────────────────────────────────

class RunLog:
    """Accumulates one run's provenance, timings, results and errors; saves JSON."""

    def __init__(
        self,
        out_root: str | Path,
        dataset: str,
        exp_name: str,
        exp_type: str,
        config: dict,
        seed: int,
        tf32: bool,
        run_id: str | None = None,
        collect_env: bool = True,
    ) -> None:
        self.config_sha256 = config_sha256(config)
        if run_id is None:
            ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            salt = f"{self.config_sha256}{time.time_ns()}{os.getpid()}".encode()
            run_id = f"{ts}-{hashlib.sha256(salt).hexdigest()[:8]}"
        self.run_id = run_id
        self.runs_dir = Path(out_root) / dataset / "runs"
        self.path = self.runs_dir / f"{run_id}.json"
        self.traceback_path = self.runs_dir / f"{run_id}.traceback.txt"
        self._orig_showwarning = None
        self.data: dict = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": "created",
            "created_utc": _utc_now(),
            "updated_utc": None,
            "exp": {"name": exp_name, "type": exp_type},
            "dataset": dataset,
            "seed": int(seed),
            "tf32": bool(tf32),
            "determinism": None,
            "config": to_jsonable(config),
            "config_sha256": self.config_sha256,
            "cli": None,
            "git": git_info() if collect_env else None,
            "versions": library_versions() if collect_env else None,
            "hardware": hardware_info() if collect_env else None,
            "env_tags": env_tags(),
            "timings": [],
            "warnings": {},
            "results": [],
            "per_image": {},
            "sections": {},
            "errors": [],
        }

    # -- recording ------------------------------------------------------------
    @property
    def errors(self) -> list:
        return self.data["errors"]

    def step(self, name: str, reraise: bool = False) -> Step:
        """Return a Step context manager bound to this log."""
        return Step(self, name, reraise)

    def add_timing(self, step: str, seconds: float, ok: bool = True) -> None:
        self.data["timings"].append({"step": step, "seconds": round(float(seconds), 4), "ok": bool(ok), "t_utc": _utc_now()})

    def record_error(self, step: str, exc: BaseException | None, tb_str: str | None = None) -> None:
        if tb_str is None:
            tb_str = traceback.format_exc()
        self.data["errors"].append({
            "step": step,
            "type": type(exc).__name__ if exc is not None else None,
            "message": str(exc) if exc is not None else None,
            "traceback": tb_str,
            "t_utc": _utc_now(),
        })

    def add_result(self, entry: dict) -> None:
        self.data["results"].append(to_jsonable(entry))

    def set_per_image(self, key: str, payload: dict) -> None:
        self.data["per_image"][key] = to_jsonable(payload)

    def set_section(self, name: str, value: Any) -> None:
        self.data["sections"][name] = to_jsonable(value)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = to_jsonable(value)

    # -- warnings -------------------------------------------------------------
    def install_warning_hook(self) -> None:
        """Count every warning that reaches showwarning; silence the patterns in
        SUPPRESSED_WARNING_PATTERNS after counting.

        The suppressed patterns get the Python "once" filter, so each distinct
        message text is surfaced (and counted) once per process, not once per
        call. The count therefore means "distinct occurrences seen", which is
        what the run log needs (did bitsandbytes cast INT8 inputs to fp16?).

        The same patterns are also intercepted on the `logging` channel
        (SUPPRESSED_LOGGERS) where bnb 0.50.2 emits them; those are counted per
        call (key prefix "log.") since logging has no "once" filter.
        """
        self._log_filter = _CountingLogFilter(self.data["warnings"])
        for name in SUPPRESSED_LOGGERS:
            logging.getLogger(name).addFilter(self._log_filter)
        for pat in SUPPRESSED_WARNING_PATTERNS:
            warnings.filterwarnings("once", message=pat)
        self._orig_showwarning = warnings.showwarning
        orig = self._orig_showwarning
        counts = self.data["warnings"]

        def _hook(message, category, filename, lineno, file=None, line=None):  # noqa: ANN001
            text = str(message)
            key = f"{getattr(category, '__name__', str(category))}: {text[:200]}"
            rec = counts.setdefault(key, {"count": 0, "first_location": f"{filename}:{lineno}", "suppressed": False})
            rec["count"] += 1
            if any(re.match(p, text) for p in SUPPRESSED_WARNING_PATTERNS):
                rec["suppressed"] = True
                if rec["count"] == 1:
                    print(f"[runlog] counted+suppressed warning: {text[:160]}", flush=True)
                return
            orig(message, category, filename, lineno, file, line)

        warnings.showwarning = _hook

    def restore_warning_hook(self) -> None:
        if getattr(self, "_log_filter", None) is not None:
            for name in SUPPRESSED_LOGGERS:
                logging.getLogger(name).removeFilter(self._log_filter)
            self._log_filter = None
        if self._orig_showwarning is not None:
            warnings.showwarning = self._orig_showwarning
            self._orig_showwarning = None

    # -- persistence ----------------------------------------------------------
    def save(self, status: str | None = None) -> Path:
        """Atomically write the JSON (and the traceback .txt if errors exist)."""
        if status is not None:
            self.data["status"] = status
        self.data["updated_utc"] = _utc_now()
        text = json.dumps(to_jsonable(self.data), indent=1, sort_keys=False, allow_nan=False)
        atomic_write_text(self.path, text)
        if self.data["errors"]:
            parts = [f"run_id: {self.run_id}\nexp: {self.data['exp']}\n"]
            for e in self.data["errors"]:
                parts.append(f"\n===== step: {e['step']}  ({e['t_utc']}) =====\n{e['traceback']}")
            atomic_write_text(self.traceback_path, "".join(parts))
        return self.path


# ── Resume helpers ────────────────────────────────────────────────────────────

def result_key(entry: dict) -> tuple[str, str, str]:
    """(precision, attack, checkpoint) key of a result entry."""
    return (str(entry["precision"]), str(entry["attack"]), str(entry["checkpoint"]))


def load_completed_results(runs_dir: str | Path, where: dict | None = None) -> dict:
    """Map (precision, attack, checkpoint) -> {"entry", "run_id", "path"} over all
    run JSONs in `runs_dir` whose result entries have status "ok" and match every
    key/value in `where`. Later files (run_id starts with a UTC timestamp, so
    lexical order = time order) override earlier ones. Unreadable files are skipped.
    """
    out: dict = {}
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return out
    for p in sorted(runs_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[runlog] skipping unreadable {p.name}: {exc!r}", flush=True)
            continue
        for entry in data.get("results", []):
            if entry.get("status", "ok") != "ok":
                continue
            if where and any(entry.get(k) != v for k, v in where.items()):
                continue
            try:
                key = result_key(entry)
            except KeyError:
                continue
            out[key] = {"entry": entry, "run_id": data.get("run_id"), "path": str(p)}
    return out


def completed_keys(runs_dir: str | Path, where: dict | None = None) -> set:
    """Set of completed (precision, attack, checkpoint) keys; see load_completed_results."""
    return set(load_completed_results(runs_dir, where))
