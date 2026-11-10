"""
scripts/env_report.py

Environment report for the reproducibility record (REVISION_PLAN C-01, Q12).
Stdlib-first, CPU-only, no GPU compute, < 1 minute, never fails hard: every
probe is wrapped and prints 'n/a' on failure.

Usage (from the repo root):
    python scripts/env_report.py
    python scripts/env_report.py --json env_report.json
    python scripts/env_report.py --paths /kaggle/input/foo /data/tiny --json out.json

Prints: OS/kernel, Python, library versions (torch, torchvision, timm,
bitsandbytes, torchattacks, numpy, PIL, yaml, transformers, accelerate,
autoattack), torch CUDA/cuDNN and device properties (metadata only), nvidia-smi
(driver, MIG mode), CPU cores, RAM, disk free, internet reachability (3 s HEAD),
pip availability, selected env vars (HF_TOKEN only as set/unset), and the
existence + top-level entry counts of the configured dataset paths.
"""

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

NA = "n/a"
REPO_ROOT = Path(__file__).resolve().parent.parent

LIBS = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("timm", "timm"),
    ("bitsandbytes", "bitsandbytes"),
    ("torchattacks", "torchattacks"),
    ("numpy", "numpy"),
    ("PIL", "PIL"),
    ("yaml", "yaml"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("autoattack", "autoattack"),
]

URLS = ["https://pypi.org", "https://github.com", "https://huggingface.co"]
ENV_VARS = ["ADVC_PLATFORM", "ADVC_ACCOUNT", "CUDA_VISIBLE_DEVICES"]
SECRET_VARS = ["HF_TOKEN"]  # reported only as set/unset, never the value


def _safe(fn, default=NA):
    """Run fn(); return default on any exception."""
    try:
        return fn()
    except Exception:
        return default


def _run(cmd: list, timeout: float = 10.0) -> str:
    """Run a command and return stripped stdout+stderr, or 'n/a'."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "") + (("\n" + r.stderr) if r.stderr and r.returncode else "")
        return out.strip() or NA
    except Exception:
        return NA


def probe_system() -> dict:
    """OS, kernel and Python interpreter."""
    return {
        "platform": _safe(platform.platform),
        "system": _safe(platform.system),
        "release": _safe(platform.release),
        "kernel_version": _safe(platform.version),
        "machine": _safe(platform.machine),
        "python_version": _safe(lambda: sys.version.replace("\n", " ")),
        "python_executable": _safe(lambda: sys.executable),
        "hostname": _safe(platform.node),
        "time_utc": _safe(lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
    }


def probe_libs() -> dict:
    """Installed version of each library (import, then dist metadata fallback)."""
    out = {}
    for label, mod_name in LIBS:
        def _ver(mod_name=mod_name):
            mod = importlib.import_module(mod_name)
            v = getattr(mod, "__version__", None)
            if v is None:
                from importlib import metadata
                dist = {"PIL": "Pillow", "yaml": "PyYAML"}.get(mod_name, mod_name)
                v = metadata.version(dist)
            return str(v)
        out[label] = _safe(_ver)
    return out


def probe_torch() -> dict:
    """torch CUDA/cuDNN build info and device metadata (no kernels launched)."""
    info = {k: NA for k in ("torch_cuda_version", "cudnn_version", "cuda_available",
                            "device_count", "devices")}
    try:
        import torch
    except Exception:
        return info
    info["torch_cuda_version"] = _safe(lambda: str(torch.version.cuda))
    info["cudnn_version"] = _safe(lambda: str(torch.backends.cudnn.version()))
    info["cuda_available"] = _safe(lambda: bool(torch.cuda.is_available()))
    info["device_count"] = _safe(lambda: int(torch.cuda.device_count()))
    devices = []
    try:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": _safe(lambda: p.name),
                "total_memory_gb": _safe(lambda: round(p.total_memory / 1e9, 2)),
                "compute_capability": _safe(lambda: f"{p.major}.{p.minor}"),
                "multi_processor_count": _safe(lambda: p.multi_processor_count),
            })
    except Exception:
        pass
    info["devices"] = devices or NA
    return info


def probe_nvidia_smi() -> dict:
    """nvidia-smi summary, device list (MIG instances show up in -L) and CSV query."""
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "summary": NA, "list": NA, "query": NA}
    return {
        "available": True,
        "summary": _run(["nvidia-smi"]),
        "list": _run(["nvidia-smi", "-L"]),
        "query": _run(["nvidia-smi",
                       "--query-gpu=name,driver_version,memory.total,mig.mode.current",
                       "--format=csv"]),
    }


def probe_cpu_ram() -> dict:
    """CPU core counts and RAM from /proc/meminfo (Linux)."""
    out = {
        "os_cpu_count": _safe(os.cpu_count),
        "sched_getaffinity": _safe(lambda: len(os.sched_getaffinity(0))),
        "ram_total_gb": NA,
        "ram_available_gb": NA,
    }

    def _meminfo():
        vals = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                vals[k.strip()] = int(v.strip().split()[0])  # kB
        return vals
    mi = _safe(_meminfo, default=None)
    if isinstance(mi, dict):
        if "MemTotal" in mi:
            out["ram_total_gb"] = round(mi["MemTotal"] / 1e6, 2)
        if "MemAvailable" in mi:
            out["ram_available_gb"] = round(mi["MemAvailable"] / 1e6, 2)
    return out


def probe_disk() -> dict:
    """Free / total disk for cwd, /kaggle/working, /tmp and $HOME."""
    out = {}
    for label, path in [("cwd", os.getcwd()), ("/kaggle/working", "/kaggle/working"),
                        ("/tmp", "/tmp"), ("HOME", os.path.expanduser("~"))]:
        def _du(path=path):
            u = shutil.disk_usage(path)
            return {"path": path, "free_gb": round(u.free / 1e9, 2),
                    "total_gb": round(u.total / 1e9, 2)}
        out[label] = _safe(_du)
    return out


def probe_internet(timeout: float = 3.0) -> dict:
    """HEAD request to each URL with a short timeout."""
    import urllib.request
    out = {}
    for url in URLS:
        def _head(url=url):
            req = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": "advc-env-report"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return f"ok ({r.status})"
        try:
            out[url] = _head()
        except Exception as exc:
            code = getattr(exc, "code", None)
            # An HTTP error status still proves the host is reachable.
            out[url] = f"reachable (HTTP {code})" if code else f"unreachable ({type(exc).__name__})"
    return out


def probe_pip() -> str:
    """`python -m pip --version` for the running interpreter."""
    return _run([sys.executable, "-m", "pip", "--version"], timeout=20.0)


def probe_env() -> dict:
    """Selected env vars; secrets only as set/unset."""
    out = {k: os.environ.get(k, "unset") for k in ENV_VARS}
    for k in SECRET_VARS:
        out[k] = "set" if os.environ.get(k) else "unset"
    return out


def _config_paths() -> list:
    """Dataset dirs from configs/base.yaml (imagenette / tiny-imagenet)."""
    paths = []
    try:
        import yaml
        with open(REPO_ROOT / "configs" / "base.yaml") as f:
            cfg = yaml.safe_load(f)
        ds = cfg.get("dataset", {}) or {}
        for name in ("imagenette", "tiny-imagenet"):
            sub = ds.get(name, {}) or {}
            for key in ("val_dir", "train_dir"):
                if sub.get(key):
                    paths.append((f"config:{name}.{key}", str(sub[key])))
    except Exception:
        pass
    return paths


def probe_paths(extra: list) -> dict:
    """Existence and top-level entry counts (dirs/files, not recursive)."""
    candidates = _config_paths()
    candidates += [(f"arg:{p}", p) for p in extra]
    candidates.append(("kaggle:/kaggle/input", "/kaggle/input"))
    out = {}
    for label, raw in candidates:
        def _probe(raw=raw):
            p = Path(raw)
            if not p.is_absolute():
                p = REPO_ROOT / p
            if not p.exists():
                return {"path": str(p), "exists": False}
            if p.is_file():
                return {"path": str(p), "exists": True, "type": "file"}
            n_dirs = n_files = 0
            names = []
            with os.scandir(p) as it:
                for e in it:
                    if e.is_dir():
                        n_dirs += 1
                    else:
                        n_files += 1
                    if len(names) < 10:
                        names.append(e.name)
            return {"path": str(p), "exists": True, "type": "dir",
                    "top_level_dirs": n_dirs, "top_level_files": n_files,
                    "sample_entries": sorted(names)}
        out[label] = _safe(_probe)
    return out


def _git_sha() -> str:
    """Current git commit of the repo (if git is available)."""
    return _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])


def _print_section(title: str, data) -> None:
    print(f"\n== {title} ==")
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str) and "\n" in v:
                print(f"{k}:")
                for line in v.splitlines():
                    print(f"    {line}")
            else:
                print(f"{k}: {v}")
    else:
        print(data)


def main() -> int:
    """Collect all probes, print them, optionally write JSON. Always exits 0."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default=None, help="write the report to this JSON file")
    ap.add_argument("--paths", nargs="*", default=[], help="extra dataset paths to check")
    args = ap.parse_args()

    t0 = time.time()
    report = {}
    for key, fn in [
        ("system", probe_system),
        ("git_sha", _git_sha),
        ("libraries", probe_libs),
        ("torch_cuda", probe_torch),
        ("nvidia_smi", probe_nvidia_smi),
        ("cpu_ram", probe_cpu_ram),
        ("disk", probe_disk),
        ("internet", probe_internet),
        ("pip", probe_pip),
        ("env", probe_env),
        ("paths", lambda: probe_paths(args.paths)),
    ]:
        report[key] = _safe(fn)
        _print_section(key, report[key])
    report["elapsed_s"] = round(time.time() - t0, 2)
    print(f"\n[env_report] done in {report['elapsed_s']} s")

    if args.json:
        try:
            with open(args.json, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"[env_report] wrote {args.json}")
        except Exception as exc:
            print(f"[env_report] could not write {args.json}: {exc!r}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # last-resort: never fail hard
        print(f"[env_report] unexpected error: {exc!r}")
        sys.exit(0)
