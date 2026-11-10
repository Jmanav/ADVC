"""
utils/exp_config.py

Config resolution for run.py: activate a dataset without editing base.yaml,
then apply experiment overrides.

`config_for_dataset` uses exactly the rule of
scripts/build_tensor_cache.py:config_for_dataset (and therefore of
models/loader.py:load_config, lines 96-102): take a FRESH copy of the raw
YAML, set dataset.name, and merge that sub-block's keys up into `dataset`.

Why a fresh copy matters: load_config already merged the file's active
sub-block (tiny-imagenet) into `dataset`. Re-merging imagenette over that
result would leave tiny-only keys behind, e.g. `finetuned_base` and `wnids`.
A stray `finetuned_base` would make load_model() apply the TinyImageNet
fine-tuned weights to an Imagenette run, and the tensor cache's metadata
check would fail.
"""

from __future__ import annotations

import copy
import re
from typing import Any

import yaml

_NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def config_for_dataset(config_path: str, name: str | None) -> dict:
    """Return base config with dataset `name` active (file is never modified).

    Args:
        config_path: Path to configs/base.yaml.
        name: Dataset sub-block to activate; None keeps the file's active one.

    Returns:
        Flattened config dict, identical to load_config() output had the file
        named `name` as active.
    """
    from models.loader import load_config

    cfg = load_config(config_path)
    if name is None or cfg["dataset"]["name"] == name:
        return cfg
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ds = raw.get("dataset", {})
    if not isinstance(ds.get(name), dict):
        raise ValueError(f"Dataset {name!r} has no sub-block under `dataset:` in {config_path}.")
    # Self-check: this merge reproduces load_config for the file's active dataset.
    check = copy.deepcopy(raw)
    act = check["dataset"].get("name")
    if act and isinstance(check["dataset"].get(act), dict):
        for k, v in check["dataset"][act].items():
            check["dataset"][k] = v
    assert check == cfg, "merge rule drifted from models.loader.load_config"
    ds["name"] = name
    for k, v in ds[name].items():  # same rule as load_config
        ds[k] = v
    return raw


def deep_merge(dst: dict, src: dict) -> dict:
    """Recursively merge `src` into `dst` in place (src wins); returns dst."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def parse_value(text: str) -> Any:
    """Parse a --set value with YAML typing, plus '1e-5'-style floats.

    PyYAML (YAML 1.1) reads '1e-5' as a string, so numeric-looking strings are
    converted to float explicitly.
    """
    val = yaml.safe_load(text)
    if isinstance(val, str) and _NUMERIC.match(val.strip()):
        return float(val)
    return val


def set_dotted(d: dict, dotted: str, value: Any) -> None:
    """Set d['a']['b']['c'] = value for dotted='a.b.c', creating dicts as needed."""
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def apply_sets(d: dict, items: list[str] | None) -> list[tuple[str, Any]]:
    """Apply ['key.sub=value', ...] to `d`; returns the parsed pairs."""
    applied = []
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--set expects key.sub=value, got {it!r}")
        k, v = it.split("=", 1)
        val = parse_value(v)
        set_dotted(d, k.strip(), val)
        applied.append((k.strip(), val))
    return applied
