"""CPU-only tests for the C-01 harness helpers (no GPU, no model, no dataset).

    pytest -q tests/test_runlog.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.runlog import (  # noqa: E402
    RunLog, completed_keys, config_sha256, load_completed_results,
    summarize_flags, to_jsonable, two_proportion_check, wilson_interval,
)


# ── Wilson interval ───────────────────────────────────────────────────────────

def test_wilson_textbook_50_of_100():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.40383, abs=1e-4)
    assert hi == pytest.approx(0.59617, abs=1e-4)


def test_wilson_zero_successes():
    lo, hi = wilson_interval(0, 600)
    assert lo == 0.0
    # z^2 / (n + z^2) for k = 0
    assert hi == pytest.approx(1.96 ** 2 / (600 + 1.96 ** 2), rel=1e-9)


def test_wilson_symmetry_and_bounds():
    for k, n in [(0, 10), (3, 10), (7, 31), (31, 31), (999, 1000)]:
        lo, hi = wilson_interval(k, n)
        lo2, hi2 = wilson_interval(n - k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0
        assert lo == pytest.approx(1 - hi2, abs=1e-12)
        assert hi == pytest.approx(1 - lo2, abs=1e-12)


def test_wilson_empty():
    assert wilson_interval(0, 0) == (None, None)


def test_summarize_flags():
    s = summarize_flags([1, 0, 1, 1, True, False])
    assert s["n"] == 6 and s["correct"] == 4
    assert s["accuracy"] == pytest.approx(4 / 6)
    assert s["ci95_wilson"] == list(wilson_interval(4, 6))


def test_two_proportion_check():
    assert two_proportion_check(0.0, 1000, 0.0, 10000)["verdict"] == "PASS"
    # 0.850 vs 0.8566: SE ~ sqrt(.1275/1000 + .1228/10000) ~ 0.01183 -> threshold ~ 0.0232
    r = two_proportion_check(0.850, 1000, 0.8566, 10000)
    assert r["verdict"] == "PASS"
    assert r["threshold"] == pytest.approx(1.96 * math.sqrt(0.85 * 0.15 / 1000 + 0.8566 * 0.1434 / 10000))
    assert two_proportion_check(0.70, 1000, 0.8566, 10000)["verdict"] == "FAIL"


# ── stratified_indices ────────────────────────────────────────────────────────

def _torch():
    return pytest.importorskip("torch")


def _sweep_reference(samples, per_class, seed):
    """Verbatim algorithm of scratchpad/checks/pgd_epoch_sweep.py:46-56."""
    torch = _torch()
    g = torch.Generator().manual_seed(seed)
    by_class: dict = {}
    for i, (_, y) in enumerate(samples):
        by_class.setdefault(y, []).append(i)
    idx = []
    for y in sorted(by_class):
        members = by_class[y]
        pick = torch.randperm(len(members), generator=g)[:per_class].tolist()
        idx.extend(members[j] for j in pick)
    return idx


def test_stratified_matches_sweep_and_is_deterministic():
    _torch()
    from utils.eval_core import stratified_indices

    targets = [(i * 7) % 5 for i in range(100)]  # 5 classes x 20, interleaved
    samples = [(f"img{i}.png", y) for i, y in enumerate(targets)]
    a = stratified_indices(targets, 3, seed=0)
    assert a == _sweep_reference(samples, 3, 0)
    assert a == stratified_indices(targets, 3, seed=0)
    assert a != stratified_indices(targets, 3, seed=1)


def test_stratified_per_class_counts_and_order():
    _torch()
    from utils.eval_core import stratified_indices

    targets = [i % 4 for i in range(40)] + [9, 9]  # classes 0..3 x 10, class 9 x 2
    idx = stratified_indices(targets, 5, seed=0)
    labels = [targets[i] for i in idx]
    assert labels == sorted(labels)  # grouped by sorted class
    counts = {c: labels.count(c) for c in set(labels)}
    assert counts == {0: 5, 1: 5, 2: 5, 3: 5, 9: 2}  # short class contributes all members
    assert len(set(idx)) == len(idx)


def test_stratified_all():
    _torch()
    from utils.eval_core import stratified_indices

    assert stratified_indices([0, 1, 1], "all", 0) == [0, 1, 2]
    assert stratified_indices([0, 1, 1], None, 0) == [0, 1, 2]


# ── config hashing / JSON ─────────────────────────────────────────────────────

def test_config_hash_stable_and_sensitive():
    a = {"exp": {"type": "eval", "precisions": ["fp32", "int8"]}, "cfg": {"pgd": {"eps": 0.03137, "steps": 20}}}
    b = {"cfg": {"pgd": {"steps": 20, "eps": 0.03137}}, "exp": {"precisions": ["fp32", "int8"], "type": "eval"}}
    assert config_sha256(a) == config_sha256(b)  # key order irrelevant
    c = json.loads(json.dumps(a))
    c["cfg"]["pgd"]["steps"] = 10
    assert config_sha256(a) != config_sha256(c)
    assert config_sha256((1, 2)) == config_sha256([1, 2])


def test_to_jsonable_tensors_and_nan():
    torch = _torch()
    out = to_jsonable({"t": torch.tensor([1, 0, 1], dtype=torch.uint8), "s": torch.tensor(0.5),
                       "nan": float("nan"), "p": Path("a") / "b", "dt": torch.float16})
    assert out == {"t": [1, 0, 1], "s": 0.5, "nan": None, "p": "a/b", "dt": "torch.float16"}
    json.dumps(out, allow_nan=False)


# ── RunLog atomic write + resume listing ──────────────────────────────────────

def _entry(prec, attack, ckpt="none", **kw):
    e = {"status": "ok", "precision": prec, "attack": attack, "checkpoint": ckpt,
         "subset_sha": "abc", "n": 10, "correct": 5, "accuracy": 0.5}
    e.update(kw)
    return e


def test_atomic_write_and_completed_keys(tmp_path):
    torch = _torch()
    log = RunLog(tmp_path, "tiny-imagenet", "t", "eval", {"x": 1}, seed=0, tf32=False, collect_env=False)
    log.add_result(_entry("fp32", "clean"))
    log.add_result(_entry("int8", "pgd20"))
    log.add_result(_entry("int4", "fgsm", status="failed"))
    log.set_per_image("fp32|none", {"dataset_indices": [3, 1], "labels": [0, 1],
                                    "correct": {"clean": torch.tensor([1, 0])}})
    p = log.save("complete")

    assert p == tmp_path / "tiny-imagenet" / "runs" / f"{log.run_id}.json"
    assert p.is_file()
    assert not list(p.parent.glob("*.tmp"))  # temp file replaced, none left behind
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["status"] == "complete"
    assert data["config_sha256"] == config_sha256({"x": 1})
    assert data["per_image"]["fp32|none"]["correct"]["clean"] == [1, 0]
    assert not log.traceback_path.exists()

    keys = completed_keys(p.parent)
    assert keys == {("fp32", "clean", "none"), ("int8", "pgd20", "none")}  # failed entry excluded
    assert completed_keys(p.parent, where={"subset_sha": "other"}) == set()
    assert completed_keys(tmp_path / "missing") == set()

    # A later run overrides; a corrupt file is skipped.
    log2 = RunLog(tmp_path, "tiny-imagenet", "t", "eval", {"x": 1}, seed=0, tf32=False,
                  run_id="99991231T000000Z-ffffffff", collect_env=False)
    log2.add_result(_entry("fp32", "clean", accuracy=0.6))
    log2.save()
    (p.parent / "zzz_corrupt.json").write_text("{not json", encoding="utf-8")
    res = load_completed_results(p.parent)
    assert res[("fp32", "clean", "none")]["entry"]["accuracy"] == 0.6
    assert res[("fp32", "clean", "none")]["run_id"] == "99991231T000000Z-ffffffff"


def test_step_records_traceback(tmp_path):
    log = RunLog(tmp_path, "imagenette", "t", "smoke", {}, seed=0, tf32=False, collect_env=False)
    with log.step("int8/fgsm") as st:
        raise RuntimeError("boom")
    assert st.ok is False
    assert log.errors[0]["step"] == "int8/fgsm"
    assert "RuntimeError: boom" in log.errors[0]["traceback"]
    assert log.traceback_path.is_file()
    assert "boom" in log.traceback_path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        with log.step("reraise", reraise=True):
            raise ValueError("x")


def test_warning_hook_counts_and_suppresses(tmp_path, capsys):
    import warnings

    log = RunLog(tmp_path, "imagenette", "t", "smoke", {}, seed=0, tf32=False, collect_env=False)
    log.install_warning_hook()
    try:
        for _ in range(3):
            warnings.warn("MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization")
    finally:
        log.restore_warning_hook()
    recs = [v for k, v in log.data["warnings"].items() if "MatMul8bitLt" in k]
    assert len(recs) == 1 and recs[0]["suppressed"] and recs[0]["count"] >= 1
