# Results log (revision 2026–27)

One entry per Kaggle / slice notebook run, recorded from the output the authors pasted.
Each entry gives the git commit, the notebook and its version, and the platform, so every number can be reproduced.
Values are copied verbatim. Interpretation goes under **Reading** and is labelled
CONFIRMED / UNVERIFIED / HYPOTHESIS. Nothing in this file is a paper number until the plan says so.

---

## R-001 · W1 checks · 2026-11-10

- **Notebook:** `notebooks/w1_checks_kaggle.ipynb`. Kaggle version: *not given (ask the authors)*.
  Cells 10–11 (calibrate, repro check) were off.
- **Git:** `89018eb3297ed1c9c5472f3b57e5dc60f127f545` (branch `revision-w1`, clean tree).
- **Platform:**
  - Kaggle, 2× Tesla T4 (sm_75, 15.64 GB), driver 580.178.04, CUDA 13.0, cuDNN 92700.
  - Python 3.13.16. 4 CPUs, 32.87 GB RAM.
- **Libraries:** torch 2.11.0+cu130, torchvision 0.26.0+cu130, timm 1.0.30, bitsandbytes 0.50.2,
  torchattacks 3.5.1, numpy 2.1.3, Pillow 13.0.0, PyYAML 6.0.3. autoattack: n/a.
- **Inputs:** `jainmanav20/advc-results`, `albert1994/tinyimagenet`.
- **Run JSON:** `20261110T111852Z-94083d79` (smoke_tiny).

### Wall times and exit codes (verbatim)

```
prepare_tiny           exit=0  114.8s
env_report             exit=0  23.2s
pytest_cpu             exit=1  7.4s
pytest_bias_guard      exit=0  8.7s
smoke_tiny             exit=1  114.6s
int8_reload_probe      exit=0  9.4s
int4_pickle_probe      exit=0  6.9s
```

### Tests

- `pytest_cpu`: 26 passed, 1 failed.
  - Failing test: `test_wilson_symmetry_and_bounds`, `assert (31 / 31) <= 0.9999999999999999` (tests/test_runlog.py:42).
- `pytest_bias_guard`: 6 passed.

### Smoke (smoke_tiny, 1 batch of 32, seed 0, tf32 false, cudnn deterministic) — timings (s, verbatim)

```
data/build 4.4221 ok | fp32/load 6.6659 ok | fp32/forward 3.4291 ok | fp32/backward_to_input 2.0339 ok
fp32/fgsm 2.5906 ok  | fp32/pgd20 39.6097 ok
int8/load 1.861 ok   | int8/forward 0.0585 FAIL | int8/backward_to_input 0.0363 FAIL | int8/fgsm 0.1432 FAIL | int8/pgd20 0.069 FAIL
int4/load 0.666 ok   | int4/forward 0.0948 ok | (rest truncated in the pasted summary)
```

INT8 error, identical in every INT8 step that was shown:

```
bitsandbytes/backends/cuda/ops.py:164 _int8_linear_matmul_impl
RuntimeError: cublasLt ran into an error!
    shapeA=torch.Size([200, 384]), shapeB=torch.Size([32, 384]), shapeC=(32, 200)
    (lda, ldb, ldc)=(384, 384, 200)
    (m, n, k)=(384, 32, 200)
```

Raised from `timm vision_transformer.py:1300 forward_head` → `self.head(x)`, the 200-way INT8 head.

### INT8 reload probe (`at/at_int8_epoch01.pt`, verbatim)

```
weights_only_default_ok: true
saved_special_keys: {SCB: 49, weight_format: 49}
saved_weight_format_values: {"0": 49}
saved_weight_dtypes: {torch.float32: 26, torch.int8: 49}
fresh_first_layer: {weight_dtype: torch.int8, SCB_is_None: true, state_SCB_is_None: false}
load_state_dict: RAISED
RuntimeError: Loading a quantized checkpoint into non-quantized Linear8bitLt is not supported.
Please call module.cuda() before module.load_state_dict()   (bnb nn/modules.py:1146)
```

### INT4 pickle probe (`at/at_int4_epoch07_full_model.pt`, CPU, verbatim)

```
n_linear4bit: 49, top_type: VisionTransformer
all 49: {bias_dtype: float32, compute_dtype: float16, compute_type_is_set: True,
         quant_state_dtype: float32, quant_type: nf4}
```

### Reading

- **CONFIRMED (A-13 hypothesis 1).** On bnb 0.50.2, a saved INT8 state_dict cannot be loaded after the model's
  first forward. Fix in `models/loader.py`: `load_model(..., int8_state_dict=)` loads it before the dummy forward.
  It is verified by W1b P3.
- **A-13 hypothesis 2 REJECTED.** The saved `weight_format` is 0 in all 49 layers. CONFIRMED that 0 = "row" in bnb 0.50.2? UNVERIFIED.
- **CONFIRMED.** The INT8 forward fails at the 200-way head for batch 32. The other 48 Linear8bitLt layers run
  at batch 32, and the head ran at batch 1 (the loader's dummy forward).
  - HYPOTHESIS: cuBLASLt int8 on sm_75 rejects this output width (200) at this batch. W1b P1 tests this.
  - HYPOTHESIS: this is also the July/Q4 INT8 sweep failure (A-13).
- **The July INT4 model computed in fp16, not fp32.** CONFIRMED for this checkpoint: compute_dtype float16,
  compute_type_is_set True, so bnb's fp32 auto-switch could not apply. The A-13 "July INT4 may have computed in fp32"
  hypothesis is REJECTED for `at_int4_epoch07`. The saved biases are fp32.
- **Throughput (single batch, includes warm-up, not a calibration).**
  - fp32 PGD-20 on 32 images took 39.6 s, about 17 image-passes/s.
  - The earlier repro_check estimate (8–12 min) assumed the §5.0 rate and is about 6× too low.
  - Revised: about 60–70 min (HYPOTHESIS until calibrate_tiny runs).
- **Wilson test failure** is float rounding at k = n. Fixed in `utils/runlog.py` (exact 0/1 bounds at k = 0 and k = n).

---

## R-002 · W1b INT8 probes · 2026-11-11

- **Notebook:** `notebooks/w1b_int8_probe_kaggle.ipynb`. Kaggle version: *not given*.
- **Git:** `d7f03bc` (branch `revision-w1`).
- **Platform:** Kaggle Tesla T4. torch 2.11.0+cu130, bnb 0.50.2.
- **Input:** `jainmanav20/advc-results`.

### P1: `Linear8bitLt(384, N)`, fp32 input, forward + backward-to-input, by batch size (verbatim)

```
N        1    2    4    8   15   16   17   31   32   33   64
200     ok   ok   ok   ok   ok   ok    X    X    X    X    X
208     ok   ok   ok   ok   ok   ok   ok   ok   ok   ok   ok
216     ok   ok   ok   ok   ok   ok    X    X    X    X    X
224     ok   ok   ok   ok   ok   ok   ok   ok   ok   ok   ok
232     ok   ok   ok   ok   ok   ok    X    X    X    X    X
240     ok   ok   ok   ok   ok   ok   ok   ok   ok   ok   ok
256     ok   ok   ok   ok   ok   ok   ok   ok   ok   ok   ok
384     ok   ok   ok   ok   ok   ok   ok   ok   ok   ok   ok
1000    ok   ok   ok   ok   ok   ok    X    X    X    X    X
1024    ok   ok   ok   ok   ok   ok   ok   ok   ok   ok   ok
X = RuntimeError: cublasLt ran into an error!  (cuda context ok after error)
```

### P2: head padded from 200 to P rows (padding rows are copies of real rows), logits sliced to 200 (verbatim)

```
{"P": 208, "CB_rows_equal": true, "SCB_rows_equal": true, "max_abs_200_vs_padded_bs1": 0.0, "max_abs_padded_bs32_vs_bs1": 0.0, "max_abs_padded_bs32_vs_fp32_ref": 0.016421109437942505, "max_abs_200_bs1_vs_fp32_ref": 0.016421109437942505, "argmax_agree_padded_vs_200": 1.0}
{"P": 224, "CB_rows_equal": true, "SCB_rows_equal": true, "max_abs_200_vs_padded_bs1": 0.0, "max_abs_padded_bs32_vs_bs1": 0.0, "max_abs_padded_bs32_vs_fp32_ref": 0.015977390110492706, "max_abs_200_bs1_vs_fp32_ref": 0.015977390110492706, "argmax_agree_padded_vs_200": 1.0}
{"P": 256, "CB_rows_equal": true, "SCB_rows_equal": true, "max_abs_200_vs_padded_bs1": 0.0, "max_abs_padded_bs32_vs_bs1": 0.0, "max_abs_padded_bs32_vs_fp32_ref": 0.01613175868988037, "max_abs_200_bs1_vs_fp32_ref": 0.01613175868988037, "argmax_agree_padded_vs_200": 1.0}
```

### P3: `load_model(int8_state_dict=)` with `at/at_int8_epoch07.pt` (verbatim)

```
load: OK, missing: [], unexpected: []
int8_weights_equal_saved: [49, 0]   biases_equal_saved: [49, 0]   norm_params_equal_saved: 50 / 50
logits_finite: true   max_abs_logit_diff_vs_untrained_base: 2.84375
```

### T: `tests/test_runlog.py`: 14 passed (4.3 s). P3 wall time 21.4 s.

### Reading

- **CONFIRMED (T4, bnb 0.50.2).** In the tested set, int8 cuBLASLt fails at batch > 16 exactly when N ≡ 8 (mod 16): N = 200, 216, 232, 1000.
  - Every multiple of 16 that was tested (208, 224, 240, 256, 384, 1024) works at every batch size.
  - The general "multiple of 16" rule is a HYPOTHESIS beyond the tested N; 1008 was not tested.
  - DeiT-S's other INT8 layers (384, 1152, 1536 outputs) are multiples of 16, so only the head is affected.
  - **The Imagenette 1000-way INT8 head is affected too on T4.** The H100 is UNVERIFIED (different kernels); run P1 there.
- **CONFIRMED.** Padding the 200-way head is numerically exact:
  - The int8 weights and per-row scales of the real rows are bit-identical.
  - The logits are bit-identical to the unpadded layer at batch 1 and independent of batch size (max abs diff 0.0).
  - The input-gradient equivalence was NOT measured (P2 compared forwards only). UNVERIFIED.
- **CONFIRMED.** The INT8 reload fix restores a July INT8 checkpoint exactly: all 49 int8 weights, 49 biases and 50 norm tensors match the saved file.
- **CONFIRMED.** The Wilson fix passes its test.
- HYPOTHESIS, strongly supported: the July/Q4 INT8 per-epoch sweep failure (A-13) was this head error, since it evaluated at batch 32.

---

## R-003 · W1c: INT8 padding check, smoke, calibrate · 2026-11-12

- **Notebook:** `notebooks/w1c_checks_kaggle.ipynb`. Kaggle version: *not given*.
- **Git:** `ec45a91` (branch `revision-w1`).
- **Platform:** Kaggle Tesla T4. torch 2.11.0+cu130, bnb 0.50.2.
- **Inputs:** `jainmanav20/advc-results`, `albert1994/tinyimagenet`.
- **Run JSONs:** smoke `20261112T055011Z-8df825ed`, calibrate `20261112T055244Z-f57b81c4`.
- **Exit codes:** all 0.
  - Wall times: prepare_tiny 117.4 s, pytest_all 15.2 s, p4 23.5 s, smoke 153.0 s, calibrate 108.9 s.
- **Tests:** 39 passed in 9.32 s.

### P4: full DeiT-S INT8, padded (208) vs unpadded (200) head, 16 real val images (verbatim)

```
base:        logits_bit_equal true, max_abs 0.0, pred_agree 1.0, acc16 [0.9375, 0.9375],
             grad_bit_equal false, grad_max_abs 1.0011717677116394e-08, grad_rel 9.731061254569795e-07, grad_sign_agree 1.0,
             padded_bs32 ok, acc32 0.84375, grad_finite true
at_int8_ep7: logits_bit_equal true, max_abs 0.0, pred_agree 1.0, acc16 [0.9375, 0.9375],
             grad_bit_equal false, grad_max_abs 2.3515895009040833e-08, grad_rel 8.729399496587575e-07, grad_sign_agree 1.0,
             padded_bs32 ok, acc32 0.90625, grad_finite true
```

### Smoke (1 batch of 32, seed 0; cudnn deterministic=True, benchmark=False) (verbatim)

```
precision  load  forward  backward      fgsm            pgd20           clean/fgsm/pgd20 correct
fp32       ok    ok       ok |g|=0.325  ok linf=0.0314  ok linf=0.0314  24/4/0 of 32
int8       ok    ok       ok |g|=0.312  ok linf=0.0314  ok linf=0.0314  23/4/0 of 32
int4       ok    ok       ok |g|=0.308  ok linf=0.0314  ok linf=0.0314  24/8/0 of 32
pgd20 step seconds: fp32 40.6603, int8 43.6166, int4 40.8284
linf_pixel 0.031370095908641815 for every attack (eps 0.03137)
MatMul8bitLt fp32->fp16 cast notice: counted 1275 (logging channel), suppressed
```

### Calibrate (GPU-only, fixed batch, 3 warm-up + 10 timed; cudnn deterministic=True, benchmark=False) (verbatim)

```
precision  tf32   bs  fwd img/s  fwd+bwd passes/s  peak GB
fp32       False  16  318.5      133.7             1.08
fp32       False  32  296.1      14.2              2.059
int8       False  16  383.4      140.8             1.043
int8       False  32  383.7      15.1              2.043
int4       False  16  613.2      223.6             0.677
int4       False  32  625.8      15.5              1.323
loader: 29.5 img/s (image_folder, num_workers=2, 160 images, includes worker start-up)
```

### Reading

- **CONFIRMED.** The padded INT8 head is equivalent to the unpadded one in the full model.
  - Logits are bit-identical.
  - Input gradients differ only by float rounding (relative max diff < 1e-6) and have identical signs, so FGSM/PGD steps are identical.
  - It holds for the base model and for a July defended checkpoint.
- **CONFIRMED.** INT8 now runs at batch 32. All three precisions pass load, forward, backward, FGSM and PGD-20, and every perturbation is within eps.
- **CONFIRMED (this setup).** Forward+backward at batch 32 is ~9–14× slower than at batch 16 at every precision, including fp32. Forward-only is not slowed.
  - Because fp32 is affected too, the cause is not bitsandbytes.
  - HYPOTHESIS: `cudnn.deterministic=True` (set only by the new harness, `utils/seeding.py:66`; the July scripts never set it) selects a slow algorithm for the patch-embedding Conv2d backward at batch 32. To be tested in W1d.
  - The smoke PGD-20 time (~41 s for 32 images) matches the slow batch-32 rate, so R-001's "~17 passes/s" was this slowdown, not the T4's real speed.
- **Rates at batch 16** (fwd+bwd passes/s): fp32 133.7, int8 140.8, int4 223.6. These are the planning rates for TinyImageNet until W1d.
- **Loader rate** of 29.5 img/s is UNVERIFIED as a steady-state rate (only 160 images, worker start-up included). If real, it would bottleneck training; measure it properly in W1d.

---

## R-004 · W1d: batch-size slowdown + loader probe · 2026-11-13

- **Notebook:** `notebooks/w1d_timing_probe_kaggle.ipynb`. Kaggle version: *not given*.
- **Commit:** `2bc09d2`. torch 2.11.0+cu130 | bnb 0.50.2 | Tesla T4. 4 CPUs.
- **Earlier attempt (commit `b6ed7bd`) failed:** the session had no GPU (`torch.cuda.is_available() is False` at the first `torch.load(map_location='cuda')`, 9.1 s, exit 1). The notebook was not at fault; the accelerator was not set. `2bc09d2` adds a GPU assert to the setup cell.
- **Wall times:** p5_gpu 150.2 s (exit 0), p6_loader 113.9 s (exit 0).
- Modes: `det1_bench0` = cudnn deterministic True / benchmark False (the harness default, `utils/seeding.py:66`); `det0_bench0` = both False (PyTorch default, which is what the July scripts ran with); `det0_bench1` = deterministic False / benchmark True. TF32 off throughout. Random inputs, 2 warm-up + 5 timed steps.

### A: fwd+bwd-to-input passes/s (verbatim values)

```
              bs16   bs24   bs32   bs48   bs64
fp32 det1_b0  139.4  15.9   15.5   133.5  135.2
fp32 det0_b0  135.9  133.5  132.2  132.1  132.4
fp32 det0_b1  132.7  130.6  127.8  127.4  127.1
int4 det1_b0  201.8  14.8   14.6   214.7  217.7
int4 det0_b0  213.1  213.9  217.6  221.7  222.1
int4 det0_b1  215.8  216.9  219.7  223.5  225.3
```

### B: patch-embedding Conv2d(3,384,16,16) alone, backward-to-input, images/s (verbatim values)

```
              bs16    bs32    bs64
det1_bench0   7976.0  16.5    7746.6
det0_bench0   7292.1  9115.1  9818.3
det0_bench1   7931.6  9909.2  10096.6
```

### C: same fp32 bs32 input gradient computed twice in one process (verbatim)

```
det1_bench0  bit_equal true   max_abs_diff 0.0       sign_agree 1.0
det0_bench0  bit_equal true   max_abs_diff 0.0       sign_agree 1.0
det0_bench1  bit_equal false  max_abs_diff 3.78e-10  sign_agree 1.0
```

### D: DataLoader, first batch excluded, same 1+40·bs images for both worker counts (verbatim)

```
val_eval    bs32  workers2  1249 images  46.62 s  26.8 img/s
val_eval    bs32  workers4  1249 images   1.60 s  778.8 img/s
train_train bs16  workers2   625 images  22.24 s  28.1 img/s
train_train bs16  workers4   625 images   0.91 s  685.5 img/s
```

### Reading

- **CONFIRMED (T4, this torch/cuDNN).** The slowdown is the patch-embedding conv backward under `cudnn.deterministic=True`, at batch 24 and 32 only.
  - Conv alone: 16.5 img/s at bs32 vs ~8000 at bs16/64.
  - Whole model: bs24/32 are ~9–15× slower; bs16/48/64 are normal.
  - It does not occur with deterministic=False (both benchmark settings). Precision is irrelevant (fp32 and int4 identical pattern).
- **Measured rates without the slowdown** (det0_bench0): fp32 ~132–136, int4 ~213–222 fwd+bwd passes/s, flat across bs16–64. INT8 was not in W1d; R-003 bs16 = 140.8.
- **CONFIRMED (one repeat, one process).** det0_bench0 gave a bit-identical bs32 input gradient; benchmark=True did not (3.8e-10, signs identical). Not shown: cross-process or cross-session reproducibility under det0_bench0.
- **Loader: the workers2→workers4 jump is confounded.** Both worker counts read the *same* images, workers2 first.
  - `/kaggle/working/tiny_if` holds symlinks into the `/kaggle/input` mount (`scripts/prepare_tiny_imagenet.py:40-47`). The first pass was a cold read; the second very likely came from the page cache.
  - A 29× speed-up from 2× workers is implausible, so the cold-read reading is the likely one. It is still a HYPOTHESIS.
  - Cold rate ≈ 27–28 img/s; warm rate ≥ 685 img/s.
  - If the cold rate holds, the first pass over the 100k TinyImageNet train images takes about 1 h per new session (100,000 / 28 ≈ 3,570 s). That is arithmetic, not a measurement.
