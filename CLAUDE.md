# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
---

### Per-dataset head: frozen ImageNet (imagenette) vs fine-tuned (tiny-imagenet)

DeiT-S loads with a pretrained 1000-way ImageNet head. Two regimes, selected by
per-dataset keys in `configs/base.yaml` (`num_classes`, `label_mode`, `finetuned_base`):

- **imagenette** — `num_classes: 1000`, `label_mode: imagenet`, no `finetuned_base`.
  Its 10 folders ARE ImageNet classes, so the FROZEN head classifies them directly
  (`utils/datasets.py` remaps the 10 folders → true ImageNet indices). No fine-tune.
- **tiny-imagenet** — `num_classes: 200`, `label_mode: raw`, `finetuned_base` set.
  The frozen head fails here (64px→224, clean_acc ~0.48, AT can't recover robustness
  — a ceiling probe confirmed no AT recipe fixes it). So DeiT-S is FIRST fine-tuned
  with a fresh 200-way head via `scripts/finetune_backbone.py`, and every subsequent
  `load_model()` applies that checkpoint BEFORE compression (int8/int4 quantise the
  fine-tuned weights, not the raw pretrained ones). Labels stay raw 0..199 (no remap).

`models/loader.py:load_model()` reads all three keys; `_resolve_finetuned_path()`
scopes the checkpoint by dataset name. AT+KD teacher and student both come from
`load_model(...)`, so they automatically share the fine-tuned 200-way head (KD over
200 logits). Adding a new fine-tuned dataset = add its sub-block + run the fine-tune.

### Running experiments (on Kaggle — scripts are not meant to run locally)

```bash
# Step 0 (tiny-imagenet ONLY) — fine-tune the 200-way head FIRST. Produces
# results/checkpoints/tiny-imagenet/base/deit_small_ft.pt, which load_model()
# then applies before every compression level. Skip for imagenette.
python scripts/finetune_backbone.py                 # full fine-tune (config: finetune.epochs)
python scripts/finetune_backbone.py --epochs 1 --n 4000   # cheap sanity run first

# Phase 1 — baseline, no defense
python experiments/eval_phase1.py --model deit_small

# Phase 2a — AT defense (trains then evaluates; use --skip-training to load checkpoint)
python experiments/eval_phase2_at.py --model deit_small [--skip-training] [--compression fp32|int8|int4]

# Phase 2b — AT+KD defense
python experiments/eval_phase2_atkd.py --model deit_small [--skip-training] [--compression fp32|int8|int4]

# Phase 3 — combined attack (eval_phase3.py not yet built — see build order below)
python experiments/eval_phase3.py --model deit_small
```

### Kaggle multi-session workflow (CRITICAL — the 36-row matrix does NOT fit in one 12h session)

The primary notebook is `notebooks/run_phases_kaggle.ipynb`. Total compute (~20 GPU-h) exceeds
Kaggle's 12h/session cap, so work is split across sessions. The atomic unit of progress is
**one compression level trained to completion** (all 7 epochs), because `--skip-training` only
reloads the FINAL-epoch checkpoint (`*_epoch07*.pt`). A level interrupted mid-training is NOT
resumable and retrains from scratch.

**Persistence rules (do not skip — these are how work survives across sessions):**
- `/kaggle/working` is WIPED between sessions. The ONLY way to carry results/checkpoints forward
  is to save `results/` to a Kaggle **dataset** (`kaggle datasets version -p .../results`) at the
  end of every session, and re-attach it next session. Do not rely on working-dir persistence.
- Notebook Cell 5b restores CSVs + checkpoints from the attached dataset to the exact paths the
  scripts read. Cell 5c verifies resume state BEFORE spending GPU. Cell 13 saves at session end.
- Cells 8/9 auto-detect a restored `*_epoch07*.pt` and add `--skip-training` (eval only, no retrain).

**Per-dataset separation (prevents imagenette/tiny-imagenet mixing) — applies to CSVs,
checkpoints, AND figures:**
- `models/loader.py:dataset_results_path()` rewrites CSV paths →
  `results/<dataset.name>/phaseN_results.csv`.
- `models/loader.py:dataset_scoped_dir()` rewrites checkpoint/figure DIRS →
  `results/checkpoints/<dataset.name>/{at,atkd}` and `results/<dataset.name>/figures`.
  This is required because checkpoint/figure FILENAMES do not encode the dataset (e.g.
  `at_fp32_epoch07.pt`), so without scoping an int8 checkpoint or a figure from one dataset
  would silently clobber the other's — a data-integrity bug for the paper.
- Every consumer routes through these helpers: both defense files (checkpoint WRITE), the phase2/3
  scripts (checkpoint LOAD for `--skip-training`), and paper_figures/visualize_samples (figure dirs).
  If you add a new consumer of `paths.checkpoints_*` or `paths.figures_dir`, wrap it in
  `dataset_scoped_dir(...)` too.
- ALWAYS restore/preview/save using the per-dataset paths. Reading flat `results/phase1_results.csv`
  finds nothing (silent "not generated"); restoring to flat dirs breaks resume. This was a real bug.
- Result of all this: tiny-imagenet and imagenette occupy entirely disjoint subtrees, so saving the
  whole `results/` tree to one Kaggle dataset can never mix them.

**Suggested session split** (each fits under 12h; set `LEVELS` in Cells 8/9):
| Session | Work | Approx |
|---|---|---|
| 1 | setup + Phase 1, then Phase 2a `LEVELS=['fp32','int8']` | ~6h |
| 2 | restore, Phase 2a `LEVELS=['int4']` | ~3h |
| 3 | restore, Phase 2b `LEVELS=['fp32','int8']` | ~7h |
| 4 | restore, Phase 2b `LEVELS=['int4']` | ~4h |
| 5 | restore, Phase 3 + figures | ~4h |

**Kaggle data-prep gotcha (fixed):** `scripts/prepare_tiny_imagenet.py` takes `--root` (read-only
source, may be a `/kaggle/input` mount) AND `--out` (writable, e.g. `/kaggle/working/tiny_if`).
Writing `train_if/`/`val_if/` into a read-only mount raises `Errno 30`. `configs/base.yaml` points
tiny-imagenet dirs at `/kaggle/working/tiny_if/...`. Never symlink the read-only mount as the output.

### Attack interface (every attack must match this signature)

```python
attack(model, images, labels) -> perturbed_images   # Tensor, same shape as images
```

`build_attack(model, config)` returns the callable; `run_attack(attack, images, labels)` is the uniform wrapper.

### Adding a new attack or experiment

1. Read `attacks/fgsm.py` as the canonical template — same structure for every attack.
2. Pull all params from `configs/base.yaml` — never hardcode epsilon, steps, patch size, etc.
3. Include a `if __name__ == "__main__":` sanity-check block (dummy forward pass).
4. For experiment scripts, copy the resumability pattern from `experiments/eval_phase1.py:load_completed_runs()`.

### Key invariants the code enforces

- **Compress before defend** — model is always quantized before AT or AT+KD fine-tuning.
- **Teacher always frozen** — `teacher.eval()` + `torch.no_grad()` on every teacher forward in `defenses/at_kd.py`.
- **INT4 checkpoints save the full model** (not state_dict) because `bitsandbytes` NF4 weights can't be loaded with `load_state_dict` alone — see `save_checkpoint()` in both defense files.
- **LogitsWrapper** — HuggingFace model outputs are dataclasses; both defense files and phase1 wrap models with `_LogitsWrapper` so downstream code receives plain tensors.
- **Per-compression learning rates** in `configs/base.yaml` (`at.lr_per_compression`) — INT4 gets 10× lower LR than FP32 to avoid destroying fragile quantized weights.

### Architecture in one paragraph

`models/loader.py` loads DeiT-S at fp32/int8/int4 via timm + bitsandbytes and returns an eval-mode model. `attacks/` wraps torchattacks (FGSM, PGD) or provides a custom class (PatchAttack); each file is standalone with its own config loader. `defenses/` fine-tunes the already-compressed model: AT with FGSM adversarial inputs only, AT+KD adding a frozen FP32 teacher KL loss. `experiments/` scripts wire the pieces together in loops over compression levels and attacks, appending to CSV immediately after each row via `utils/metrics.py:save_results_to_csv()`. All hyperparameters live exclusively in `configs/base.yaml`.

### Files not yet built (build in this order)

```
attacks/combined.py          # chains fgsm → pgd → patch sequentially
experiments/eval_phase3.py   # combined attack × all defenses → phase3_results.csv
notebooks/01_results_viz.ipynb
```

---

# Research spec — Adversarial Robustness Under Compression for Edge ViTs

## Project overview

Research project studying whether adversarial defenses remain effective after model
compression (quantization) on Vision Transformers (ViTs) deployed at the edge.

We evaluate two defenses — Adversarial Training (AT) and Adversarial Training with
Knowledge Distillation (AT+KD) — applied to a compressed DeiT-S model, then attacked
individually and in combination across three compression levels.

### Core research questions

> 1. After compressing DeiT-S (INT8, INT4), does AT still recover meaningful robustness
>    — and does recovery degrade as compression becomes more aggressive?
> 2. Does AT+KD recover MORE robustness than AT alone on a compressed model — and does
>    this advantage shrink at INT4 where quantization is most aggressive?

### Scope decision

- **Model: DeiT-S only.** DeiT-B excluded to stay within Kaggle GPU quota.
  DeiT-B is noted as future work in the paper.
- **Compute budget: Kaggle free tier** — 30 GPU hours/week on T4. Every experiment
  decision must account for this. Estimated usage: ~20h experiments + ~10h buffer for
  reruns and debugging.

---

## Core pipeline (always in this order — never deviate)

```
Pretrained DeiT-S (FP32)
        ↓
    Compress                         ← PTQ to INT8 or INT4 via bitsandbytes
        ↓
Compressed DeiT-S                    ← this is the edge-deployed model
        ↓
Apply Defense (AT or AT+KD)
        ↓  AT:    fine-tune compressed model on FGSM adversarial inputs
        ↓  AT+KD: fine-tune using FGSM adversarial inputs
        ↓          + FP32 teacher soft labels via KL divergence
        ↓
Apply Attack (FGSM / PGD / Patch / Combined)
        ↓
Measure and save results
```

### Why these two defenses and not others

- **AT** — directly retrains compressed weights. The interaction with quantization
  precision is real and measurable. Included.
- **AT+KD** — adds a FP32 teacher to guide compressed student recovery. Teacher
  soft labels provide richer supervision signal than hard labels alone, potentially
  recovering vision representations that PTQ destroyed. Included.

---

## Hardware & environment

- **Platform:** Kaggle Notebooks
- **GPU:** T4 x2 (16GB VRAM each) — select in Settings > Accelerator > GPU T4 x2
- **Session length:** up to 12h per session (9h interactive + up to 12h scheduled)
- **Code editor:** VS Code with Claude Code — write here, push to GitHub, run on Kaggle
- **Dataset:** ImageNette (10-class ImageNet subset) — attach via notebook settings
- **Persistence:** Results written to `/kaggle/working/` — download or save as dataset version after each run

### Kaggle notebook setup (run every session)

> **Prerequisites before running:**
> 1. In notebook Settings → Add-ons → Secrets: add `HF_TOKEN` with your HuggingFace token
> 2. In notebook Settings → Add-ons → Datasets: add `rodgzilla/imagenette`
> 3. Enable internet access in Settings → Internet → On

```python
# Cell 1 — verify GPU
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))   # should say Tesla T4
print(f"{torch.cuda.mem_get_info()[0]/1e9:.1f} GB free")

# Cell 2 — set HF token, install dependencies
import os
from kaggle_secrets import UserSecretsClient
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret("HF_TOKEN")
!pip install -q timm torchattacks bitsandbytes optimum pyyaml

# Cell 3 — extract ImageNette (only needed once per session)
import os
if not os.path.exists('/kaggle/working/imagenette2'):
    !tar -xzf /kaggle/input/datasets/adityakane/imagenette2/imagenette2.tgz -C /kaggle/working/
    print("Extracted.")
else:
    print("Already extracted — skipping.")

# Cell 4 — pull latest code
!git clone https://github.com/Jmanav/ADVC.git
%cd /kaggle/working/ADVC

# Cell 5 — restore any previous results (from a saved Kaggle dataset output)
# If you saved a previous run as a Kaggle dataset named "advc-results", attach it
# in notebook settings and uncomment the block below.
# import shutil, os
# results_input = '/kaggle/input/advc-results'
# os.makedirs('results', exist_ok=True)
# for f in ['phase1_results.csv', 'phase2_at_results.csv',
#           'phase2_atkd_results.csv', 'phase3_results.csv']:
#     src = f'{results_input}/{f}'
#     if os.path.exists(src):
#         shutil.copy(src, f'results/{f}')
#         print(f'Restored {f}')
```

> **Dataset path:** ImageNette is mounted at `/kaggle/input/imagenette/imagenette2-320/`.
> This is already set in `configs/base.yaml` — no path changes needed.
> The dataset is read directly from the mount point (no copying needed; Kaggle SSD is fast).

### After every run — save results

Results land in `/kaggle/working/ADVC/results/`. To persist them across sessions:

```python
# Option A — download directly from the notebook output panel (right sidebar)
# Option B — save as a new Kaggle dataset version (reusable across sessions)
import subprocess
subprocess.run([
    "kaggle", "datasets", "version",
    "-p", "/kaggle/working/ADVC/results",
    "-m", "phase results update",
], check=True)
```

---

## Model

| Model | timm name | HuggingFace ID | Params |
|-------|-----------|---------------|--------|
| DeiT-S | `deit_small_patch16_224` | `facebook/deit-small-patch16-224` | 22M |

- Load via `timm` for all cases
- For AT+KD: load FP32 teacher and compressed student simultaneously
  (~85MB + ~25MB = ~110MB total — trivially fits on T4 16GB)
- Always `model.eval()` before inference, `model.train()` before fine-tuning
- Teacher in AT+KD must always stay frozen: `teacher.eval()` + `torch.no_grad()`

---

## Compression levels

| Level | Method | Tool | Notes |
|-------|--------|------|-------|
| FP32 | None — baseline | — | Always run first, establishes upper bound |
| INT8 | Post-training quantization | `bitsandbytes` | `load_in_8bit=True` |
| INT4 | Post-training quantization NF4 | `bitsandbytes` | `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"` |

All compression is post-training (PTQ). Compression always before defense.

---

## Attacks in scope

### Individual attacks (Phase 1 and Phase 2)

| Attack | File | Params | Notes |
|--------|------|--------|-------|
| FGSM | `attacks/fgsm.py` | eps=8/255 | ✅ done |
| PGD | `attacks/pgd.py` | eps=8/255, alpha=2/255, steps=20 | Standard in robustness literature |
| Adversarial Patch | `attacks/patch.py` | patch_size=32, steps=150 | Physically realizable |

### Combined attack (Phase 3)

FGSM + PGD + Patch applied sequentially to the same input.
Implemented in `attacks/combined.py`.

**Fixed across all experiments:**
- Epsilon: 8/255 L-inf for FGSM and PGD
- Patch: 32×32 pixels on 224×224 image
- Validation set: all ~3925 ImageNette val images, seed=42
- Training set: all ~9469 ImageNette train images, seed=42
- Batch size: 32 eval / 16 training (T4 16GB)

---

## Defenses in scope

### Defense 1 — Adversarial Training (AT)

Fine-tune the already-compressed model on FGSM adversarial inputs. Backbone is
frozen except the last `num_unfrozen_blocks` transformer blocks (default 4) +
classifier head — see `_freeze_backbone()` in `defenses/adversarial_training.py`.

AT parameters (actual, `configs/base.yaml` → `defense`):
- Optimizer: AdamW, weight_decay=0.01
- LR: per-compression (`defense.lr.{fp32,int8,int4}` = 1e-5 / 5e-6 / 1e-6) —
  quantized weights are fragile, INT4 gets 10× lower LR than FP32
- Warmup: epoch 1 runs at lr/10, full lr from epoch 2 (`defense.warmup_epochs`)
- Epochs: 7
- AT epsilon: 8/255 — must always match attack epsilon
- Checkpoint: save after every epoch to `results/checkpoints/<dataset>/at/`
  (see "Kaggle multi-session workflow" for the per-dataset scoping)

**FGSM-RS + clean/adversarial batch mixing (`defense.random_start`,
`defense.clean_batch_frac` — default `true` / `0.5`):**

Pure single-step FGSM AT (100% of every batch adversarial, FGSM step taken
directly from the clean image, no random start) is a documented failure mode:
Wong et al., 2020, *"Fast is Better than Free: Revisiting Adversarial
Training"* (ICLR 2020, arXiv:2001.03994) shows this setup lets the model
exploit local linearity around the clean input to trivially "solve" the
single-step attack without learning genuinely robust features —
**catastrophic overfitting**. Symptoms: training-time adversarial accuracy
looks fine, but PGD (multi-step) robust accuracy collapses toward 0, and
clean accuracy can collapse as a side effect since the model was never
anchored to the real data distribution.

This project hit this exact failure on tiny-imagenet's fine-tuned 200-way
head: fp32 AT collapsed `clean_acc` 0.857→0.39 and `pgd_robust_acc`→0.0,
reproducibly across two independent runs (2026-07-11 and 2026-07-14/15,
same config, same result). The fix implemented (both `adversarial_training.py`
and `at_kd.py`) combines two mitigations from the AT robustness literature:
- **Random start (FGSM-RS)**: perturb from a uniformly random point inside
  the L∞ eps-ball (in pixel space) before taking the FGSM gradient step,
  instead of starting from the clean image. This is the specific fix Wong et
  al. (2020) identify for catastrophic overfitting in single-step AT.
- **Clean/adversarial batch mixing** (`clean_batch_frac=0.5`): half of every
  training batch is left unperturbed. This is a complementary, not redundant,
  mitigation — it protects clean accuracy specifically by keeping the decision
  boundary anchored to the real data distribution, whereas random-start fixes
  the adversarial-example quality itself. Neither alone addresses what the
  other does.

This is a deliberate, documented deviation from vanilla single-step FGSM AT —
cite it in the paper's methodology as "FGSM-RS with 50% clean/adversarial
batch mixing," not plain "FGSM adversarial training."

**Guardrail history:** both defense scripts print
`[AT] Pre-training baseline clean_acc=...` before epoch 1 and warn loudly
(`*** WARNING: clean_acc dropped ... ***`) if any epoch's clean_acc falls
>15% below that baseline. Prior to 2026-07-14 this baseline was hardcoded to
`0.0` (`baseline_clean_acc = 0.0  # baseline check skipped — diagnostic only`),
which meant the drop was always computed against 0 and the warning could
never fire — the fp32 AT collapse ran to completion undetected. Fixed by
measuring the real baseline via `_measure_clean_acc()` before training starts.

### Defense 2 — Adversarial Training + Knowledge Distillation (AT+KD)

Fine-tune compressed student using adversarial inputs and FP32 teacher soft labels.
Same FGSM-RS + batch-mixing treatment as AT above applies here — the teacher
and student both see whatever `adv_images` ends up containing (mixed clean +
random-start-FGSM), so the CE and KL loss terms stay consistent with each other.

Loss function:
```
adv_images   = FGSM(student, images_or_random_start, labels, eps=8/255)  # 50% clean, 50% adversarial
teacher_soft = softmax(teacher(adv_images) / temperature)   # no gradients
student_soft = softmax(student(adv_images) / temperature)

loss = alpha * CrossEntropy(student(adv_images), true_labels)
     + (1 - alpha) * KLDivergence(student_soft, teacher_soft)
```

AT+KD parameters:
- All AT params above (including FGSM-RS + batch-mixing), plus:
- Temperature: 4.0
- Alpha: 0.5
- Teacher: FP32 DeiT-S, frozen throughout — never update teacher weights
- Checkpoint: save after every epoch to `results/checkpoints/<dataset>/atkd/`

---

## Experiment matrix — DeiT-S only

### Phase 1 — No defense (baseline) — ~4 CU

| Compression | FGSM | PGD | Patch |
|-------------|------|-----|-------|
| FP32 | ○ | ○ | ○ |
| INT8 | ○ | ○ | ○ |
| INT4 | ○ | ○ | ○ |

9 rows → `results/phase1_results.csv`

### Phase 2a — AT defense — ~18 CU

| Compression | FGSM | PGD | Patch |
|-------------|------|-----|-------|
| FP32 + AT | ○ | ○ | ○ |
| INT8 + AT | ○ | ○ | ○ |
| INT4 + AT | ○ | ○ | ○ |

9 rows → `results/phase2_at_results.csv`

### Phase 2b — AT+KD defense — ~22 CU

| Compression | FGSM | PGD | Patch |
|-------------|------|-----|-------|
| FP32 + AT+KD | ○ | ○ | ○ |
| INT8 + AT+KD | ○ | ○ | ○ |
| INT4 + AT+KD | ○ | ○ | ○ |

9 rows → `results/phase2_atkd_results.csv`

### Phase 3 — Combined attack vs all defenses — ~9 CU

| Compression | No defense | AT | AT+KD |
|-------------|-----------|-----|-------|
| FP32 | ○ | ○ | ○ |
| INT8 | ○ | ○ | ○ |
| INT4 | ○ | ○ | ○ |

9 rows → `results/phase3_results.csv`

**Total: 36 rows, ~53 CU estimated, ~47 CU buffer for reruns and debugging.**

---

## Metrics — every CSV row must contain all fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | str | `deit_small` |
| `compression` | str | `fp32`, `int8`, or `int4` |
| `defense` | str | `none`, `at`, or `at_kd` |
| `attack` | str | `fgsm`, `pgd`, `patch`, or `combined` |
| `clean_acc` | float | Accuracy on clean inputs (0–1) |
| `robust_acc` | float | Accuracy under attack (0–1) |
| `asr` | float | Attack success rate = 1 − robust_acc |
| `robustness_gap` | float | clean_acc − robust_acc |
| `phase` | int | 1, 2, or 3 |

---

## Project structure

```
ADVC/
├── CLAUDE.md                              # this file — read before every task
├── README.md
├── requirements.txt
├── configs/
│   └── base.yaml                          # all hyperparameters — never hardcode
├── models/
│   └── loader.py                          ✅ done
├── attacks/
│   ├── fgsm.py                            ✅ done
│   ├── pgd.py                             ← build next
│   ├── patch.py                           ← after pgd
│   └── combined.py                        ← last (Phase 3)
├── defenses/
│   ├── adversarial_training.py            ← AT on compressed model
│   └── at_kd.py                           ← AT+KD, frozen FP32 teacher
├── experiments/
│   ├── eval_phase1.py                     ← no defense, 3 attacks
│   ├── eval_phase2_at.py                  ← AT defense, 3 attacks
│   ├── eval_phase2_atkd.py                ← AT+KD defense, 3 attacks
│   └── eval_phase3.py                     ← combined attack, all defenses
├── results/
│   ├── phase1_results.csv
│   ├── phase2_at_results.csv
│   ├── phase2_atkd_results.csv
│   ├── phase3_results.csv
│   ├── checkpoints/
│   │   ├── at/                            ← epoch checkpoints for AT
│   │   └── atkd/                          ← epoch checkpoints for AT+KD
│   └── figures/                           ← all paper plots saved here
├── notebooks/
│   ├── 00_setup_check.ipynb
│   └── 01_results_viz.ipynb
└── utils/
    └── metrics.py                         ✅ done
```

---

## Build order for remaining files

```
1. attacks/pgd.py                   same structure as fgsm.py
2. attacks/patch.py                 same calling convention
3. defenses/adversarial_training.py AT on compressed model, checkpoint every epoch
4. defenses/at_kd.py                frozen FP32 teacher + compressed student
5. experiments/eval_phase1.py       no defense — phase1_results.csv
6. experiments/eval_phase2_at.py    AT defense — phase2_at_results.csv
7. experiments/eval_phase2_atkd.py  AT+KD defense — phase2_atkd_results.csv
8. attacks/combined.py              chains fgsm → pgd → patch
9. experiments/eval_phase3.py       combined attack — phase3_results.csv
10. notebooks/01_results_viz.ipynb  paper plots
```

---

## Coding conventions

- **Read CLAUDE.md first** before writing any file
- **No hardcoded values** — all params from `configs/base.yaml`
- **Attack interface** — every attack callable as
  `attack(model, images, labels) → perturbed_images`
- **Compress before defend** — never apply AT or AT+KD before compression
- **Teacher always frozen** in AT+KD — `teacher.eval()` and `torch.no_grad()`
  on every teacher forward pass, no exceptions
- **Seeds everywhere** — `torch.manual_seed(42)`, `random.seed(42)`,
  `np.random.seed(42)` at top of every script
- **tqdm on all loops** — progress visible in Colab output
- **Type hints** on all function signatures
- **Docstrings** on every function
- **Append to CSV immediately** after each row — never accumulate in memory
- **T4 batch size** — use 32 for eval, 16 for AT/AT+KD training

---

## Resumability pattern (mandatory in every experiment script)

```python
import os, csv

RESULTS_FILE = "results/phase1_results.csv"

completed = set()
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        for row in csv.DictReader(f):
            completed.add((row['model'], row['compression'],
                           row['defense'], row['attack']))

for compression in COMPRESSIONS:
    for attack_name in ATTACKS:
        key = ('deit_small', compression, defense_name, attack_name)
        if key in completed:
            print(f"Skipping {key} — already done")
            continue
        result = run_eval(compression, attack_name)
        append_to_csv(RESULTS_FILE, result)
```

---

## What NOT to do

- Do not apply defense before compression — always compress first
- Do not use input preprocessing or randomized smoothing
- Do not load full ImageNet — use ImageNette (10-class subset) only
- Do not run more than 7 AT/AT+KD epochs — CU budget
- Do not apply gradients to the teacher in AT+KD — frozen always
- Do not use `plt.show()` in scripts — save to `results/figures/`
- Do not commit weights, checkpoints, or CSVs to git
- Do not hardcode any path or hyperparameter
- Do not switch to CPU — always enable T4 x2 GPU in Kaggle notebook settings

---

## GPU hours tracker

Update this manually after each experiment run. Kaggle free tier: 30 GPU h/week on T4.

| Phase | Estimated GPU hrs | Actual GPU hrs | Status |
|-------|-------------------|----------------|--------|
| Phase 1 | ~2h | — | pending |
| Phase 2a (AT) | ~8h | — | pending |
| Phase 2b (AT+KD) | ~10h | — | pending |
| Phase 3 | ~4h | — | pending |
| Buffer | ~6h | — | reserved |
| **Total** | **~30h** | | |

---

## Expected findings to watch for

**Phase 1:**
- ASR increases FP32 → INT8 → INT4 — compression makes models more vulnerable
- PGD ASR > FGSM ASR at every level — PGD is strictly stronger
- If this pattern does not hold, check the attack implementation

**Phase 2:**
- Both AT and AT+KD reduce ASR vs Phase 1
- AT+KD should outperform AT — richer teacher supervision
- The AT vs AT+KD gap should narrow at INT4 — core finding if confirmed
- If gap does NOT narrow: also significant, means KD helps equally regardless of compression

**Phase 3:**
- Combined ASR > any individual attack
- AT+KD should still outperform AT under combined attack
- If AT+KD collapses at INT4 under combined attack — headline result

**Sanity checks — flag immediately if any of these occur:**
- clean_acc drops more than 5% FP32 → INT4 → quantization setup wrong
- AT+KD performs worse than AT → teacher not frozen or temperature wrong
- ASR = 0.0 or 1.0 for any cell → implementation bug, do not include in paper
- T4 GPU not available → do not run on CPU; wait for Kaggle GPU quota to reset
- clean_acc collapses >15% after AT/AT+KD training AND pgd_robust_acc≈0 →
  catastrophic overfitting (see "Defense 1 — Adversarial Training (AT)" above,
  Wong et al. 2020). The per-epoch guardrail prints
  `*** WARNING: clean_acc dropped ... ***` when this happens — do not ignore it.
