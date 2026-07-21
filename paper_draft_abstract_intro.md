# Adversarial Robustness Under Compression for Edge Vision Transformers

> **Draft — Abstract and Introduction.** All quantitative claims are drawn verbatim
> from `results/imagenette/*.csv` and `results/tiny-imagenet/*.csv` (Phases 1, 2a, 2b).
> Phase 3 (combined attack) is deliberately omitted here — see the assumptions list.

---

## Abstract

Vision Transformers (ViTs) are increasingly quantized to meet the memory budgets of
edge hardware, yet it remains unclear whether adversarial defenses applied to a
*compressed* model retain their protective value, or whether their effectiveness
erodes as quantization becomes more aggressive. Prior work has studied adversarial
training, knowledge distillation, and post-training quantization largely in isolation;
their interaction—defending a model that has *already* been compressed for deployment—is
comparatively under-examined. We construct a controlled compress-then-defend-then-attack
pipeline and evaluate two defenses, Adversarial Training (AT) and AT with Knowledge
Distillation (AT+KD), on a DeiT-Small backbone across three weight-only precision
levels (FP32, INT8, INT4). Both defenses fine-tune the quantized model with single-step
FGSM adversarial examples using random-start and 50% clean/adversarial batch mixing, a
recipe we adopt to avoid the catastrophic-overfitting collapse we reproduced under naive
FGSM training. We evaluate on two datasets—Imagenette (10 classes, frozen ImageNet head)
and Tiny-ImageNet (200 classes, fine-tuned head)—against FGSM, PGD, and adversarial-patch
attacks. The defenses recover clean accuracy and improve robustness to weak (single-step
FGSM) and patch attacks at full precision—on Imagenette, patch robust accuracy rises from
0.192 undefended to 0.748 with AT+KD—but **robustness to multi-step PGD remains at or near
zero regardless of defense or precision**, and the advantage of AT+KD over AT is
inconsistent and does not survive INT8 or INT4 quantization. These results caution that
single-step adversarial training, even augmented with distillation, is insufficient to
confer strong robustness on compressed edge ViTs.

---

## 1. Introduction

Vision Transformers now match or exceed convolutional networks across image recognition
tasks, and their deployment is shifting from data-center accelerators toward
resource-constrained edge devices—phones, cameras, embedded controllers—where memory and
energy are scarce. To fit within these budgets, practitioners routinely apply post-training
quantization (PTQ), storing weights at INT8 or INT4 rather than FP32 and shrinking the
model's memory footprint several-fold. The same models are also exposed to adversarial
inputs: small crafted perturbations that flip predictions while remaining imperceptible,
or conspicuous but physically realizable patches. An edge model is thus subject to two
pressures at once—it must be *small* and it must be *robust*—yet these are typically
addressed by separate techniques never designed to compose. Whether a defense applied to
an already-compressed model actually holds is of direct practical consequence for anyone
shipping a ViT to the edge.

The adversarial-robustness and model-compression literatures have each matured
independently. Adversarial training (Goodfellow et al., 2014; Madry et al., 2018) and
robustness-oriented distillation (Zhang et al., 2019) are well understood on full-precision
models; quantization schemes such as NF4 (Dettmers et al., 2023) are well understood as
compression tools. What is far less studied is the *interaction*: if a model is quantized
first—as it must be for deployment—and only then defended, does the defense recover the
robustness that compression and the attack together destroy, and does that recovery survive
as precision drops? Compounding this, single-step adversarial training is known to be
fragile: it is prone to *catastrophic overfitting*, in which robustness to multi-step
attacks collapses abruptly while the model appears to solve the single-step objective
(Wong et al., 2020; Andriushchenko & Flammarion, 2020). This failure mode is not a
side-issue for a compute-constrained edge study—it is the central pitfall, since multi-step
adversarial training across several precisions and datasets is often infeasible within a
realistic GPU budget.

We study this interaction directly with a fixed, deliberately ordered pipeline:
**compress, then defend, then attack.** A pretrained DeiT-Small backbone is quantized to the
target precision (FP32 baseline, INT8, or INT4) via weight-only PTQ; the *compressed* model
is then adversarially fine-tuned with one of two defenses—AT, or AT with a frozen
full-precision teacher supplying distillation targets (AT+KD)—and finally attacked
white-box. Both defenses generate their training adversarials with single-step FGSM using
random start (FGSM-RS) and mix 50% clean examples into every batch; this pairing was adopted
after we reproduced a catastrophic-overfitting collapse under naive FGSM training and traced
it to the two failure modes those mitigations independently address. We evaluate on two
datasets that exercise different regimes of the compressed model: Imagenette, a 10-class
ImageNet subset classified directly through the frozen pretrained head, and Tiny-ImageNet,
a harder 200-class task requiring a fine-tuned head that is quantized *after* fine-tuning.
Across both, we measure clean accuracy, robust accuracy, attack success rate, and the
robustness gap against FGSM, PGD, and adversarial-patch attacks.

Our findings are reported without smoothing. The defenses do help where prior intuition
predicts: they recover and even raise clean accuracy on the compressed model, and they
improve robustness to single-step FGSM and to patch attacks at full precision (on
Imagenette, patch robust accuracy rises from 0.192 undefended to 0.688 under AT and 0.748
under AT+KD). But two negative results dominate. First, robustness to
multi-step PGD remains at or near zero across every compression level, defended or not—the
single-step training signal does not transfer to the stronger attack. Second, the expected
advantage of AT+KD over AT is inconsistent and fails to persist under quantization: at INT8
and INT4 the two defenses are nearly indistinguishable on FGSM, and both can underperform
the undefended model. We view this honest map of where compression-time defenses hold and
where they break as the paper's contribution.

**Contributions.**

- A controlled evaluation matrix spanning three precision levels (FP32/INT8/INT4), three
  attacks (FGSM/PGD/patch), and three defense conditions (none/AT/AT+KD) on DeiT-Small,
  replicated across two datasets with contrasting classification-head regimes (frozen
  1000-way vs. fine-tuned 200-way).
- Empirical evidence that AT and AT+KD, applied to an already-quantized ViT, recover clean
  accuracy and weak-attack (FGSM/patch) robustness at full precision but **do not confer
  robustness to multi-step PGD** at any precision.
- Evidence that the benefit of knowledge distillation over plain adversarial training is
  inconsistent and **does not survive aggressive INT8/INT4 quantization**, tempering the
  common assumption that a full-precision teacher reliably aids a compressed student.
- A documented single-step training recipe—FGSM with random start and 50% clean/adversarial
  batch mixing—that avoids the catastrophic-overfitting collapse we observed under naive
  FGSM adversarial training on the compressed model.

The remainder of the paper is organized as follows. Section 2 details the methodology—the
compress-then-defend-then-attack pipeline, the DeiT-Small backbone and weight-only PTQ
scheme, the two defenses and their training recipe, and the attack configurations. Section 3
presents results phase by phase—undefended baselines, AT, and AT+KD—across both datasets.
Section 4 discusses the negative results and their implications for edge deployment, and
Section 5 concludes with limitations and future work.

---

## Assumptions and open questions for the author

Please confirm or correct the following before submission:

1. **Defense description vs. `methodology.md`.** I described the training adversary as
   **FGSM-RS with 50% clean/adversarial batch mixing**, following the *code*
   (`defenses/adversarial_training.py`, `defenses/at_kd.py`) and `CLAUDE.md`. Your
   `methodology.md` §4.1 still describes plain single-step FGSM-AT and lists catastrophic
   overfitting only as a limitation. **The methodology section needs updating to match the
   code** — the abstract/intro here assume the code is the ground truth. Confirm.

2. **Phase 3 (combined attack) omitted.** Per your decision, no combined-attack numbers
   appear in the abstract or intro. Note that Phase 3 is also incomplete on disk
   (Imagenette 5/9 rows; Tiny-ImageNet absent), and every recorded combined `robust_acc` is
   0.0. If you later complete it, the intro's "roadmap" paragraph already reserves a slot
   for it in the results section.

3. **Numbers cited in the draft — all verbatim from the CSVs.** For your quick cross-check:
   - Imagenette patch, FP32: undefended `robust_acc` **0.192** (`phase1_results.csv`), AT
     **0.688**, AT+KD **0.748** (`phase2_at`/`phase2_atkd_results.csv`). ✔ used in both
     abstract and intro.
   - Imagenette FGSM, INT4: undefended **0.565** → AT **0.363** → AT+KD **0.361** (the
     "defense underperforms undefended" claim). ✔
   - PGD `robust_acc` ≈ 0 across all cells, defended or not — verified in every
     phase1/2a/2b CSV for both datasets. ✔
   - Tiny-ImageNet FGSM, FP32: undefended **0.055** → AT **0.274** → AT+KD **0.170**;
     INT8 AT **0.024**. ✔ (These support the "recovery vanishes under quantization" claim
     but are not individually quoted in the prose — say if you'd like them added.)

4. **"Weight-only, data-free PTQ" scope.** I kept the edge/compression claims scoped to
   memory-footprint reduction (not integer-only inference), matching `methodology.md` §3.3.
   If you intend a stronger deployment claim, flag it.

5. **No reference paper was attached.** I matched a generic ML-security-venue register
   (terse, claim-forward, honest). If you have a specific target venue/paper whose tone you
   want mirrored, share it and I'll re-tune.

6. **A framing choice to confirm.** The intro presents the two *negative* results (no PGD
   robustness; KD advantage doesn't survive quantization) as the paper's headline
   contribution. This is the honest reading of your data, but it reframes the project away
   from the "AT+KD wins" hypothesis in `CLAUDE.md`'s expected-findings. Confirm you're
   comfortable leading with the negative result.
