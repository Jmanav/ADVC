# Adversarial Robustness Under Compression for Edge Vision Transformers

> **Draft — Abstract, Introduction, and Conclusion.** All quantitative claims are drawn
> verbatim from `results/imagenette/*.csv` and `results/tiny-imagenet/*.csv` (Phases 1, 2a, 2b).

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
Section 4 concludes, interpreting the negative results and their implications for edge
deployment alongside the study's limitations and future work.

---

## 4. Conclusion

We asked whether adversarial defenses applied to an already-compressed Vision Transformer
retain their value, and whether that value degrades as quantization sharpens. The answer our
experiments give is narrow and mostly negative: single-step adversarial training—with or
without knowledge distillation—recovers clean accuracy and blunts weak and patch attacks at
full precision, but it does not deliver robustness to a strong multi-step adversary, and
whatever advantage distillation offers does not survive compression.

The negative results are the substance, not a caveat. Robustness to 20-step PGD stays at or
near zero in every configuration we measured—`robust_acc = 0.0` across all defended
Imagenette cells, and never above 0.0079 anywhere on Tiny-ImageNet—so the single-step FGSM
training signal simply does not transfer to the stronger attack the defense was meant to
resist. The distillation term is likewise no reliable help: on Imagenette FGSM, AT+KD and AT
are effectively tied under quantization (INT4: 0.361 vs. 0.363; INT8: 0.425 vs. 0.426), and
on Tiny-ImageNet the two swap places depending on the attack—AT leads on FGSM at full
precision (0.274 vs. 0.170) while AT+KD leads on the patch attack (0.130 vs. 0.058). Where
the defenses do earn their keep is exactly where the attack is weakest: Imagenette patch
robustness climbs from 0.192 undefended to 0.688 (AT) and 0.748 (AT+KD) at FP32, alongside a
clean-accuracy recovery into the high 0.90s. A full-precision teacher, in short, does not
reliably rescue a compressed student against the attacks that matter most.

These conclusions are bounded by the study's scope. Compression here is *weight-only,
data-free* PTQ, which reduces the memory footprint but is not integer-only inference, so our
numbers should not be read as robustness under fully-integer edge kernels. The evidence rests
on a single backbone, DeiT-Small, and two datasets whose classification heads differ by
construction (a frozen 1000-way ImageNet head for Imagenette, a fine-tuned 200-way head for
Tiny-ImageNet), leaving architecture and task breadth open. The training adversary is
single-step FGSM throughout—chosen under a fixed dual-T4 budget—so we do not separate "the
defense is weak" from "single-step training is weak," and the evaluation attacks are limited
to FGSM, PGD, and the adversarial patch.

The future work that matters follows directly from those bounds. First, substitute a
multi-step inner maximiser (PGD-AT or TRADES) for single-step FGSM training and re-run the
matrix, which would disentangle defense strength from training-attack strength and test
whether PGD robustness is recoverable at all under compression. Second, extend beyond
DeiT-Small to at least one larger ViT and one convolutional baseline, and toward genuinely
integer-only quantization, to establish whether the near-zero PGD result and the vanishing
distillation benefit are properties of this model and this weight-only regime or of
compressed edge models generally.

---

## Assumptions and open questions for the author

Please confirm or correct the following before submission:

1. **Defense description vs. `methodology.md`.** I described the training adversary as
   **FGSM-RS with 50% clean/adversarial batch mixing**, following the *code*
   (`defenses/adversarial_training.py`, `defenses/at_kd.py`) and `CLAUDE.md`. Your
   `methodology.md` §4.1 still describes plain single-step FGSM-AT and lists catastrophic
   overfitting only as a limitation. **The methodology section needs updating to match the
   code** — the abstract/intro here assume the code is the ground truth. Confirm.

2. **Numbers cited in the draft — all verbatim from the CSVs.** For your quick cross-check:
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

3. **"Weight-only, data-free PTQ" scope.** I kept the edge/compression claims scoped to
   memory-footprint reduction (not integer-only inference), matching `methodology.md` §3.3.
   If you intend a stronger deployment claim, flag it.

4. **No reference paper was attached.** I matched a generic ML-security-venue register
   (terse, claim-forward, honest). If you have a specific target venue/paper whose tone you
   want mirrored, share it and I'll re-tune.

5. **A framing choice to confirm.** The abstract, intro, and conclusion present the two
   *negative* results (no PGD robustness; KD advantage doesn't survive quantization) as the
   paper's headline contribution. This is the honest reading of your data, but it reframes
   the project away from the "AT+KD wins" hypothesis in `CLAUDE.md`'s expected-findings.
   Confirm you're comfortable leading with the negative result.

6. **Section structure.** Phase 3 (combined attack) is now removed entirely from this draft,
   and there is no separate Discussion section: the negative-result interpretation lives in
   the Conclusion (§4), and the roadmap and section numbering (§1 Intro → §2 Methodology →
   §3 Results → §4 Conclusion) reflect this. `methodology.md` still describes the Phase 3
   combined attack (§5.4) — if this paper is to drop Phase 3 completely, that section of
   `methodology.md` should be cut too. Confirm.

7. **New numbers introduced in the Conclusion — all verbatim from the CSVs, none new to the
   paper's claim set.** For cross-check:
   - Imagenette FGSM: AT+KD vs AT — INT4 **0.361019** vs **0.362803**; INT8 **0.425478** vs
     **0.426497** (`phase2_atkd`/`phase2_at_results.csv`). Rounded to 3 dp in prose. ✔
   - Tiny-ImageNet FGSM FP32: AT **0.2738** vs AT+KD **0.1697**; patch FP32: AT+KD **0.130**
     vs AT **0.058** (`tiny-imagenet/phase2_*`). ✔ These illustrate the "AT and AT+KD swap
     places by attack" point; they extend the intro's claim with specifics but introduce no
     *new* claim.
   - "never above 0.0079" for Tiny-ImageNet PGD = the undefended INT4 cell (0.0079); all
     defended Tiny-ImageNet PGD cells are ≤0.0054. ✔ Imagenette defended PGD = 0.0
     everywhere. ✔
