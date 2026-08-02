# Methodology Notes

This document records design decisions and debugging history that shaped the
final experimental setup, for transparency and reproducibility.

## Base setup

- Base encoder: `xlm-roberta-base` (278,045,955 total parameters), frozen except
  for adapters and the classification head.
- Task: 3-way NLI classification (entailment / neutral / contradiction).
- Dataset: IndicXNLI (Hindi and Telugu, forward direction).
- Sequence length: 128 tokens, justified by truncation analysis in notebook 03
  (>99% of examples in both languages fit within 128 tokens).

## Budget design

Training budgets: 50, 100, 500, 1000, 2000, 20000 samples per language, built as
*nested* stratified subsets (the 50-sample set is a subset of the 100-sample set,
and so on), using `StratifiedShuffleSplit` with a fixed seed (42) to preserve
label balance at every size.

The largest budget (20,000) stands in for "full dataset" from the original
project proposal. The true training set size is ~392,702 samples per language;
20,000 was chosen as a compute-constrained substitute after measuring that
training at the full size would cost multiple hours per individual run, making
a 108-run sweep impractical on single-GPU Kaggle sessions. This is a deliberate,
documented deviation, not an oversight.

## A critical bug, found and fixed

**Symptom:** across three separate full-scale training sweeps (30, then two
different 108-run attempts), a large fraction of runs converged to exactly
33.3% accuracy and 16.7% macro F1 -- the random-chance signature for 3-class
classification -- regardless of training budget or random seed.

**Root cause:** `peft.get_peft_model()` freezes every parameter except the
injected adapter weights (LoRA/DoRA matrices, or IA3 rescaling vectors) by
default. Since `xlm-roberta-base` has no pretrained sequence-classification
head, `AutoModelForSequenceClassification` initializes one at random on load.
Left frozen at its random initialization, this head can never learn to
classify, regardless of how well the underlying adapters train.

**Fix:** `modules_to_save=["classifier"]` added to every PEFT config (LoRA,
DoRA, IA3), which keeps the classification head trainable alongside the
adapters. The optimizer was also corrected to filter to
`p.requires_grad`-only parameters.

**Process fix:** two hard assertions were added to prevent this class of bug
from silently producing invalid results again:
1. In the hyperparameter search notebook, a check that the best-performing
   learning rate for each method actually clears random chance before being
   saved for downstream use.
2. In the full sweep notebook, a sanity-check run (a single configuration)
   that must clear random chance before the full 108-run loop is allowed to
   proceed.

## A second issue, caught before it caused damage

An earlier version of the hyperparameter search picked a *different* learning
rate per language (e.g., LoRA: 5e-4 for Hindi vs. 1e-4 for Telugu). This was
identified as a methodological risk: since the project's core question is
whether PEFT rankings hold *across* languages, allowing the learning rate
itself to vary by language would confound any later Hindi-vs-Telugu comparison
-- a ranking difference could then reflect the LR choice rather than a real
script/language effect. The fix was to pool both languages together when
selecting the single best LR per method, so LR is a controlled variable and
language remains the one thing that varies in the final comparison.

## A stability issue at small budgets, and its partial fix

After the classifier-head fix, an early full-sweep run still showed unusual
behavior: `budget=1000` performed *worse* than `budget=500` for LoRA and DoRA,
despite having twice the data. Investigation traced this to the training loop
having no learning-rate warmup, combined with a relatively low total step count
at that budget (5 epochs x ~30 steps/epoch), leaving fp16 training briefly
unstable early on. Adding a linear warmup-then-decay schedule
(`get_linear_schedule_with_warmup`, 10% warmup) resolved the `budget=1000`
regression specifically.

A related instability remains at the smallest budgets (50 and 100 samples),
concentrated heavily in IA3, whose selected learning rate (5e-3) is 50x higher
than LoRA/DoRA's (1e-4). Rather than further modify the training recipe to
eliminate this, it is reported as-is in `results/collapsed_runs.csv` and
treated as a legitimate finding about IA3's stability profile at extremely low
data regimes, discussed in the main README's Limitations section.

## What was deliberately left out of scope

- **Cross-task transfer** (e.g., evaluating trained NLI adapters on Telugu
  sentiment or hate-speech datasets) was considered during development but
  excluded, since it is not part of the original proposal's defined
  methodology and would meaningfully extend the project's timeline and
  compute budget without answering the stated research question.
- **Full fine-tuning** was benchmarked once in notebook 03 (time and memory
  only, as a reference point) but was never trained end-to-end or evaluated
  for accuracy, since it is outside the proposal's three-method comparison.
