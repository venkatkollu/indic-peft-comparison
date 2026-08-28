A PROJECT REPORT ON

DOES PEFT METHOD RANKING HOLD ACROSS SCRIPT FAMILIES?

A Data-Efficiency Study on Low-Resource Indic Languages


Submitted by

[YOUR NAME]
[ROLL NUMBER]

Under the Esteemed Guidance of

[FACULTY ADVISOR NAME]

[DEPARTMENT / SCHOOL NAME]
[UNIVERSITY / COLLEGE NAME]
[MONTH YEAR]

# CERTIFICATE

This is to certify that the project report entitled “Does PEFT Method Ranking Hold Across Script Families? A Data-Efficiency Study on Low-Resource Indic Languages” is a bona fide record of work carried out by [Your Name] under the guidance of [Faculty Advisor]. The work is submitted in partial fulfilment of the requirements for the award of [Degree Name].

# DECLARATION

I declare that this report is an original account of the study documented in the accompanying repository. All external sources are identified through citations, and all numerical results reported here are derived from the preserved experiment outputs in the project repository.

# ACKNOWLEDGEMENT

I express my sincere gratitude to my advisor, department, peers, and the open-source research community whose software, datasets, and publications made this study possible.

# ABSTRACT

This study examines whether the ranking of parameter-efficient fine-tuning methods remains invariant across data budgets and two Indic languages with different scripts. LoRA, DoRA, and IA³ were applied to XLM-RoBERTa-base for three-class natural language inference on Hindi and Telugu subsets of IndicXNLI. The primary validation sweep contains 108 runs: three methods, two languages, six training budgets, and three random seeds. Mean macro-F1 rankings are budget-dependent: DoRA is highest at 50 examples in both languages, IA³ leads at 500–2,000 examples in both languages and at 20,000 examples in Hindi, while LoRA leads Telugu at 20,000. Thus, the repository does not support a universally invariant PEFT ranking. An independent held-out test follow-up compares a hybrid IA³+LoRA configuration with the best primary method in each cell; the hybrid wins 8 of 12 cells. The report treats seed intervals as descriptive variability, separates validation and held-out evidence, documents the classifier-head safeguard, and records the limitations arising from pooled learning-rate selection and changing epoch schedules.

# TABLE OF CONTENTS

This document is structured into six chapters followed by references and reproducibility appendices. Word fields may be updated after opening the DOCX to refresh page-aware contents.

# LIST OF FIGURES

Figure 1. Primary validation learning curves. Figure 2. Winning primary method by language and budget. Figure 3. Primary computational measurements. Figure 4. Held-out hybrid deltas.

# LIST OF TABLES

Tables 1–12 summarize the dataset, design, configurations, primary results, rankings, compute, collapse cases, hybrid follow-up, research-question findings, and reproducibility index.

# LIST OF ABBREVIATIONS

PEFT — parameter-efficient fine-tuning; NLI — natural language inference; F1 — F-measure; IA³ — Infused Adapter by Inhibiting and Amplifying Inner Activations; LR — learning rate; CI — confidence interval; GPU — graphics processing unit.

# CHAPTER 1 — INTRODUCTION

## 1.1 Background

Modern NLP systems are commonly adapted from broadly pretrained encoders to task-specific data. Full fine-tuning updates every parameter, which increases storage and optimization cost when many task variants must be maintained. PEFT addresses this problem by freezing most pretrained weights and learning a small task-specific parameter set [2], [4].

## 1.2 NLP and Transfer Learning

Natural language inference asks whether a hypothesis is entailed by, contradicts, or neutral with respect to a premise. It is a useful test of sentence-pair representations because success requires more than lexical matching. Multilingual transfer is especially challenging when languages differ in morphology, word order, tokenization behaviour, and script.

## 1.3 Low-Resource and Indic NLP

Indic languages are underrepresented unevenly in digital corpora and benchmarks. IndicXNLI was introduced as an NLI dataset for 11 Indic languages, constructed from English XNLI through high-quality translation [5]. The present study selects Hindi and Telugu to examine two language/script settings rather than assuming that a result in one language transfers automatically.

## 1.4 Research Problem and Gap

Many PEFT comparisons report a single ranking at a fixed data scale. Such a ranking can conceal regime changes: an adapter that is effective with thousands of examples may be unstable when the classifier head must learn from only dozens of examples. The gap addressed here is an explicit budget-by-language comparison using the same base model, task, and three seeds.

## 1.5 Research Questions

RQ1: Does the ranking of LoRA, DoRA, and IA³ change as the training-data budget changes? RQ2: Is the observed ranking consistent for Hindi and Telugu? A supplemental question asks whether a separately tuned IA³+LoRA configuration improves over the best primary method on held-out test data.

## 1.6 Objectives and Contributions

The objectives are to implement a controlled PEFT sweep; quantify macro-F1, seed variability, trainable parameters, memory, and time; document engineering safeguards; distinguish validation evidence from held-out evidence; and provide reproducible tables and figures. The principal contribution is not a claim of universal superiority, but a budget-conditioned account of ranking behaviour.

## 1.7 Scope and Limitations

The scope is XLM-RoBERTa-base, IndicXNLI NLI, Hindi and Telugu, three PEFT methods, six budgets, and three seeds. The study does not establish performance on all Indic scripts, all tasks, full-data training, or all PEFT implementations. The primary sweep uses validation data for learning-rate selection, and the hybrid follow-up has an asymmetric tuning protocol.

# CHAPTER 2 — LITERATURE REVIEW

## 2.1 Transformer Encoders and XLM-R

XLM-RoBERTa is a multilingual masked-language model trained at scale and evaluated for cross-lingual transfer [6]. The study uses the base checkpoint as a common representation space. This choice fixes the pretrained encoder so the comparison focuses on adaptation mechanisms rather than changing the backbone.

## 2.2 LoRA

LoRA freezes pretrained weights and represents the update to selected weight matrices as a low-rank product, ΔW = B A, scaled by α/r [2]. In this implementation, rank r=8 and α=16 are applied to query and value modules. Its appeal is that the trainable update is much smaller than the dense matrix.

## 2.3 DoRA

DoRA decomposes a weight into magnitude and direction and uses a LoRA-style update for directional adaptation [3]. The intended benefit is a better separation of the two components. In the present experiment, DoRA shares the LoRA rank, alpha, dropout, target modules, and classifier-head treatment, while enabling the DoRA decomposition.

## 2.4 IA³

IA³ learns vectors that rescale internal activations rather than inserting low-rank matrices [4]. The repository targets key, value, and output.dense modules and marks output.dense as a feed-forward module. This produces the smallest trainable parameter count among the three primary methods.

## 2.5 Parameter Efficiency and Data Regimes

Parameter count is not identical to wall-clock efficiency. A method can update fewer parameters while incurring different kernels, memory traffic, or implementation overhead. A budget sweep is therefore necessary to examine both predictive quality and practical cost.

## 2.6 IndicXNLI and Research Position

IndicXNLI supplies a multilingual NLI setting in which language-specific performance can be compared using a common task definition [5]. The present work extends the comparison dimension from method-only ranking to ranking as a function of labelled-data budget, while treating Hindi and Telugu as separate evaluation strata.

## 2.7 Summary

The literature motivates three distinct adaptation mechanisms, but it does not imply that one method must dominate under every data regime. The repository experiment is designed to test that assumption descriptively and reproducibly.

# CHAPTER 3 — SYSTEM DESIGN AND METHODOLOGY

## 3.1 Research Design

The primary design is a full factorial sweep over method, language, budget, and seed. The 3×2×6×3 design yields 108 validation runs. Every cell is summarized with mean macro-F1 and standard deviation across the three seeds.

## 3.2 Dataset and Preprocessing

The executed notebooks download or access IndicXNLI-derived parquet files in the Kaggle environment. For each language and budget, the corresponding train file is paired with a shared valid file. Premise and hypothesis are jointly tokenized, padded to max_length=128, and reduced to input_ids, attention_mask, and labels.

## 3.3 Base Model and Classifier

The base checkpoint is xlm-roberta-base with three output labels. The sequence-classification classifier is newly initialized because the pretrained checkpoint is not a task-specific NLI head. A randomly initialized head must remain trainable; otherwise the adapter cannot map representations to the three labels.

## 3.4 PEFT Configurations

LoRA uses r=8, alpha=16, dropout=0.1, bias=none, query/value targets, and modules_to_save=[classifier]. DoRA uses the same configuration with use_dora=True. IA³ targets key, value, and output.dense, identifies output.dense as the feed-forward module, and also saves the classifier.

## 3.5 Classifier-Head Safeguard

The notebook performs a hard assertion that classifier parameters are trainable. Because some PEFT versions may not reliably serialize modules_to_save, the pipeline explicitly saves classifier parameters to classifier_head.pt and restores them during load. This is essential: a frozen random classifier would confound the comparison and can produce near-chance behaviour unrelated to adapter quality.

## 3.6 Hyperparameter Search

The pooled search evaluates learning rates from 5×10⁻⁶ through 5×10⁻³ using two seeds and both languages at representative budget 2,000. The selected values are LoRA 1×10⁻⁴, DoRA 1×10⁻⁴, and IA³ 5×10⁻³. Pooling prevents per-language tuning from becoming an uncontrolled source of ranking differences, but the choice is not a proof that each method is globally optimal.

## 3.7 Training Procedure

The primary sweep uses batch size 32, AdamW, mixed-precision CUDA training, and a deterministic seed initialization for torch, CUDA, and NumPy. Epochs are budget-dependent: 10 for 50–500, 5 for 1,000–2,000, and 3 for 20,000. This schedule is recorded rather than hidden, but it means examples and optimizer-update exposure are not fully isolated.

## 3.8 Metrics and Intervals

Accuracy and macro-F1 are computed on the validation split. Macro-F1 averages class-wise F1 and is appropriate when a constant-class predictor can have misleading accuracy. The repository’s 95% intervals use a two-sided t interval over three seeds, df=2, t*=4.302652729. These intervals describe seed variability, not pairwise statistical significance.

## 3.9 Near-Baseline Detection

A run is flagged when accuracy is within 0.001 of 1/3, the chance accuracy for three classes. This rule identifies 24 of 108 runs, or 22.2%, with method counts LoRA 8, DoRA 7, and IA³ 9. The flag is diagnostic and does not delete runs.

## 3.10 Reproducibility

The repository preserves notebooks 00–10, result CSV/JSONL files, adapter configuration artifacts, generated plots, requirements, and report materials. Re-execution requires reconstructing processed parquet subsets and adapting Kaggle-specific paths. The raw dataset is not stored in the repository.

Table 1. Primary PEFT configurations.

Table 2. Experimental variables.

# CHAPTER 4 — COMPUTATIONAL COST ANALYSIS

## 4.1 Compute Environment

The executed primary notebook records CUDA availability and a Tesla T4 GPU. It reports torch 2.10.0+cu128, transformers 5.0.0, datasets 5.0.0, and PEFT 0.19.1. These values describe the recorded Kaggle execution environment, not a guarantee of identical performance on other hardware.

## 4.2 Trainable Parameters

IA³ updates 657,411 parameters, compared with 887,811 for LoRA and 906,243 for DoRA. Thus IA³ is the most parameter-efficient primary method, while DoRA adds magnitude parameters to the LoRA-style update.

## 4.3 Memory and Time

The stored outputs include per-run peak GPU memory and training time. In the primary results, memory is approximately 1.16 GB for LoRA, 1.17 GB for DoRA, and 1.16 GB for IA³ in the recorded measurement procedure. DoRA generally has the longest wall-clock time, while the absolute values should be interpreted as implementation-specific.

## 4.4 Practical Trade-off

Parameter efficiency is useful for storage and optimizer state, but a practitioner also cares about time, stability, and accuracy at the available budget. The results therefore report these dimensions separately rather than collapsing them into an unsupported universal efficiency score.

Table 3. Repository compute-efficiency summary.

Figure 1. Primary computational measurements derived from stored run outputs.

# CHAPTER 5 — RESULTS AND DISCUSSION

This chapter distinguishes the primary validation sweep from the later held-out hybrid comparison. Numerical values below are computed from the repository CSV outputs. No claim of statistical significance is made because the repository contains descriptive three-seed intervals rather than pairwise tests.

Figure 2. Primary validation macro-F1 learning curves; points are means and bars are one standard deviation across seeds.

Figure 3. Winning primary method by language and training budget.

## 5.1 Primary Validation Results

Table 4. Primary validation ranking table.

The recomputed winners are DoRA for Hindi and Telugu at 50 examples; DoRA for Hindi and IA³ for Telugu at 100; IA³ for both languages at 500, 1,000, and 2,000; IA³ for Hindi and LoRA for Telugu at 20,000. This is direct evidence against a single ranking that is invariant across budgets and languages.

## 5.2 Hindi

Hindi rises from near-chance macro-F1 at 50–100 examples to approximately 0.66–0.67 at 20,000. IA³ leads the mean at 500, 1,000, 2,000, and 20,000. At 50 and 100, all methods exhibit substantial seed variability and near-baseline outcomes, so the nominal winner should not be interpreted as robust superiority.

## 5.3 Telugu

Telugu shows the same intermediate IA³ leadership but a different large-budget result: LoRA leads at 20,000. IA³ has the smallest trainable parameter count but does not retain the top mean at the largest Telugu budget, reinforcing the central budget- and language-conditioned interpretation.

## 5.4 Seed Variability and Collapse

The 24 near-baseline runs are concentrated in the smallest budgets and include all three methods. IA³ has nine flagged runs, DoRA seven, and LoRA eight. These cases explain why small-budget means and rankings should be reported with variability and diagnostic counts rather than as definitive method orderings.

## 5.5 Learning-Curve Interpretation

The curves are non-monotonic at intermediate budgets because the epoch schedule changes and the three seeds can produce materially different outcomes. The broad pattern is nevertheless clear: substantial task signal emerges by 500–2,000 examples, and all methods improve strongly by 20,000.

## 5.6 LoRA versus DoRA

DoRA’s decomposition does not produce a stable accuracy advantage over LoRA in this experiment, while its trainable count and recorded time are higher. This is a descriptive result within the tested architecture, task, budgets, and hyperparameters; it is not evidence that DoRA is ineffective generally.

## 5.7 IA³ Behaviour

IA³ is the leading primary method across the intermediate budget range in both languages and the leading Hindi method from 500 onward. Its higher selected learning rate, 5×10⁻³, differs fifty-fold from the LoRA/DoRA rate and may interact with the small-data regime. The repository supports reporting the association, not a causal claim that the learning rate alone caused every collapse.

## 5.8 Confidence Intervals

Three-seed intervals are wide in several low- and intermediate-budget cells. Because df=2 gives a large critical value, the intervals are useful for communicating instability but should not be read as formal evidence of significance or non-significance between methods.

## 5.9 Threats to Validity

Internal validity is strengthened by a fixed backbone, explicit seed grid, saved outputs, and classifier safeguards. It is weakened by the budget-dependent epoch schedule and pooled rather than per-language LR tuning. External validity is limited to two languages, one task, one backbone, and the selected implementation.

## 5.10 Held-Out IA³+LoRA Hybrid Follow-Up

The hybrid follow-up is separate from the 108-run validation study. It evaluates a dual-component IA³+LoRA configuration on held-out test data and compares it with the best primary method in each budget-language cell. Its learning-rate setup was not validated with the same search protocol, so the result is supplemental rather than a replacement for the main study.

Table 5. Held-out hybrid comparison by language and budget.

Figure 4. Held-out hybrid delta relative to the best primary method in each cell.

The hybrid wins 8 of 12 cells, with wins at 100, 500, 2,000, and 20,000 examples in both languages, and losses at 50 and 1,000. The outcome motivates systematic hybrid studies but does not justify claiming that the hybrid dominates under a matched tuning budget.

# CHAPTER 6 — CONCLUSION AND FUTURE WORK

## 6.1 Summary

This report audited the project repository and reconstructed its maintained evidence. The primary study is a 108-run validation sweep; the hybrid analysis is a separate held-out follow-up. The corrected interpretation is budget-dependent and explicitly avoids universal superiority claims.

## 6.2 Answer to RQ1

Yes, the ranking changes with training budget. DoRA leads at the smallest Hindi and Telugu budget; IA³ leads in the intermediate range; and the 20,000-example winner differs by language.

## 6.3 Answer to RQ2

No, the ranking is not fully consistent across Hindi and Telugu. The languages agree on IA³ at 500–2,000 but differ at 100 and 20,000, and small-budget cells are unstable.

## 6.4 Practical Recommendations

For an intermediate labelled-data regime in this exact task and backbone, IA³ is a strong baseline candidate. At extremely small budgets, practitioners should inspect seed variability and collapse diagnostics rather than trusting a nominal winner. At larger budgets, language-specific validation remains necessary.

## 6.5 Contributions and Limitations

The study contributes a reproducible budget-by-language comparison, engineering documentation, cost measurements, and a separated hybrid follow-up. It does not establish broad cross-script universality, full-data performance, statistical significance, or causal explanations for every observed ranking.

## 6.6 Future Work

Future work should evaluate more Indic languages and scripts, additional tasks, larger encoders, more seeds, matched optimizer/update budgets, per-language and nested hyperparameter protocols, and systematically controlled hybrid configurations. These are proposals, not completed experiments in this repository.

# APPENDIX A — COMPLETE PRIMARY EXPERIMENTAL RESULTS

Table A1. Complete 108-run primary validation output.

# APPENDIX B — SUMMARY WITH CONFIDENCE INTERVALS

Table B1. Mean, standard deviation, and two-sided t intervals across three seeds.

# APPENDIX C — NEAR-BASELINE RUNS

Table C1. Runs flagged by the repository collapse rule.

# APPENDIX D — REPRODUCIBILITY ARTIFACT INDEX

notebooks/00-environment-setup.ipynb — environment and CUDA checks.

notebooks/01-dataset-validation.ipynb — dataset validation.

notebooks/02-data-preprocessingv2.ipynb — processed subsets.

notebooks/03-experimental-infrastructure-validation.ipynb — adapter attachment, counts, and checks.

notebooks/04-hyperparameter-search.ipynb — pooled LR search.

notebooks/05-full-experiment-sweep.ipynb — 108-run primary sweep.

notebooks/06-results-analysis.ipynb — corrected summaries, ranking, CI, and collapse analysis.

notebooks/07-test-set-evaluation.ipynb — held-out primary evaluation.

notebooks/08-hybrid-formal.ipynb and 09-hybrid-test-evaluation.ipynb — hybrid follow-up.

notebooks/10-final-comparison.ipynb — final cell-wise comparison.

results/05-full-experiment-sweep/experiment_results.csv — raw primary output.

results/06-results-analysis/summary_with_ci.csv and ranking_table.csv — corrected derived outputs.

results/10-final-comparison/hybrid_vs_best_primary.csv — held-out comparison.

requirements.txt, README.md, docs/validated-results.md, docs/PROJECT_STATUS.md — environment and interpretation documentation.

# REFERENCES

[1] V. Kollu, “indic-peft-comparison,” GitHub repository, commit 20a00ef, 2026. https://github.com/venkatkollu/indic-peft-comparison

[2] E. J. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” arXiv:2106.09685, 2021. https://arxiv.org/abs/2106.09685

[3] S.-Y. Liu et al., “DoRA: Weight-Decomposed Low-Rank Adaptation,” Proc. ICML, 2024. https://proceedings.mlr.press/v235/liu24bn.html

[4] H. Liu et al., “Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning,” NeurIPS, 2022. https://arxiv.org/abs/2205.05638

[5] D. Aggarwal, V. Gupta, and A. Kunchukuttan, “IndicXNLI: Evaluating Multilingual Inference for Indian Languages,” EMNLP, pp. 10994–11006, 2022. https://aclanthology.org/2022.emnlp-main.755/

[6] A. Conneau et al., “Unsupervised Cross-lingual Representation Learning at Scale,” ACL, pp. 8440–8451, 2020. https://aclanthology.org/2020.acl-main.747/

# APPENDIX E — BUDGET-WISE SEED-LEVEL ANALYSIS

## Hindi at 50 Training Examples

Table E1. Hindi seed-level primary output at budget 50.

Table E7. Hindi aggregate statistics at budget 50.

Interpretation. At 50 examples, the highest observed mean macro-F1 in Hindi is DoRA. The seed-level table is retained to show the dispersion underlying that mean. In the smallest budgets, near-baseline outcomes and wide intervals make the nominal ordering fragile; at larger budgets, the gap between methods can be interpreted descriptively within this fixed experimental protocol, without converting it into a universal or statistically significant claim.

## Hindi at 100 Training Examples

Table E2. Hindi seed-level primary output at budget 100.

Table E8. Hindi aggregate statistics at budget 100.

Interpretation. At 100 examples, the highest observed mean macro-F1 in Hindi is DoRA. The seed-level table is retained to show the dispersion underlying that mean. In the smallest budgets, near-baseline outcomes and wide intervals make the nominal ordering fragile; at larger budgets, the gap between methods can be interpreted descriptively within this fixed experimental protocol, without converting it into a universal or statistically significant claim.

## Hindi at 500 Training Examples

Table E3. Hindi seed-level primary output at budget 500.

Table E9. Hindi aggregate statistics at budget 500.

Interpretation. At 500 examples, the highest observed mean macro-F1 in Hindi is IA³. The seed-level table is retained to show the dispersion underlying that mean. In the smallest budgets, near-baseline outcomes and wide intervals make the nominal ordering fragile; at larger budgets, the gap between methods can be interpreted descriptively within this fixed experimental protocol, without converting it into a universal or statistically significant claim.

## Hindi at 1,000 Training Examples

Table E4. Hindi seed-level primary output at budget 1,000.

Table E10. Hindi aggregate statistics at budget 1,000.

Interpretation. At 1,000 examples, the highest observed mean macro-F1 in Hindi is IA³. The seed-level table is retained to show the dispersion underlying that mean. In the smallest budgets, near-baseline outcomes and wide intervals make the nominal ordering fragile; at larger budgets, the gap between methods can be interpreted descriptively within this fixed experimental protocol, without converting it into a universal or statistically significant claim.

## Hindi at 2,000 Training Examples

Table E5. Hindi seed-level primary output at budget 2,000.

Table E11. Hindi aggregate statistics at budget 2,000.

Interpretation. At 2,000 examples, the highest observed mean macro-F1 in Hindi is IA³. The seed-level table is retained to show the dispersion underlying that mean. In the smallest budgets, near-baseline outcomes and wide intervals make the nominal ordering fragile; at larger budgets, the gap between methods can be interpreted descriptively within this fixed experimental protocol, without converting it into a universal or statistically significant claim.

## Hindi at 20,000 Training Examples

Table E6. Hindi seed-level primary output at budget 20,000.

Table E12. Hindi aggregate statistics at budget 20,000.

Interpretation. At 20,000 examples, the highest observed mean macro-F1 in Hindi is IA³. The seed-level table is retained to show the dispersion underlying that mean. In the smallest budgets, near-baseline outcomes and wide intervals make the nominal ordering fragile; at larger budgets, the gap between methods can be interpreted descriptively within this fixed experimental protocol, without converting it into a universal or statistically significant claim.

## Telugu at 50 Training Examples

Table E1. Telugu seed-level primary output at budget 50.

Table E7. Telugu aggregate statistics at budget 50.

Interpretation. At 50 examples, the highest observed mean macro-F1 in Telugu is DoRA. The seed-level table is retained to show the dispersion underlying that mean. In the smallest budgets, near-baseline outcomes and wide intervals make the nominal ordering fragile; at larger budgets, the gap between methods can be interpreted descriptively within this fixed experimental protocol, without converting it into a universal or statistically significant claim.

## Telugu at 100 Training Examples

Table E2. Telugu seed-level primary output at budget 100.

Table E8. Telugu aggregate statistics at budget 100.

Interpretation. At 100 examples, the highest observed mean macro-F1 in Telugu is IA³. The seed-level table is retained to show the dispersion underlying that mean. In the smallest budgets, near-baseline outcomes and wide intervals make the nominal ordering fragile; at larger budgets, the gap between methods can be interpreted descriptively within this fixed experimental protocol, without converting it into a universal or statistically significant claim.

## Telugu at 500 Training Examples

Table E3. Telugu seed-level primary output at budget 500.

Table E9. Telugu aggregate statistics at budget 500.

Interpretation. At 500 examples, the highest observed mean macro-F1 in Telugu is IA³. The seed-level table is retained to show the dispersion underlying that mean. In the smallest budgets, near-baseline outcomes and wide intervals make the nominal ordering fragile; at larger budgets, the gap between methods can be interpreted descriptively within this fixed experimental protocol, without converting it into a universal or statistically significant claim.

## Telugu at 1,000 Training Examples

Table E4. Telugu seed-level primary output at budget 1,000.

Table E10. Telugu aggregate statistics at budget 1,000.

Interpretation. At 1,000 examples, the highest observed mean macro-F1 in Telugu is IA³. The seed-level table is retained to show the dispersion underlying that mean. In the smallest budgets, near-baseline outcomes and wide intervals make the nominal ordering fragile; at larger budgets, the gap between methods can be interpreted descriptively within this fixed experimental protocol, without converting it into a universal or statistically significant claim.

## Telugu at 2,000 Training Examples

Table E5. Telugu seed-level primary output at budget 2,000.

Table E11. Telugu aggregate statistics at budget 2,000.

Interpretation. At 2,000 examples, the highest observed mean macro-F1 in Telugu is IA³. The seed-level table is retained to show the dispersion underlying that mean. In the smallest budgets, near-baseline outcomes and wide intervals make the nominal ordering fragile; at larger budgets, the gap between methods can be interpreted descriptively within this fixed experimental protocol, without converting it into a universal or statistically significant claim.

## Telugu at 20,000 Training Examples

Table E6. Telugu seed-level primary output at budget 20,000.

Table E12. Telugu aggregate statistics at budget 20,000.

Interpretation. At 20,000 examples, the highest observed mean macro-F1 in Telugu is LoRA. The seed-level table is retained to show the dispersion underlying that mean. In the smallest budgets, near-baseline outcomes and wide intervals make the nominal ordering fragile; at larger budgets, the gap between methods can be interpreted descriptively within this fixed experimental protocol, without converting it into a universal or statistically significant claim.

# APPENDIX F — IMPLEMENTATION AND REPRODUCIBILITY NOTES

## F.1 Execution sequence

The notebooks form a sequential pipeline: environment setup, dataset validation, preprocessing, infrastructure validation, pooled learning-rate search, primary sweep, corrected analysis, held-out evaluation, hybrid formal experiment, hybrid test evaluation, and final comparison. The output directories preserve the link between each stage and its downstream tables.

## F.2 Configuration excerpt

The primary sweep uses MODEL_NAME = xlm-roberta-base, NUM_LABELS = 3, MAX_LENGTH = 128, BATCH_SIZE = 32, METHODS = [lora, dora, ia3], LANGUAGES = [hi, te], BUDGETS = [50, 100, 500, 1000, 2000, 20000], and SEEDS = [42, 123, 456].

## F.3 Save/load safeguard

The model builder uses modules_to_save=[classifier] and asserts that classifier parameters are trainable. The save routine writes the PEFT adapter and classifier_head.pt separately, while the load routine restores the classifier state with strict=False. This workaround is part of the experiment definition, not an incidental convenience.

## F.4 Data availability

The repository does not include the complete raw dataset. Reproduction therefore requires access to IndicXNLI and reconstruction of the processed parquet subsets. The preserved CSV/JSONL outputs permit result inspection without rerunning training, but exact reruns remain environment- and path-dependent.

## F.5 Interpretation policy

The report treats docs/validated-results.md and corrected result CSVs as authoritative over older report-material drafts. Conflicts are recorded in CHANGELOG.md. Validation results and held-out hybrid results are never merged into one primary table.

# APPENDIX G — DATA INTEGRITY CHECKLIST

Verified: 108 primary rows verified

Verified: 3 methods verified

Verified: 2 languages verified

Verified: 6 budgets verified

Verified: 3 seeds verified

Verified: Macro-F1 and accuracy columns verified

Verified: Ranking winners recomputed from raw primary CSV

Verified: 24 near-baseline runs recomputed with atol=0.001

Verified: Three-seed interval multiplier recorded as 4.302652729

Verified: Hybrid comparison contains 12 cells and 8 wins

Verified: Primary and hybrid experiments kept separate

Verified: No pairwise significance claim made

Verified: All figures generated from stored repository outputs

Verified: Repository commit recorded in audit materials

# APPENDIX H — CELL-WISE ANALYTICAL RECORD

## Hindi — 50-example cell

The observed primary ordering in this cell is DoRA > LoRA > IA³. The means and standard deviations are: DORA mean=0.2213, SD=0.0445, LORA mean=0.2191, SD=0.0415, IA3 mean=0.1922, SD=0.0368.

This cell is interpreted jointly with its three seed outputs, the near-baseline diagnostic, and the confidence interval. The ordering is a descriptive result of the recorded protocol. It does not establish a statistically significant difference, and it should not be extrapolated beyond the tested language, model, task, or budget. The purpose of retaining this record is to make every ranking decision auditable from the underlying runs.

## Hindi — 100-example cell

The observed primary ordering in this cell is DoRA > LoRA > IA³. The means and standard deviations are: DORA mean=0.1993, SD=0.0566, LORA mean=0.1992, SD=0.0563, IA3 mean=0.1755, SD=0.0153.

This cell is interpreted jointly with its three seed outputs, the near-baseline diagnostic, and the confidence interval. The ordering is a descriptive result of the recorded protocol. It does not establish a statistically significant difference, and it should not be extrapolated beyond the tested language, model, task, or budget. The purpose of retaining this record is to make every ranking decision auditable from the underlying runs.

## Hindi — 500-example cell

The observed primary ordering in this cell is IA³ > LoRA > DoRA. The means and standard deviations are: IA3 mean=0.3799, SD=0.0095, LORA mean=0.3359, SD=0.0527, DORA mean=0.3302, SD=0.0472.

This cell is interpreted jointly with its three seed outputs, the near-baseline diagnostic, and the confidence interval. The ordering is a descriptive result of the recorded protocol. It does not establish a statistically significant difference, and it should not be extrapolated beyond the tested language, model, task, or budget. The purpose of retaining this record is to make every ranking decision auditable from the underlying runs.

## Hindi — 1,000-example cell

The observed primary ordering in this cell is IA³ > LoRA > DoRA. The means and standard deviations are: IA3 mean=0.3951, SD=0.0634, LORA mean=0.2035, SD=0.0362, DORA mean=0.2025, SD=0.0380.

This cell is interpreted jointly with its three seed outputs, the near-baseline diagnostic, and the confidence interval. The ordering is a descriptive result of the recorded protocol. It does not establish a statistically significant difference, and it should not be extrapolated beyond the tested language, model, task, or budget. The purpose of retaining this record is to make every ranking decision auditable from the underlying runs.

## Hindi — 2,000-example cell

The observed primary ordering in this cell is IA³ > LoRA > DoRA. The means and standard deviations are: IA3 mean=0.4915, SD=0.0703, LORA mean=0.4850, SD=0.0269, DORA mean=0.4427, SD=0.0623.

This cell is interpreted jointly with its three seed outputs, the near-baseline diagnostic, and the confidence interval. The ordering is a descriptive result of the recorded protocol. It does not establish a statistically significant difference, and it should not be extrapolated beyond the tested language, model, task, or budget. The purpose of retaining this record is to make every ranking decision auditable from the underlying runs.

## Hindi — 20,000-example cell

The observed primary ordering in this cell is IA³ > DoRA > LoRA. The means and standard deviations are: IA3 mean=0.6682, SD=0.0120, DORA mean=0.6658, SD=0.0057, LORA mean=0.6641, SD=0.0123.

This cell is interpreted jointly with its three seed outputs, the near-baseline diagnostic, and the confidence interval. The ordering is a descriptive result of the recorded protocol. It does not establish a statistically significant difference, and it should not be extrapolated beyond the tested language, model, task, or budget. The purpose of retaining this record is to make every ranking decision auditable from the underlying runs.

## Telugu — 50-example cell

The observed primary ordering in this cell is DoRA > LoRA > IA³. The means and standard deviations are: DORA mean=0.2053, SD=0.0269, LORA mean=0.2041, SD=0.0255, IA3 mean=0.1667, SD=0.0000.

This cell is interpreted jointly with its three seed outputs, the near-baseline diagnostic, and the confidence interval. The ordering is a descriptive result of the recorded protocol. It does not establish a statistically significant difference, and it should not be extrapolated beyond the tested language, model, task, or budget. The purpose of retaining this record is to make every ranking decision auditable from the underlying runs.

## Telugu — 100-example cell

The observed primary ordering in this cell is IA³ > DoRA > LoRA. The means and standard deviations are: IA3 mean=0.2063, SD=0.0686, DORA mean=0.2044, SD=0.0654, LORA mean=0.2038, SD=0.0643.

This cell is interpreted jointly with its three seed outputs, the near-baseline diagnostic, and the confidence interval. The ordering is a descriptive result of the recorded protocol. It does not establish a statistically significant difference, and it should not be extrapolated beyond the tested language, model, task, or budget. The purpose of retaining this record is to make every ranking decision auditable from the underlying runs.

## Telugu — 500-example cell

The observed primary ordering in this cell is IA³ > LoRA > DoRA. The means and standard deviations are: IA3 mean=0.3512, SD=0.0428, LORA mean=0.2872, SD=0.0641, DORA mean=0.2777, SD=0.0554.

This cell is interpreted jointly with its three seed outputs, the near-baseline diagnostic, and the confidence interval. The ordering is a descriptive result of the recorded protocol. It does not establish a statistically significant difference, and it should not be extrapolated beyond the tested language, model, task, or budget. The purpose of retaining this record is to make every ranking decision auditable from the underlying runs.

## Telugu — 1,000-example cell

The observed primary ordering in this cell is IA³ > DoRA > LoRA. The means and standard deviations are: IA3 mean=0.2678, SD=0.0900, DORA mean=0.2052, SD=0.0656, LORA mean=0.2039, SD=0.0627.

This cell is interpreted jointly with its three seed outputs, the near-baseline diagnostic, and the confidence interval. The ordering is a descriptive result of the recorded protocol. It does not establish a statistically significant difference, and it should not be extrapolated beyond the tested language, model, task, or budget. The purpose of retaining this record is to make every ranking decision auditable from the underlying runs.

## Telugu — 2,000-example cell

The observed primary ordering in this cell is IA³ > DoRA > LoRA. The means and standard deviations are: IA3 mean=0.3949, SD=0.0617, DORA mean=0.3155, SD=0.0634, LORA mean=0.2404, SD=0.0870.

This cell is interpreted jointly with its three seed outputs, the near-baseline diagnostic, and the confidence interval. The ordering is a descriptive result of the recorded protocol. It does not establish a statistically significant difference, and it should not be extrapolated beyond the tested language, model, task, or budget. The purpose of retaining this record is to make every ranking decision auditable from the underlying runs.

## Telugu — 20,000-example cell

The observed primary ordering in this cell is LoRA > DoRA > IA³. The means and standard deviations are: LORA mean=0.6262, SD=0.0119, DORA mean=0.6227, SD=0.0084, IA3 mean=0.6165, SD=0.0107.

This cell is interpreted jointly with its three seed outputs, the near-baseline diagnostic, and the confidence interval. The ordering is a descriptive result of the recorded protocol. It does not establish a statistically significant difference, and it should not be extrapolated beyond the tested language, model, task, or budget. The purpose of retaining this record is to make every ranking decision auditable from the underlying runs.


**Table 1.**

| Method | Trainable parameters | Learning rate | Targets | Classifier saved |
| --- | --- | --- | --- | --- |
| LoRA | 887811 | 1e-4 | query, value | Yes |
| DoRA | 906243 | 1e-4 | query, value | Yes |
| IA³ | 657411 | 5e-3 | key, value, output.dense | Yes |


**Table 2.**

| Factor | Values |
| --- | --- |
| Methods | LoRA, DoRA, IA³ |
| Languages | Hindi (hi), Telugu (te) |
| Budgets | 50, 100, 500, 1,000, 2,000, 20,000 |
| Seeds | 42, 123, 456 |
| Task | Three-class NLI |
| Metric | Macro-F1 |


**Table 3.**

| method | trainable_params | avg_forward_pass_memory_gb | total_training_time_sec | total_training_time_min |
| --- | --- | --- | --- | --- |
| dora | 906243 | 1.1721 | 3213.2500 | 53.5542 |
| ia3 | 657411 | 1.2402 | 2864.7400 | 47.7457 |
| lora | 887811 | 1.1567 | 2323.6600 | 38.7277 |


**Table 4.**

| language | budget | lora | dora | ia3 | highest_mean_method |
| --- | --- | --- | --- | --- | --- |
| hi | 50 | 0.2191 | 0.2213 | 0.1922 | dora |
| hi | 100 | 0.1992 | 0.1993 | 0.1755 | dora |
| hi | 500 | 0.3359 | 0.3302 | 0.3799 | ia3 |
| hi | 1000 | 0.2035 | 0.2025 | 0.3951 | ia3 |
| hi | 2000 | 0.4850 | 0.4427 | 0.4915 | ia3 |
| hi | 20000 | 0.6641 | 0.6658 | 0.6682 | ia3 |
| te | 50 | 0.2041 | 0.2053 | 0.1667 | dora |
| te | 100 | 0.2038 | 0.2044 | 0.2063 | ia3 |
| te | 500 | 0.2872 | 0.2777 | 0.3512 | ia3 |
| te | 1000 | 0.2039 | 0.2052 | 0.2678 | ia3 |
| te | 2000 | 0.2404 | 0.3155 | 0.3949 | ia3 |
| te | 20000 | 0.6262 | 0.6227 | 0.6165 | lora |


**Table 5.**

| language | budget | best_primary_method | best_primary | hybrid_ia3_lora | hybrid_delta | hybrid_wins |
| --- | --- | --- | --- | --- | --- | --- |
| hi | 50 | dora | 0.2211 | 0.1667 | -0.0544 | False |
| hi | 100 | lora | 0.1987 | 0.2758 | 0.0771 | True |
| hi | 500 | ia3 | 0.3756 | 0.4361 | 0.0604 | True |
| hi | 1000 | ia3 | 0.4012 | 0.3492 | -0.0520 | False |
| hi | 2000 | ia3 | 0.4901 | 0.5425 | 0.0524 | True |
| hi | 20000 | ia3 | 0.6700 | 0.6913 | 0.0213 | True |
| te | 50 | lora | 0.2026 | 0.1667 | -0.0360 | False |
| te | 100 | lora | 0.2033 | 0.2754 | 0.0722 | True |
| te | 500 | ia3 | 0.3539 | 0.4060 | 0.0522 | True |
| te | 1000 | ia3 | 0.2635 | 0.2577 | -0.0058 | False |
| te | 2000 | ia3 | 0.4010 | 0.4752 | 0.0742 | True |
| te | 20000 | lora | 0.6280 | 0.6539 | 0.0260 | True |


**Table 6.**

| method | language | budget | seed | lr | epochs | accuracy | macro_f1 | trainable_params | peak_gpu_memory_gb | training_time_sec | adapter_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora | hi | 50 | 42 | 0.0001 | 10 | 0.3361 | 0.2543 | 887811 | 1.1563 | 2.2500 | /kaggle/working/adapters/lora/hi/budget50_seed42 |
| lora | hi | 50 | 123 | 0.0001 | 10 | 0.3273 | 0.2295 | 887811 | 1.1563 | 2.2800 | /kaggle/working/adapters/lora/hi/budget50_seed123 |
| lora | hi | 50 | 456 | 0.0001 | 10 | 0.3361 | 0.1733 | 887811 | 1.1556 | 2.3300 | /kaggle/working/adapters/lora/hi/budget50_seed456 |
| lora | hi | 100 | 42 | 0.0001 | 10 | 0.3333 | 0.1667 | 887811 | 1.1563 | 4.8000 | /kaggle/working/adapters/lora/hi/budget100_seed42 |
| lora | hi | 100 | 123 | 0.0001 | 10 | 0.3333 | 0.1667 | 887811 | 1.1578 | 5.0100 | /kaggle/working/adapters/lora/hi/budget100_seed123 |
| lora | hi | 100 | 456 | 0.0001 | 10 | 0.3398 | 0.2641 | 887811 | 1.1571 | 5.1600 | /kaggle/working/adapters/lora/hi/budget100_seed456 |
| lora | hi | 500 | 42 | 0.0001 | 10 | 0.3606 | 0.2771 | 887811 | 1.1571 | 23.7600 | /kaggle/working/adapters/lora/hi/budget500_seed42 |
| lora | hi | 500 | 123 | 0.0001 | 10 | 0.3847 | 0.3519 | 887811 | 1.1563 | 23.5100 | /kaggle/working/adapters/lora/hi/budget500_seed123 |
| lora | hi | 500 | 456 | 0.0001 | 10 | 0.3791 | 0.3788 | 887811 | 1.1563 | 24.0500 | /kaggle/working/adapters/lora/hi/budget500_seed456 |
| lora | hi | 1000 | 42 | 0.0001 | 5 | 0.3414 | 0.2032 | 887811 | 1.1563 | 23.5500 | /kaggle/working/adapters/lora/hi/budget1000_seed42 |
| lora | hi | 1000 | 123 | 0.0001 | 5 | 0.3337 | 0.1675 | 887811 | 1.1571 | 23.8700 | /kaggle/working/adapters/lora/hi/budget1000_seed123 |
| lora | hi | 1000 | 456 | 0.0001 | 5 | 0.3341 | 0.2399 | 887811 | 1.1578 | 23.8100 | /kaggle/working/adapters/lora/hi/budget1000_seed456 |
| lora | hi | 2000 | 42 | 0.0001 | 5 | 0.4767 | 0.4738 | 887811 | 1.1563 | 47.7000 | /kaggle/working/adapters/lora/hi/budget2000_seed42 |
| lora | hi | 2000 | 123 | 0.0001 | 5 | 0.5141 | 0.5158 | 887811 | 1.1563 | 47.7700 | /kaggle/working/adapters/lora/hi/budget2000_seed123 |
| lora | hi | 2000 | 456 | 0.0001 | 5 | 0.4839 | 0.4655 | 887811 | 1.1563 | 47.5300 | /kaggle/working/adapters/lora/hi/budget2000_seed456 |
| lora | hi | 20000 | 42 | 0.0001 | 3 | 0.6502 | 0.6516 | 887811 | 1.1563 | 284.3300 | /kaggle/working/adapters/lora/hi/budget20000_seed42 |
| lora | hi | 20000 | 123 | 0.0001 | 3 | 0.6635 | 0.6644 | 887811 | 1.1571 | 284.5400 | /kaggle/working/adapters/lora/hi/budget20000_seed123 |
| lora | hi | 20000 | 456 | 0.0001 | 3 | 0.6755 | 0.6762 | 887811 | 1.1563 | 284.9600 | /kaggle/working/adapters/lora/hi/budget20000_seed456 |
| lora | te | 50 | 42 | 0.0001 | 10 | 0.3438 | 0.2148 | 887811 | 1.1571 | 2.3900 | /kaggle/working/adapters/lora/te/budget50_seed42 |
| lora | te | 50 | 123 | 0.0001 | 10 | 0.3217 | 0.2224 | 887811 | 1.1556 | 2.4100 | /kaggle/working/adapters/lora/te/budget50_seed123 |
| lora | te | 50 | 456 | 0.0001 | 10 | 0.3365 | 0.1750 | 887811 | 1.1563 | 2.4100 | /kaggle/working/adapters/lora/te/budget50_seed456 |
| lora | te | 100 | 42 | 0.0001 | 10 | 0.3333 | 0.1667 | 887811 | 1.1578 | 5.0100 | /kaggle/working/adapters/lora/te/budget100_seed42 |
| lora | te | 100 | 123 | 0.0001 | 10 | 0.3333 | 0.1667 | 887811 | 1.1571 | 5.0100 | /kaggle/working/adapters/lora/te/budget100_seed123 |
| lora | te | 100 | 456 | 0.0001 | 10 | 0.3478 | 0.2781 | 887811 | 1.1578 | 4.9500 | /kaggle/working/adapters/lora/te/budget100_seed456 |
| lora | te | 500 | 42 | 0.0001 | 10 | 0.3518 | 0.2529 | 887811 | 1.1563 | 23.8900 | /kaggle/working/adapters/lora/te/budget500_seed42 |
| lora | te | 500 | 123 | 0.0001 | 10 | 0.3715 | 0.3612 | 887811 | 1.1571 | 23.7400 | /kaggle/working/adapters/lora/te/budget500_seed123 |
| lora | te | 500 | 456 | 0.0001 | 10 | 0.3522 | 0.2476 | 887811 | 1.1563 | 23.7700 | /kaggle/working/adapters/lora/te/budget500_seed456 |
| lora | te | 1000 | 42 | 0.0001 | 5 | 0.3325 | 0.1687 | 887811 | 1.1571 | 23.7800 | /kaggle/working/adapters/lora/te/budget1000_seed42 |
| lora | te | 1000 | 123 | 0.0001 | 5 | 0.3333 | 0.1667 | 887811 | 1.1571 | 23.7600 | /kaggle/working/adapters/lora/te/budget1000_seed123 |
| lora | te | 1000 | 456 | 0.0001 | 5 | 0.3474 | 0.2763 | 887811 | 1.1578 | 23.8400 | /kaggle/working/adapters/lora/te/budget1000_seed456 |
| lora | te | 2000 | 42 | 0.0001 | 5 | 0.3398 | 0.2009 | 887811 | 1.1563 | 47.5600 | /kaggle/working/adapters/lora/te/budget2000_seed42 |
| lora | te | 2000 | 123 | 0.0001 | 5 | 0.3373 | 0.1801 | 887811 | 1.1563 | 47.8100 | /kaggle/working/adapters/lora/te/budget2000_seed123 |
| lora | te | 2000 | 456 | 0.0001 | 5 | 0.3932 | 0.3402 | 887811 | 1.1563 | 47.5800 | /kaggle/working/adapters/lora/te/budget2000_seed456 |
| lora | te | 20000 | 42 | 0.0001 | 3 | 0.6112 | 0.6125 | 887811 | 1.1563 | 284.3500 | /kaggle/working/adapters/lora/te/budget20000_seed42 |
| lora | te | 20000 | 123 | 0.0001 | 3 | 0.6337 | 0.6339 | 887811 | 1.1571 | 285.0700 | /kaggle/working/adapters/lora/te/budget20000_seed123 |
| lora | te | 20000 | 456 | 0.0001 | 3 | 0.6341 | 0.6322 | 887811 | 1.1563 | 285.1200 | /kaggle/working/adapters/lora/te/budget20000_seed456 |
| dora | hi | 50 | 42 | 0.0001 | 10 | 0.3378 | 0.2613 | 906243 | 1.1721 | 3.3500 | /kaggle/working/adapters/dora/hi/budget50_seed42 |
| dora | hi | 50 | 123 | 0.0001 | 10 | 0.3269 | 0.2293 | 906243 | 1.1721 | 3.3300 | /kaggle/working/adapters/dora/hi/budget50_seed123 |
| dora | hi | 50 | 456 | 0.0001 | 10 | 0.3361 | 0.1733 | 906243 | 1.1721 | 3.3800 | /kaggle/working/adapters/dora/hi/budget50_seed456 |
| dora | hi | 100 | 42 | 0.0001 | 10 | 0.3333 | 0.1667 | 906243 | 1.1720 | 7.0000 | /kaggle/working/adapters/dora/hi/budget100_seed42 |
| dora | hi | 100 | 123 | 0.0001 | 10 | 0.3333 | 0.1667 | 906243 | 1.1720 | 6.9200 | /kaggle/working/adapters/dora/hi/budget100_seed123 |
| dora | hi | 100 | 456 | 0.0001 | 10 | 0.3406 | 0.2647 | 906243 | 1.1720 | 6.8700 | /kaggle/working/adapters/dora/hi/budget100_seed456 |
| dora | hi | 500 | 42 | 0.0001 | 10 | 0.3586 | 0.2861 | 906243 | 1.1723 | 33.3400 | /kaggle/working/adapters/dora/hi/budget500_seed42 |
| dora | hi | 500 | 123 | 0.0001 | 10 | 0.3811 | 0.3245 | 906243 | 1.1723 | 32.5200 | /kaggle/working/adapters/dora/hi/budget500_seed123 |
| dora | hi | 500 | 456 | 0.0001 | 10 | 0.3811 | 0.3800 | 906243 | 1.1723 | 33.0700 | /kaggle/working/adapters/dora/hi/budget500_seed456 |
| dora | hi | 1000 | 42 | 0.0001 | 5 | 0.3406 | 0.1985 | 906243 | 1.1734 | 32.8200 | /kaggle/working/adapters/dora/hi/budget1000_seed42 |
| dora | hi | 1000 | 123 | 0.0001 | 5 | 0.3333 | 0.1667 | 906243 | 1.1734 | 32.9100 | /kaggle/working/adapters/dora/hi/budget1000_seed123 |
| dora | hi | 1000 | 456 | 0.0001 | 5 | 0.3349 | 0.2424 | 906243 | 1.1734 | 33.0200 | /kaggle/working/adapters/dora/hi/budget1000_seed456 |
| dora | hi | 2000 | 42 | 0.0001 | 5 | 0.4554 | 0.4087 | 906243 | 1.1727 | 65.6700 | /kaggle/working/adapters/dora/hi/budget2000_seed42 |
| dora | hi | 2000 | 123 | 0.0001 | 5 | 0.5133 | 0.5146 | 906243 | 1.1727 | 65.7400 | /kaggle/working/adapters/dora/hi/budget2000_seed123 |
| dora | hi | 2000 | 456 | 0.0001 | 5 | 0.4643 | 0.4049 | 906243 | 1.1727 | 65.9300 | /kaggle/working/adapters/dora/hi/budget2000_seed456 |
| dora | hi | 20000 | 42 | 0.0001 | 3 | 0.6703 | 0.6719 | 906243 | 1.1727 | 393.5700 | /kaggle/working/adapters/dora/hi/budget20000_seed42 |
| dora | hi | 20000 | 123 | 0.0001 | 3 | 0.6590 | 0.6605 | 906243 | 1.1705 | 393.4900 | /kaggle/working/adapters/dora/hi/budget20000_seed123 |
| dora | hi | 20000 | 456 | 0.0001 | 3 | 0.6659 | 0.6650 | 906243 | 1.1698 | 394.2300 | /kaggle/working/adapters/dora/hi/budget20000_seed456 |
| dora | te | 50 | 42 | 0.0001 | 10 | 0.3430 | 0.2146 | 906243 | 1.1699 | 3.3300 | /kaggle/working/adapters/dora/te/budget50_seed42 |
| dora | te | 50 | 123 | 0.0001 | 10 | 0.3253 | 0.2263 | 906243 | 1.1705 | 3.3700 | /kaggle/working/adapters/dora/te/budget50_seed123 |
| dora | te | 50 | 456 | 0.0001 | 10 | 0.3365 | 0.1750 | 906243 | 1.1721 | 3.4400 | /kaggle/working/adapters/dora/te/budget50_seed456 |
| dora | te | 100 | 42 | 0.0001 | 10 | 0.3333 | 0.1667 | 906243 | 1.1720 | 6.9400 | /kaggle/working/adapters/dora/te/budget100_seed42 |
| dora | te | 100 | 123 | 0.0001 | 10 | 0.3333 | 0.1667 | 906243 | 1.1720 | 6.9000 | /kaggle/working/adapters/dora/te/budget100_seed123 |
| dora | te | 100 | 456 | 0.0001 | 10 | 0.3502 | 0.2799 | 906243 | 1.1720 | 6.9100 | /kaggle/working/adapters/dora/te/budget100_seed456 |
| dora | te | 500 | 42 | 0.0001 | 10 | 0.3522 | 0.2586 | 906243 | 1.1723 | 33.2100 | /kaggle/working/adapters/dora/te/budget500_seed42 |
| dora | te | 500 | 123 | 0.0001 | 10 | 0.3715 | 0.3401 | 906243 | 1.1723 | 32.7300 | /kaggle/working/adapters/dora/te/budget500_seed123 |
| dora | te | 500 | 456 | 0.0001 | 10 | 0.3498 | 0.2344 | 906243 | 1.1723 | 32.9000 | /kaggle/working/adapters/dora/te/budget500_seed456 |
| dora | te | 1000 | 42 | 0.0001 | 5 | 0.3325 | 0.1679 | 906243 | 1.1734 | 33.0100 | /kaggle/working/adapters/dora/te/budget1000_seed42 |
| dora | te | 1000 | 123 | 0.0001 | 5 | 0.3333 | 0.1667 | 906243 | 1.1734 | 32.7500 | /kaggle/working/adapters/dora/te/budget1000_seed123 |
| dora | te | 1000 | 456 | 0.0001 | 5 | 0.3514 | 0.2809 | 906243 | 1.1734 | 33.0500 | /kaggle/working/adapters/dora/te/budget1000_seed456 |
| dora | te | 2000 | 42 | 0.0001 | 5 | 0.3859 | 0.3642 | 906243 | 1.1727 | 65.6800 | /kaggle/working/adapters/dora/te/budget2000_seed42 |
| dora | te | 2000 | 123 | 0.0001 | 5 | 0.3530 | 0.2438 | 906243 | 1.1727 | 65.7700 | /kaggle/working/adapters/dora/te/budget2000_seed123 |
| dora | te | 2000 | 456 | 0.0001 | 5 | 0.3747 | 0.3384 | 906243 | 1.1727 | 65.8700 | /kaggle/working/adapters/dora/te/budget2000_seed456 |
| dora | te | 20000 | 42 | 0.0001 | 3 | 0.6197 | 0.6209 | 906243 | 1.1727 | 392.9000 | /kaggle/working/adapters/dora/te/budget20000_seed42 |
| dora | te | 20000 | 123 | 0.0001 | 3 | 0.6317 | 0.6318 | 906243 | 1.1705 | 394.0900 | /kaggle/working/adapters/dora/te/budget20000_seed123 |
| dora | te | 20000 | 456 | 0.0001 | 3 | 0.6185 | 0.6153 | 906243 | 1.1698 | 393.2400 | /kaggle/working/adapters/dora/te/budget20000_seed456 |
| ia3 | hi | 50 | 42 | 0.0050 | 10 | 0.3301 | 0.2344 | 657411 | 1.2395 | 2.9400 | /kaggle/working/adapters/ia3/hi/budget50_seed42 |
| ia3 | hi | 50 | 123 | 0.0050 | 10 | 0.3341 | 0.1755 | 657411 | 1.2395 | 2.9800 | /kaggle/working/adapters/ia3/hi/budget50_seed123 |
| ia3 | hi | 50 | 456 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2395 | 3.0400 | /kaggle/working/adapters/ia3/hi/budget50_seed456 |
| ia3 | hi | 100 | 42 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2402 | 6.0100 | /kaggle/working/adapters/ia3/hi/budget100_seed42 |
| ia3 | hi | 100 | 123 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2402 | 6.1900 | /kaggle/working/adapters/ia3/hi/budget100_seed123 |
| ia3 | hi | 100 | 456 | 0.0050 | 10 | 0.3345 | 0.1932 | 657411 | 1.2402 | 5.9300 | /kaggle/working/adapters/ia3/hi/budget100_seed456 |
| ia3 | hi | 500 | 42 | 0.0050 | 10 | 0.4141 | 0.3835 | 657411 | 1.2395 | 29.7500 | /kaggle/working/adapters/ia3/hi/budget500_seed42 |
| ia3 | hi | 500 | 123 | 0.0050 | 10 | 0.4112 | 0.3691 | 657411 | 1.2395 | 29.0400 | /kaggle/working/adapters/ia3/hi/budget500_seed123 |
| ia3 | hi | 500 | 456 | 0.0050 | 10 | 0.4169 | 0.3870 | 657411 | 1.2395 | 29.5700 | /kaggle/working/adapters/ia3/hi/budget500_seed456 |
| ia3 | hi | 1000 | 42 | 0.0050 | 5 | 0.4691 | 0.4416 | 657411 | 1.2402 | 29.2600 | /kaggle/working/adapters/ia3/hi/budget1000_seed42 |
| ia3 | hi | 1000 | 123 | 0.0050 | 5 | 0.4197 | 0.3229 | 657411 | 1.2402 | 29.2600 | /kaggle/working/adapters/ia3/hi/budget1000_seed123 |
| ia3 | hi | 1000 | 456 | 0.0050 | 5 | 0.4414 | 0.4209 | 657411 | 1.2402 | 29.4100 | /kaggle/working/adapters/ia3/hi/budget1000_seed456 |
| ia3 | hi | 2000 | 42 | 0.0050 | 5 | 0.4663 | 0.4169 | 657411 | 1.2395 | 58.6000 | /kaggle/working/adapters/ia3/hi/budget2000_seed42 |
| ia3 | hi | 2000 | 123 | 0.0050 | 5 | 0.5233 | 0.5010 | 657411 | 1.2409 | 58.6800 | /kaggle/working/adapters/ia3/hi/budget2000_seed123 |
| ia3 | hi | 2000 | 456 | 0.0050 | 5 | 0.5602 | 0.5565 | 657411 | 1.2417 | 58.7800 | /kaggle/working/adapters/ia3/hi/budget2000_seed456 |
| ia3 | hi | 20000 | 42 | 0.0050 | 3 | 0.6534 | 0.6554 | 657411 | 1.2402 | 350.8800 | /kaggle/working/adapters/ia3/hi/budget20000_seed42 |
| ia3 | hi | 20000 | 123 | 0.0050 | 3 | 0.6675 | 0.6698 | 657411 | 1.2409 | 350.6400 | /kaggle/working/adapters/ia3/hi/budget20000_seed123 |
| ia3 | hi | 20000 | 456 | 0.0050 | 3 | 0.6779 | 0.6794 | 657411 | 1.2402 | 350.8400 | /kaggle/working/adapters/ia3/hi/budget20000_seed456 |
| ia3 | te | 50 | 42 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2409 | 2.9500 | /kaggle/working/adapters/ia3/te/budget50_seed42 |
| ia3 | te | 50 | 123 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2402 | 2.9800 | /kaggle/working/adapters/ia3/te/budget50_seed123 |
| ia3 | te | 50 | 456 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2402 | 3.0200 | /kaggle/working/adapters/ia3/te/budget50_seed456 |
| ia3 | te | 100 | 42 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2409 | 6.0600 | /kaggle/working/adapters/ia3/te/budget100_seed42 |
| ia3 | te | 100 | 123 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2409 | 5.9600 | /kaggle/working/adapters/ia3/te/budget100_seed123 |
| ia3 | te | 100 | 456 | 0.0050 | 10 | 0.3683 | 0.2855 | 657411 | 1.2402 | 5.9600 | /kaggle/working/adapters/ia3/te/budget100_seed456 |
| ia3 | te | 500 | 42 | 0.0050 | 10 | 0.3815 | 0.3261 | 657411 | 1.2395 | 29.7000 | /kaggle/working/adapters/ia3/te/budget500_seed42 |
| ia3 | te | 500 | 123 | 0.0050 | 10 | 0.3948 | 0.3268 | 657411 | 1.2395 | 29.0100 | /kaggle/working/adapters/ia3/te/budget500_seed123 |
| ia3 | te | 500 | 456 | 0.0050 | 10 | 0.4137 | 0.4006 | 657411 | 1.2395 | 29.5600 | /kaggle/working/adapters/ia3/te/budget500_seed456 |
| ia3 | te | 1000 | 42 | 0.0050 | 5 | 0.3727 | 0.2691 | 657411 | 1.2402 | 29.2200 | /kaggle/working/adapters/ia3/te/budget1000_seed42 |
| ia3 | te | 1000 | 123 | 0.0050 | 5 | 0.4133 | 0.3571 | 657411 | 1.2402 | 29.3800 | /kaggle/working/adapters/ia3/te/budget1000_seed123 |
| ia3 | te | 1000 | 456 | 0.0050 | 5 | 0.3345 | 0.1771 | 657411 | 1.2402 | 29.3200 | /kaggle/working/adapters/ia3/te/budget1000_seed456 |
| ia3 | te | 2000 | 42 | 0.0050 | 5 | 0.4321 | 0.3460 | 657411 | 1.2395 | 58.6400 | /kaggle/working/adapters/ia3/te/budget2000_seed42 |
| ia3 | te | 2000 | 123 | 0.0050 | 5 | 0.4382 | 0.3746 | 657411 | 1.2409 | 58.6800 | /kaggle/working/adapters/ia3/te/budget2000_seed123 |
| ia3 | te | 2000 | 456 | 0.0050 | 5 | 0.4699 | 0.4643 | 657411 | 1.2417 | 58.6100 | /kaggle/working/adapters/ia3/te/budget2000_seed456 |
| ia3 | te | 20000 | 42 | 0.0050 | 3 | 0.6197 | 0.6207 | 657411 | 1.2402 | 351.3200 | /kaggle/working/adapters/ia3/te/budget20000_seed42 |
| ia3 | te | 20000 | 123 | 0.0050 | 3 | 0.6233 | 0.6244 | 657411 | 1.2409 | 351.4100 | /kaggle/working/adapters/ia3/te/budget20000_seed123 |
| ia3 | te | 20000 | 456 | 0.0050 | 3 | 0.6068 | 0.6043 | 657411 | 1.2402 | 351.1600 | /kaggle/working/adapters/ia3/te/budget20000_seed456 |


**Table 7.**

| method | language | budget | f1_mean | f1_std | acc_mean | acc_std | n | f1_ci95 | acc_ci95 | ci_method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dora | hi | 50 | 0.2213 | 0.0445 | 0.3336 | 0.0059 | 3 | 0.1106 | 0.0145 | two-sided t interval; df=2; t*=4.302652729 |
| dora | hi | 100 | 0.1993 | 0.0566 | 0.3357 | 0.0042 | 3 | 0.1405 | 0.0104 | two-sided t interval; df=2; t*=4.302652729 |
| dora | hi | 500 | 0.3302 | 0.0472 | 0.3736 | 0.0130 | 3 | 0.1172 | 0.0323 | two-sided t interval; df=2; t*=4.302652729 |
| dora | hi | 1000 | 0.2025 | 0.0380 | 0.3363 | 0.0038 | 3 | 0.0944 | 0.0094 | two-sided t interval; df=2; t*=4.302652729 |
| dora | hi | 2000 | 0.4427 | 0.0623 | 0.4776 | 0.0312 | 3 | 0.1547 | 0.0774 | two-sided t interval; df=2; t*=4.302652729 |
| dora | hi | 20000 | 0.6658 | 0.0057 | 0.6651 | 0.0057 | 3 | 0.0142 | 0.0141 | two-sided t interval; df=2; t*=4.302652729 |
| ia3 | hi | 50 | 0.1922 | 0.0368 | 0.3325 | 0.0021 | 3 | 0.0914 | 0.0053 | two-sided t interval; df=2; t*=4.302652729 |
| ia3 | hi | 100 | 0.1755 | 0.0153 | 0.3337 | 0.0007 | 3 | 0.0380 | 0.0017 | two-sided t interval; df=2; t*=4.302652729 |
| ia3 | hi | 500 | 0.3799 | 0.0095 | 0.4141 | 0.0028 | 3 | 0.0236 | 0.0070 | two-sided t interval; df=2; t*=4.302652729 |
| ia3 | hi | 1000 | 0.3951 | 0.0634 | 0.4434 | 0.0248 | 3 | 0.1576 | 0.0615 | two-sided t interval; df=2; t*=4.302652729 |
| ia3 | hi | 2000 | 0.4915 | 0.0703 | 0.5166 | 0.0473 | 3 | 0.1746 | 0.1176 | two-sided t interval; df=2; t*=4.302652729 |
| ia3 | hi | 20000 | 0.6682 | 0.0120 | 0.6663 | 0.0123 | 3 | 0.0299 | 0.0305 | two-sided t interval; df=2; t*=4.302652729 |
| lora | hi | 50 | 0.2191 | 0.0415 | 0.3332 | 0.0051 | 3 | 0.1031 | 0.0127 | two-sided t interval; df=2; t*=4.302652729 |
| lora | hi | 100 | 0.1992 | 0.0563 | 0.3355 | 0.0037 | 3 | 0.1398 | 0.0092 | two-sided t interval; df=2; t*=4.302652729 |
| lora | hi | 500 | 0.3359 | 0.0527 | 0.3748 | 0.0126 | 3 | 0.1310 | 0.0313 | two-sided t interval; df=2; t*=4.302652729 |
| lora | hi | 1000 | 0.2035 | 0.0362 | 0.3364 | 0.0043 | 3 | 0.0899 | 0.0107 | two-sided t interval; df=2; t*=4.302652729 |
| lora | hi | 2000 | 0.4850 | 0.0269 | 0.4916 | 0.0198 | 3 | 0.0669 | 0.0492 | two-sided t interval; df=2; t*=4.302652729 |
| lora | hi | 20000 | 0.6641 | 0.0123 | 0.6631 | 0.0127 | 3 | 0.0306 | 0.0314 | two-sided t interval; df=2; t*=4.302652729 |
| dora | te | 50 | 0.2053 | 0.0269 | 0.3349 | 0.0089 | 3 | 0.0668 | 0.0222 | two-sided t interval; df=2; t*=4.302652729 |
| dora | te | 100 | 0.2044 | 0.0654 | 0.3390 | 0.0097 | 3 | 0.1624 | 0.0242 | two-sided t interval; df=2; t*=4.302652729 |
| dora | te | 500 | 0.2777 | 0.0554 | 0.3578 | 0.0119 | 3 | 0.1375 | 0.0295 | two-sided t interval; df=2; t*=4.302652729 |
| dora | te | 1000 | 0.2052 | 0.0656 | 0.3391 | 0.0107 | 3 | 0.1630 | 0.0265 | two-sided t interval; df=2; t*=4.302652729 |
| dora | te | 2000 | 0.3155 | 0.0634 | 0.3712 | 0.0167 | 3 | 0.1575 | 0.0416 | two-sided t interval; df=2; t*=4.302652729 |
| dora | te | 20000 | 0.6227 | 0.0084 | 0.6233 | 0.0073 | 3 | 0.0209 | 0.0182 | two-sided t interval; df=2; t*=4.302652729 |
| ia3 | te | 50 | 0.1667 | 0.0000 | 0.3333 | 0.0000 | 3 | 0.0000 | 0.0000 | two-sided t interval; df=2; t*=4.302652729 |
| ia3 | te | 100 | 0.2063 | 0.0686 | 0.3450 | 0.0202 | 3 | 0.1704 | 0.0501 | two-sided t interval; df=2; t*=4.302652729 |
| ia3 | te | 500 | 0.3512 | 0.0428 | 0.3967 | 0.0161 | 3 | 0.1063 | 0.0401 | two-sided t interval; df=2; t*=4.302652729 |
| ia3 | te | 1000 | 0.2678 | 0.0900 | 0.3735 | 0.0394 | 3 | 0.2236 | 0.0978 | two-sided t interval; df=2; t*=4.302652729 |
| ia3 | te | 2000 | 0.3949 | 0.0617 | 0.4467 | 0.0203 | 3 | 0.1534 | 0.0504 | two-sided t interval; df=2; t*=4.302652729 |
| ia3 | te | 20000 | 0.6165 | 0.0107 | 0.6166 | 0.0087 | 3 | 0.0266 | 0.0215 | two-sided t interval; df=2; t*=4.302652729 |
| lora | te | 50 | 0.2041 | 0.0255 | 0.3340 | 0.0113 | 3 | 0.0632 | 0.0280 | two-sided t interval; df=2; t*=4.302652729 |
| lora | te | 100 | 0.2038 | 0.0643 | 0.3382 | 0.0083 | 3 | 0.1598 | 0.0207 | two-sided t interval; df=2; t*=4.302652729 |
| lora | te | 500 | 0.2872 | 0.0641 | 0.3585 | 0.0112 | 3 | 0.1593 | 0.0279 | two-sided t interval; df=2; t*=4.302652729 |
| lora | te | 1000 | 0.2039 | 0.0627 | 0.3378 | 0.0084 | 3 | 0.1558 | 0.0208 | two-sided t interval; df=2; t*=4.302652729 |
| lora | te | 2000 | 0.2404 | 0.0870 | 0.3568 | 0.0316 | 3 | 0.2162 | 0.0784 | two-sided t interval; df=2; t*=4.302652729 |
| lora | te | 20000 | 0.6262 | 0.0119 | 0.6264 | 0.0131 | 3 | 0.0296 | 0.0325 | two-sided t interval; df=2; t*=4.302652729 |


**Table 8.**

| method | language | budget | seed | lr | epochs | accuracy | macro_f1 | trainable_params | peak_gpu_memory_gb | training_time_sec | adapter_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora | hi | 100 | 42 | 0.0001 | 10 | 0.3333 | 0.1667 | 887811 | 1.1563 | 4.8000 | /kaggle/working/adapters/lora/hi/budget100_seed42 |
| lora | hi | 100 | 123 | 0.0001 | 10 | 0.3333 | 0.1667 | 887811 | 1.1578 | 5.0100 | /kaggle/working/adapters/lora/hi/budget100_seed123 |
| lora | hi | 1000 | 123 | 0.0001 | 5 | 0.3337 | 0.1675 | 887811 | 1.1571 | 23.8700 | /kaggle/working/adapters/lora/hi/budget1000_seed123 |
| lora | hi | 1000 | 456 | 0.0001 | 5 | 0.3341 | 0.2399 | 887811 | 1.1578 | 23.8100 | /kaggle/working/adapters/lora/hi/budget1000_seed456 |
| lora | te | 100 | 42 | 0.0001 | 10 | 0.3333 | 0.1667 | 887811 | 1.1578 | 5.0100 | /kaggle/working/adapters/lora/te/budget100_seed42 |
| lora | te | 100 | 123 | 0.0001 | 10 | 0.3333 | 0.1667 | 887811 | 1.1571 | 5.0100 | /kaggle/working/adapters/lora/te/budget100_seed123 |
| lora | te | 1000 | 42 | 0.0001 | 5 | 0.3325 | 0.1687 | 887811 | 1.1571 | 23.7800 | /kaggle/working/adapters/lora/te/budget1000_seed42 |
| lora | te | 1000 | 123 | 0.0001 | 5 | 0.3333 | 0.1667 | 887811 | 1.1571 | 23.7600 | /kaggle/working/adapters/lora/te/budget1000_seed123 |
| dora | hi | 100 | 42 | 0.0001 | 10 | 0.3333 | 0.1667 | 906243 | 1.1720 | 7.0000 | /kaggle/working/adapters/dora/hi/budget100_seed42 |
| dora | hi | 100 | 123 | 0.0001 | 10 | 0.3333 | 0.1667 | 906243 | 1.1720 | 6.9200 | /kaggle/working/adapters/dora/hi/budget100_seed123 |
| dora | hi | 1000 | 123 | 0.0001 | 5 | 0.3333 | 0.1667 | 906243 | 1.1734 | 32.9100 | /kaggle/working/adapters/dora/hi/budget1000_seed123 |
| dora | te | 100 | 42 | 0.0001 | 10 | 0.3333 | 0.1667 | 906243 | 1.1720 | 6.9400 | /kaggle/working/adapters/dora/te/budget100_seed42 |
| dora | te | 100 | 123 | 0.0001 | 10 | 0.3333 | 0.1667 | 906243 | 1.1720 | 6.9000 | /kaggle/working/adapters/dora/te/budget100_seed123 |
| dora | te | 1000 | 42 | 0.0001 | 5 | 0.3325 | 0.1679 | 906243 | 1.1734 | 33.0100 | /kaggle/working/adapters/dora/te/budget1000_seed42 |
| dora | te | 1000 | 123 | 0.0001 | 5 | 0.3333 | 0.1667 | 906243 | 1.1734 | 32.7500 | /kaggle/working/adapters/dora/te/budget1000_seed123 |
| ia3 | hi | 50 | 123 | 0.0050 | 10 | 0.3341 | 0.1755 | 657411 | 1.2395 | 2.9800 | /kaggle/working/adapters/ia3/hi/budget50_seed123 |
| ia3 | hi | 50 | 456 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2395 | 3.0400 | /kaggle/working/adapters/ia3/hi/budget50_seed456 |
| ia3 | hi | 100 | 42 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2402 | 6.0100 | /kaggle/working/adapters/ia3/hi/budget100_seed42 |
| ia3 | hi | 100 | 123 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2402 | 6.1900 | /kaggle/working/adapters/ia3/hi/budget100_seed123 |
| ia3 | te | 50 | 42 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2409 | 2.9500 | /kaggle/working/adapters/ia3/te/budget50_seed42 |
| ia3 | te | 50 | 123 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2402 | 2.9800 | /kaggle/working/adapters/ia3/te/budget50_seed123 |
| ia3 | te | 50 | 456 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2402 | 3.0200 | /kaggle/working/adapters/ia3/te/budget50_seed456 |
| ia3 | te | 100 | 42 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2409 | 6.0600 | /kaggle/working/adapters/ia3/te/budget100_seed42 |
| ia3 | te | 100 | 123 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2409 | 5.9600 | /kaggle/working/adapters/ia3/te/budget100_seed123 |


**Table 9.**

| method | seed | lr | epochs | accuracy | macro_f1 | trainable_params | peak_gpu_memory_gb | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 42 | 0.0001 | 10 | 0.3361 | 0.2543 | 887811 | 1.1563 | 2.2500 |
| LoRA | 123 | 0.0001 | 10 | 0.3273 | 0.2295 | 887811 | 1.1563 | 2.2800 |
| LoRA | 456 | 0.0001 | 10 | 0.3361 | 0.1733 | 887811 | 1.1556 | 2.3300 |
| DoRA | 42 | 0.0001 | 10 | 0.3378 | 0.2613 | 906243 | 1.1721 | 3.3500 |
| DoRA | 123 | 0.0001 | 10 | 0.3269 | 0.2293 | 906243 | 1.1721 | 3.3300 |
| DoRA | 456 | 0.0001 | 10 | 0.3361 | 0.1733 | 906243 | 1.1721 | 3.3800 |
| IA³ | 42 | 0.0050 | 10 | 0.3301 | 0.2344 | 657411 | 1.2395 | 2.9400 |
| IA³ | 123 | 0.0050 | 10 | 0.3341 | 0.1755 | 657411 | 1.2395 | 2.9800 |
| IA³ | 456 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2395 | 3.0400 |


**Table 10.**

| method | f1_mean | f1_std | f1_ci95 | acc_mean | acc_std |
| --- | --- | --- | --- | --- | --- |
| DoRA | 0.2213 | 0.0445 | 0.1106 | 0.3336 | 0.0059 |
| IA³ | 0.1922 | 0.0368 | 0.0914 | 0.3325 | 0.0021 |
| LoRA | 0.2191 | 0.0415 | 0.1031 | 0.3332 | 0.0051 |


**Table 11.**

| method | seed | lr | epochs | accuracy | macro_f1 | trainable_params | peak_gpu_memory_gb | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 42 | 0.0001 | 10 | 0.3333 | 0.1667 | 887811 | 1.1563 | 4.8000 |
| LoRA | 123 | 0.0001 | 10 | 0.3333 | 0.1667 | 887811 | 1.1578 | 5.0100 |
| LoRA | 456 | 0.0001 | 10 | 0.3398 | 0.2641 | 887811 | 1.1571 | 5.1600 |
| DoRA | 42 | 0.0001 | 10 | 0.3333 | 0.1667 | 906243 | 1.1720 | 7.0000 |
| DoRA | 123 | 0.0001 | 10 | 0.3333 | 0.1667 | 906243 | 1.1720 | 6.9200 |
| DoRA | 456 | 0.0001 | 10 | 0.3406 | 0.2647 | 906243 | 1.1720 | 6.8700 |
| IA³ | 42 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2402 | 6.0100 |
| IA³ | 123 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2402 | 6.1900 |
| IA³ | 456 | 0.0050 | 10 | 0.3345 | 0.1932 | 657411 | 1.2402 | 5.9300 |


**Table 12.**

| method | f1_mean | f1_std | f1_ci95 | acc_mean | acc_std |
| --- | --- | --- | --- | --- | --- |
| DoRA | 0.1993 | 0.0566 | 0.1405 | 0.3357 | 0.0042 |
| IA³ | 0.1755 | 0.0153 | 0.0380 | 0.3337 | 0.0007 |
| LoRA | 0.1992 | 0.0563 | 0.1398 | 0.3355 | 0.0037 |


**Table 13.**

| method | seed | lr | epochs | accuracy | macro_f1 | trainable_params | peak_gpu_memory_gb | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 42 | 0.0001 | 10 | 0.3606 | 0.2771 | 887811 | 1.1571 | 23.7600 |
| LoRA | 123 | 0.0001 | 10 | 0.3847 | 0.3519 | 887811 | 1.1563 | 23.5100 |
| LoRA | 456 | 0.0001 | 10 | 0.3791 | 0.3788 | 887811 | 1.1563 | 24.0500 |
| DoRA | 42 | 0.0001 | 10 | 0.3586 | 0.2861 | 906243 | 1.1723 | 33.3400 |
| DoRA | 123 | 0.0001 | 10 | 0.3811 | 0.3245 | 906243 | 1.1723 | 32.5200 |
| DoRA | 456 | 0.0001 | 10 | 0.3811 | 0.3800 | 906243 | 1.1723 | 33.0700 |
| IA³ | 42 | 0.0050 | 10 | 0.4141 | 0.3835 | 657411 | 1.2395 | 29.7500 |
| IA³ | 123 | 0.0050 | 10 | 0.4112 | 0.3691 | 657411 | 1.2395 | 29.0400 |
| IA³ | 456 | 0.0050 | 10 | 0.4169 | 0.3870 | 657411 | 1.2395 | 29.5700 |


**Table 14.**

| method | f1_mean | f1_std | f1_ci95 | acc_mean | acc_std |
| --- | --- | --- | --- | --- | --- |
| DoRA | 0.3302 | 0.0472 | 0.1172 | 0.3736 | 0.0130 |
| IA³ | 0.3799 | 0.0095 | 0.0236 | 0.4141 | 0.0028 |
| LoRA | 0.3359 | 0.0527 | 0.1310 | 0.3748 | 0.0126 |


**Table 15.**

| method | seed | lr | epochs | accuracy | macro_f1 | trainable_params | peak_gpu_memory_gb | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 42 | 0.0001 | 5 | 0.3414 | 0.2032 | 887811 | 1.1563 | 23.5500 |
| LoRA | 123 | 0.0001 | 5 | 0.3337 | 0.1675 | 887811 | 1.1571 | 23.8700 |
| LoRA | 456 | 0.0001 | 5 | 0.3341 | 0.2399 | 887811 | 1.1578 | 23.8100 |
| DoRA | 42 | 0.0001 | 5 | 0.3406 | 0.1985 | 906243 | 1.1734 | 32.8200 |
| DoRA | 123 | 0.0001 | 5 | 0.3333 | 0.1667 | 906243 | 1.1734 | 32.9100 |
| DoRA | 456 | 0.0001 | 5 | 0.3349 | 0.2424 | 906243 | 1.1734 | 33.0200 |
| IA³ | 42 | 0.0050 | 5 | 0.4691 | 0.4416 | 657411 | 1.2402 | 29.2600 |
| IA³ | 123 | 0.0050 | 5 | 0.4197 | 0.3229 | 657411 | 1.2402 | 29.2600 |
| IA³ | 456 | 0.0050 | 5 | 0.4414 | 0.4209 | 657411 | 1.2402 | 29.4100 |


**Table 16.**

| method | f1_mean | f1_std | f1_ci95 | acc_mean | acc_std |
| --- | --- | --- | --- | --- | --- |
| DoRA | 0.2025 | 0.0380 | 0.0944 | 0.3363 | 0.0038 |
| IA³ | 0.3951 | 0.0634 | 0.1576 | 0.4434 | 0.0248 |
| LoRA | 0.2035 | 0.0362 | 0.0899 | 0.3364 | 0.0043 |


**Table 17.**

| method | seed | lr | epochs | accuracy | macro_f1 | trainable_params | peak_gpu_memory_gb | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 42 | 0.0001 | 5 | 0.4767 | 0.4738 | 887811 | 1.1563 | 47.7000 |
| LoRA | 123 | 0.0001 | 5 | 0.5141 | 0.5158 | 887811 | 1.1563 | 47.7700 |
| LoRA | 456 | 0.0001 | 5 | 0.4839 | 0.4655 | 887811 | 1.1563 | 47.5300 |
| DoRA | 42 | 0.0001 | 5 | 0.4554 | 0.4087 | 906243 | 1.1727 | 65.6700 |
| DoRA | 123 | 0.0001 | 5 | 0.5133 | 0.5146 | 906243 | 1.1727 | 65.7400 |
| DoRA | 456 | 0.0001 | 5 | 0.4643 | 0.4049 | 906243 | 1.1727 | 65.9300 |
| IA³ | 42 | 0.0050 | 5 | 0.4663 | 0.4169 | 657411 | 1.2395 | 58.6000 |
| IA³ | 123 | 0.0050 | 5 | 0.5233 | 0.5010 | 657411 | 1.2409 | 58.6800 |
| IA³ | 456 | 0.0050 | 5 | 0.5602 | 0.5565 | 657411 | 1.2417 | 58.7800 |


**Table 18.**

| method | f1_mean | f1_std | f1_ci95 | acc_mean | acc_std |
| --- | --- | --- | --- | --- | --- |
| DoRA | 0.4427 | 0.0623 | 0.1547 | 0.4776 | 0.0312 |
| IA³ | 0.4915 | 0.0703 | 0.1746 | 0.5166 | 0.0473 |
| LoRA | 0.4850 | 0.0269 | 0.0669 | 0.4916 | 0.0198 |


**Table 19.**

| method | seed | lr | epochs | accuracy | macro_f1 | trainable_params | peak_gpu_memory_gb | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 42 | 0.0001 | 3 | 0.6502 | 0.6516 | 887811 | 1.1563 | 284.3300 |
| LoRA | 123 | 0.0001 | 3 | 0.6635 | 0.6644 | 887811 | 1.1571 | 284.5400 |
| LoRA | 456 | 0.0001 | 3 | 0.6755 | 0.6762 | 887811 | 1.1563 | 284.9600 |
| DoRA | 42 | 0.0001 | 3 | 0.6703 | 0.6719 | 906243 | 1.1727 | 393.5700 |
| DoRA | 123 | 0.0001 | 3 | 0.6590 | 0.6605 | 906243 | 1.1705 | 393.4900 |
| DoRA | 456 | 0.0001 | 3 | 0.6659 | 0.6650 | 906243 | 1.1698 | 394.2300 |
| IA³ | 42 | 0.0050 | 3 | 0.6534 | 0.6554 | 657411 | 1.2402 | 350.8800 |
| IA³ | 123 | 0.0050 | 3 | 0.6675 | 0.6698 | 657411 | 1.2409 | 350.6400 |
| IA³ | 456 | 0.0050 | 3 | 0.6779 | 0.6794 | 657411 | 1.2402 | 350.8400 |


**Table 20.**

| method | f1_mean | f1_std | f1_ci95 | acc_mean | acc_std |
| --- | --- | --- | --- | --- | --- |
| DoRA | 0.6658 | 0.0057 | 0.0142 | 0.6651 | 0.0057 |
| IA³ | 0.6682 | 0.0120 | 0.0299 | 0.6663 | 0.0123 |
| LoRA | 0.6641 | 0.0123 | 0.0306 | 0.6631 | 0.0127 |


**Table 21.**

| method | seed | lr | epochs | accuracy | macro_f1 | trainable_params | peak_gpu_memory_gb | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 42 | 0.0001 | 10 | 0.3438 | 0.2148 | 887811 | 1.1571 | 2.3900 |
| LoRA | 123 | 0.0001 | 10 | 0.3217 | 0.2224 | 887811 | 1.1556 | 2.4100 |
| LoRA | 456 | 0.0001 | 10 | 0.3365 | 0.1750 | 887811 | 1.1563 | 2.4100 |
| DoRA | 42 | 0.0001 | 10 | 0.3430 | 0.2146 | 906243 | 1.1699 | 3.3300 |
| DoRA | 123 | 0.0001 | 10 | 0.3253 | 0.2263 | 906243 | 1.1705 | 3.3700 |
| DoRA | 456 | 0.0001 | 10 | 0.3365 | 0.1750 | 906243 | 1.1721 | 3.4400 |
| IA³ | 42 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2409 | 2.9500 |
| IA³ | 123 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2402 | 2.9800 |
| IA³ | 456 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2402 | 3.0200 |


**Table 22.**

| method | f1_mean | f1_std | f1_ci95 | acc_mean | acc_std |
| --- | --- | --- | --- | --- | --- |
| DoRA | 0.2053 | 0.0269 | 0.0668 | 0.3349 | 0.0089 |
| IA³ | 0.1667 | 0.0000 | 0.0000 | 0.3333 | 0.0000 |
| LoRA | 0.2041 | 0.0255 | 0.0632 | 0.3340 | 0.0113 |


**Table 23.**

| method | seed | lr | epochs | accuracy | macro_f1 | trainable_params | peak_gpu_memory_gb | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 42 | 0.0001 | 10 | 0.3333 | 0.1667 | 887811 | 1.1578 | 5.0100 |
| LoRA | 123 | 0.0001 | 10 | 0.3333 | 0.1667 | 887811 | 1.1571 | 5.0100 |
| LoRA | 456 | 0.0001 | 10 | 0.3478 | 0.2781 | 887811 | 1.1578 | 4.9500 |
| DoRA | 42 | 0.0001 | 10 | 0.3333 | 0.1667 | 906243 | 1.1720 | 6.9400 |
| DoRA | 123 | 0.0001 | 10 | 0.3333 | 0.1667 | 906243 | 1.1720 | 6.9000 |
| DoRA | 456 | 0.0001 | 10 | 0.3502 | 0.2799 | 906243 | 1.1720 | 6.9100 |
| IA³ | 42 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2409 | 6.0600 |
| IA³ | 123 | 0.0050 | 10 | 0.3333 | 0.1667 | 657411 | 1.2409 | 5.9600 |
| IA³ | 456 | 0.0050 | 10 | 0.3683 | 0.2855 | 657411 | 1.2402 | 5.9600 |


**Table 24.**

| method | f1_mean | f1_std | f1_ci95 | acc_mean | acc_std |
| --- | --- | --- | --- | --- | --- |
| DoRA | 0.2044 | 0.0654 | 0.1624 | 0.3390 | 0.0097 |
| IA³ | 0.2063 | 0.0686 | 0.1704 | 0.3450 | 0.0202 |
| LoRA | 0.2038 | 0.0643 | 0.1598 | 0.3382 | 0.0083 |


**Table 25.**

| method | seed | lr | epochs | accuracy | macro_f1 | trainable_params | peak_gpu_memory_gb | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 42 | 0.0001 | 10 | 0.3518 | 0.2529 | 887811 | 1.1563 | 23.8900 |
| LoRA | 123 | 0.0001 | 10 | 0.3715 | 0.3612 | 887811 | 1.1571 | 23.7400 |
| LoRA | 456 | 0.0001 | 10 | 0.3522 | 0.2476 | 887811 | 1.1563 | 23.7700 |
| DoRA | 42 | 0.0001 | 10 | 0.3522 | 0.2586 | 906243 | 1.1723 | 33.2100 |
| DoRA | 123 | 0.0001 | 10 | 0.3715 | 0.3401 | 906243 | 1.1723 | 32.7300 |
| DoRA | 456 | 0.0001 | 10 | 0.3498 | 0.2344 | 906243 | 1.1723 | 32.9000 |
| IA³ | 42 | 0.0050 | 10 | 0.3815 | 0.3261 | 657411 | 1.2395 | 29.7000 |
| IA³ | 123 | 0.0050 | 10 | 0.3948 | 0.3268 | 657411 | 1.2395 | 29.0100 |
| IA³ | 456 | 0.0050 | 10 | 0.4137 | 0.4006 | 657411 | 1.2395 | 29.5600 |


**Table 26.**

| method | f1_mean | f1_std | f1_ci95 | acc_mean | acc_std |
| --- | --- | --- | --- | --- | --- |
| DoRA | 0.2777 | 0.0554 | 0.1375 | 0.3578 | 0.0119 |
| IA³ | 0.3512 | 0.0428 | 0.1063 | 0.3967 | 0.0161 |
| LoRA | 0.2872 | 0.0641 | 0.1593 | 0.3585 | 0.0112 |


**Table 27.**

| method | seed | lr | epochs | accuracy | macro_f1 | trainable_params | peak_gpu_memory_gb | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 42 | 0.0001 | 5 | 0.3325 | 0.1687 | 887811 | 1.1571 | 23.7800 |
| LoRA | 123 | 0.0001 | 5 | 0.3333 | 0.1667 | 887811 | 1.1571 | 23.7600 |
| LoRA | 456 | 0.0001 | 5 | 0.3474 | 0.2763 | 887811 | 1.1578 | 23.8400 |
| DoRA | 42 | 0.0001 | 5 | 0.3325 | 0.1679 | 906243 | 1.1734 | 33.0100 |
| DoRA | 123 | 0.0001 | 5 | 0.3333 | 0.1667 | 906243 | 1.1734 | 32.7500 |
| DoRA | 456 | 0.0001 | 5 | 0.3514 | 0.2809 | 906243 | 1.1734 | 33.0500 |
| IA³ | 42 | 0.0050 | 5 | 0.3727 | 0.2691 | 657411 | 1.2402 | 29.2200 |
| IA³ | 123 | 0.0050 | 5 | 0.4133 | 0.3571 | 657411 | 1.2402 | 29.3800 |
| IA³ | 456 | 0.0050 | 5 | 0.3345 | 0.1771 | 657411 | 1.2402 | 29.3200 |


**Table 28.**

| method | f1_mean | f1_std | f1_ci95 | acc_mean | acc_std |
| --- | --- | --- | --- | --- | --- |
| DoRA | 0.2052 | 0.0656 | 0.1630 | 0.3391 | 0.0107 |
| IA³ | 0.2678 | 0.0900 | 0.2236 | 0.3735 | 0.0394 |
| LoRA | 0.2039 | 0.0627 | 0.1558 | 0.3378 | 0.0084 |


**Table 29.**

| method | seed | lr | epochs | accuracy | macro_f1 | trainable_params | peak_gpu_memory_gb | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 42 | 0.0001 | 5 | 0.3398 | 0.2009 | 887811 | 1.1563 | 47.5600 |
| LoRA | 123 | 0.0001 | 5 | 0.3373 | 0.1801 | 887811 | 1.1563 | 47.8100 |
| LoRA | 456 | 0.0001 | 5 | 0.3932 | 0.3402 | 887811 | 1.1563 | 47.5800 |
| DoRA | 42 | 0.0001 | 5 | 0.3859 | 0.3642 | 906243 | 1.1727 | 65.6800 |
| DoRA | 123 | 0.0001 | 5 | 0.3530 | 0.2438 | 906243 | 1.1727 | 65.7700 |
| DoRA | 456 | 0.0001 | 5 | 0.3747 | 0.3384 | 906243 | 1.1727 | 65.8700 |
| IA³ | 42 | 0.0050 | 5 | 0.4321 | 0.3460 | 657411 | 1.2395 | 58.6400 |
| IA³ | 123 | 0.0050 | 5 | 0.4382 | 0.3746 | 657411 | 1.2409 | 58.6800 |
| IA³ | 456 | 0.0050 | 5 | 0.4699 | 0.4643 | 657411 | 1.2417 | 58.6100 |


**Table 30.**

| method | f1_mean | f1_std | f1_ci95 | acc_mean | acc_std |
| --- | --- | --- | --- | --- | --- |
| DoRA | 0.3155 | 0.0634 | 0.1575 | 0.3712 | 0.0167 |
| IA³ | 0.3949 | 0.0617 | 0.1534 | 0.4467 | 0.0203 |
| LoRA | 0.2404 | 0.0870 | 0.2162 | 0.3568 | 0.0316 |


**Table 31.**

| method | seed | lr | epochs | accuracy | macro_f1 | trainable_params | peak_gpu_memory_gb | training_time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 42 | 0.0001 | 3 | 0.6112 | 0.6125 | 887811 | 1.1563 | 284.3500 |
| LoRA | 123 | 0.0001 | 3 | 0.6337 | 0.6339 | 887811 | 1.1571 | 285.0700 |
| LoRA | 456 | 0.0001 | 3 | 0.6341 | 0.6322 | 887811 | 1.1563 | 285.1200 |
| DoRA | 42 | 0.0001 | 3 | 0.6197 | 0.6209 | 906243 | 1.1727 | 392.9000 |
| DoRA | 123 | 0.0001 | 3 | 0.6317 | 0.6318 | 906243 | 1.1705 | 394.0900 |
| DoRA | 456 | 0.0001 | 3 | 0.6185 | 0.6153 | 906243 | 1.1698 | 393.2400 |
| IA³ | 42 | 0.0050 | 3 | 0.6197 | 0.6207 | 657411 | 1.2402 | 351.3200 |
| IA³ | 123 | 0.0050 | 3 | 0.6233 | 0.6244 | 657411 | 1.2409 | 351.4100 |
| IA³ | 456 | 0.0050 | 3 | 0.6068 | 0.6043 | 657411 | 1.2402 | 351.1600 |


**Table 32.**

| method | f1_mean | f1_std | f1_ci95 | acc_mean | acc_std |
| --- | --- | --- | --- | --- | --- |
| DoRA | 0.6227 | 0.0084 | 0.0209 | 0.6233 | 0.0073 |
| IA³ | 0.6165 | 0.0107 | 0.0266 | 0.6166 | 0.0087 |
| LoRA | 0.6262 | 0.0119 | 0.0296 | 0.6264 | 0.0131 |


**Table 33.**

| Method | Seed 42 | Seed 123 | Seed 456 |
| --- | --- | --- | --- |
| LoRA | 0.2543 | 0.2295 | 0.1733 |
| DoRA | 0.2613 | 0.2293 | 0.1733 |
| IA³ | 0.2344 | 0.1755 | 0.1667 |


**Table 34.**

| Method | Seed 42 | Seed 123 | Seed 456 |
| --- | --- | --- | --- |
| LoRA | 0.1667 | 0.1667 | 0.2641 |
| DoRA | 0.1667 | 0.1667 | 0.2647 |
| IA³ | 0.1667 | 0.1667 | 0.1932 |


**Table 35.**

| Method | Seed 42 | Seed 123 | Seed 456 |
| --- | --- | --- | --- |
| LoRA | 0.2771 | 0.3519 | 0.3788 |
| DoRA | 0.2861 | 0.3245 | 0.3800 |
| IA³ | 0.3835 | 0.3691 | 0.3870 |


**Table 36.**

| Method | Seed 42 | Seed 123 | Seed 456 |
| --- | --- | --- | --- |
| LoRA | 0.2032 | 0.1675 | 0.2399 |
| DoRA | 0.1985 | 0.1667 | 0.2424 |
| IA³ | 0.4416 | 0.3229 | 0.4209 |


**Table 37.**

| Method | Seed 42 | Seed 123 | Seed 456 |
| --- | --- | --- | --- |
| LoRA | 0.4738 | 0.5158 | 0.4655 |
| DoRA | 0.4087 | 0.5146 | 0.4049 |
| IA³ | 0.4169 | 0.5010 | 0.5565 |


**Table 38.**

| Method | Seed 42 | Seed 123 | Seed 456 |
| --- | --- | --- | --- |
| LoRA | 0.6516 | 0.6644 | 0.6762 |
| DoRA | 0.6719 | 0.6605 | 0.6650 |
| IA³ | 0.6554 | 0.6698 | 0.6794 |


**Table 39.**

| Method | Seed 42 | Seed 123 | Seed 456 |
| --- | --- | --- | --- |
| LoRA | 0.2148 | 0.2224 | 0.1750 |
| DoRA | 0.2146 | 0.2263 | 0.1750 |
| IA³ | 0.1667 | 0.1667 | 0.1667 |


**Table 40.**

| Method | Seed 42 | Seed 123 | Seed 456 |
| --- | --- | --- | --- |
| LoRA | 0.1667 | 0.1667 | 0.2781 |
| DoRA | 0.1667 | 0.1667 | 0.2799 |
| IA³ | 0.1667 | 0.1667 | 0.2855 |


**Table 41.**

| Method | Seed 42 | Seed 123 | Seed 456 |
| --- | --- | --- | --- |
| LoRA | 0.2529 | 0.3612 | 0.2476 |
| DoRA | 0.2586 | 0.3401 | 0.2344 |
| IA³ | 0.3261 | 0.3268 | 0.4006 |


**Table 42.**

| Method | Seed 42 | Seed 123 | Seed 456 |
| --- | --- | --- | --- |
| LoRA | 0.1687 | 0.1667 | 0.2763 |
| DoRA | 0.1679 | 0.1667 | 0.2809 |
| IA³ | 0.2691 | 0.3571 | 0.1771 |


**Table 43.**

| Method | Seed 42 | Seed 123 | Seed 456 |
| --- | --- | --- | --- |
| LoRA | 0.2009 | 0.1801 | 0.3402 |
| DoRA | 0.3642 | 0.2438 | 0.3384 |
| IA³ | 0.3460 | 0.3746 | 0.4643 |


**Table 44.**

| Method | Seed 42 | Seed 123 | Seed 456 |
| --- | --- | --- | --- |
| LoRA | 0.6125 | 0.6339 | 0.6322 |
| DoRA | 0.6209 | 0.6318 | 0.6153 |
| IA³ | 0.6207 | 0.6244 | 0.6043 |
