# Maintained results summary

This is the current interpretation of the CSV files preserved in this folder.
It supersedes the earlier discussion draft in `archive/`.

## Evidence available in this archive

| File | Content |
|---|---|
| `../results/final_primary_results.csv` | 108 seed-level validation results |
| `../results/final_summary_with_ci.csv` | 36 primary aggregates and three-seed intervals |
| `../results/final_hybrid_comparison.csv` | 12 held-out hybrid comparison cells |

## Primary validation sweep

The main experiment contains 108 validation runs:

`3 methods × 2 languages × 6 training budgets × 3 seeds`

The methods are LoRA, DoRA, and IA³. The highest observed mean macro-F1 in each
language-budget cell is:

| Budget | Hindi winner | Telugu winner |
|---:|---|---|
| 50 | DoRA | DoRA |
| 100 | DoRA | IA³ |
| 500 | IA³ | IA³ |
| 1,000 | IA³ | IA³ |
| 2,000 | IA³ | IA³ |
| 20,000 | IA³ | LoRA |

The supported descriptive conclusion is that IA³ is the highest-mean method at
the intermediate 500–2,000 budgets in both languages and at every Hindi budget
from 500 onward. It is not supported to say that IA³ wins every budget from 500
onward in both languages.

## Hybrid held-out comparison

The IA³+LoRA hybrid is a separate follow-up, evaluated against the best primary
method in each budget-language cell. It wins 8 of 12 held-out cells (67%).

- Wins: 100, 500, 2,000, and 20,000 examples in both languages.
- Losses: 50 and 1,000 examples in both languages.

This is supplemental held-out evidence and must not replace or be pooled with
the primary validation-sweep rankings.

## Interpretation limits

- The primary sweep is validation-set analysis; it is not a final held-out
  evaluation.
- Intervals use a two-sided t interval with three seeds (`df=2`,
  `t*=4.302652729`). They describe seed variability, not pairwise significance.
- Twenty-four of 108 primary runs (22.2%) are at or within 0.001 of the
  one-class chance-accuracy baseline. The counts are LoRA 8, DoRA 7, and IA³ 9.
- The hybrid configuration used a dual learning-rate setup that was not selected
  under the same protocol as the primary methods.
- Training budgets use different epoch schedules, so example count and number
  of optimizer updates are not fully isolated.
