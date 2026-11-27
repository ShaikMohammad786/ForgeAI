# Factory vs Human Baseline Comparison (Bonus 4)

- **Evaluation Suite:** `27` auto-generated cases (`demos\hr_policy\output\eval_set.json`)
- **Model Under Test:** `ollama` / `qwen2.5:3b` | **Judge:** `qwen2.5:1.5b`

## Head-to-Head Scorecard
| Metric | Human Baseline | ForgeAI Optimized | Delta |
| :--- | :---: | :---: | :---: |
| Pass Rate | 19/27 (70.4%) | 20/27 (74.1%) | +3.7pp |
| Composite Score | 0.745 | 0.747 | +0.002 |
| Groundedness | 0.752 | 0.750 | -0.002 |
| Fact Recall | 0.623 | 0.625 | +0.002 |
| Refusal Accuracy | 0.778 | 0.778 | +0.000 |
| Contract Adherence | 1.000 | 1.000 | +0.000 |

## Where ForgeAI Beats the Human Baseline (1 cases)
- `eval_14`: What is the total annual paid leave entitlement for full-time permanent shore staff, including both base entitlement and carry-forward days? *(Human failed via `hallucination`: Ungrounded claims/numbers detected: unverified numeric claim)*

## Where the Human Baseline Beats ForgeAI (0 cases)
- None in this run.
