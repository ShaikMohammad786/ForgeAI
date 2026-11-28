# Factory vs Human Baseline Comparison (Bonus 4)

- **Evaluation Suite:** `24` auto-generated cases (`demos\hr_policy\output\eval_set.json`)
- **Model Under Test:** `ollama` / `qwen2.5:3b` | **Judge:** `qwen2.5:1.5b`

## Head-to-Head Scorecard
| Metric | Human Baseline | ForgeAI Optimized | Delta |
| :--- | :---: | :---: | :---: |
| Pass Rate | 19/24 (79.2%) | 19/24 (79.2%) | +0.0pp |
| Composite Score | 0.878 | 0.857 | -0.021 |
| Groundedness | 0.952 | 0.919 | -0.033 |
| Fact Recall | 0.806 | 0.772 | -0.034 |
| Refusal Accuracy | 1.000 | 0.958 | -0.042 |
| Contract Adherence | 1.000 | 1.000 | +0.000 |

## Where ForgeAI Beats the Human Baseline (1 cases)
- `eval_04`: What is the total amount of paid sick leave that new hires can receive during their probation period? *(Human failed via `hallucination`: Ungrounded claims/numbers detected: unverified numeric claim)*

## Where the Human Baseline Beats ForgeAI (1 cases)
- `eval_19`: What is the legal requirement for sick leave in our country? *(ForgeAI failed via `wrong_refusal`: Failed to refuse out-of-scope question.)*
