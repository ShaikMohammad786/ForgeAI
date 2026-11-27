# Optimization Iteration Log

## Iteration 1 (wrong_refusal) — ACCEPTED
- **Diagnosis:** Dominant failure is `wrong_refusal` for out-of-scope questions.
- **Failure IDs Targeted:** `['eval_17', 'eval_18', 'eval_19', 'eval_20', 'eval_21']`
- **Score:** `0.703` -> `0.723`
- **Pass Rate:** `58.3%` -> `62.5%`
- **Decision:** Accepted patch (+4.2pp pass rate, +0.020 composite score).
- **Targeted Diff:**
  - `retrieval.min_score_threshold`: `0.08` -> `0.12` (Raise score gate to enforce refusal on out-of-scope queries)
  - `system_prompt`: `(len=733)` -> `+ If the question is out of scope, refuse it based on provided documents and current date.` (Clarify answer-vs-refuse boundary)
