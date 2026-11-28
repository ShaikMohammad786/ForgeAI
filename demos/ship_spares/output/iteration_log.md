# Optimization Iteration Log

## Iteration 1 (prompt_defect) — ACCEPTED
- **Diagnosis:** Dominant failure is retrieval misses due to low recall and score thresholds.
- **Failure IDs Targeted:** `['eval_10', 'eval_12', 'eval_13', 'eval_16', 'eval_18']`
- **Score:** `0.791` -> `0.810`
- **Pass Rate:** `58.3%` -> `70.8%`
- **Decision:** Accepted patch (+12.5pp pass rate, +0.020 composite score).
- **Targeted Diff:**
  - `system_prompt`: `(len=923)` -> `+ FORMAT & COMPLETENESS RULE: Include all key facts, exact numbers, and at least one valid filename in citations; return s` (Repair prompt compliance and fact/citation completeness)
  - `retrieval.top_k`: `5` -> `6` (Provide +1 chunk for multi-document synthesis)
