# Optimization Iteration Log

## Iteration 1 (prompt_defect) — ROLLED BACK
- **Diagnosis:** Dominant failure is numeric inaccuracy due to low min_score_threshold, and prompt defects related to retrieval misses.
- **Failure IDs Targeted:** `['eval_03', 'eval_20', 'eval_24']`
- **Score:** `0.854` -> `0.803`
- **Pass Rate:** `79.2%` -> `66.7%`
- **Decision:** Rolled back patch (pass_delta=-0.125, score_delta=-0.051 did not improve over best).
- **Targeted Diff:**
  - `system_prompt`: `(len=721)` -> `+ FORMAT & COMPLETENESS RULE: Include all key facts, exact numbers, and at least one valid filename in citations; return s` (Repair prompt compliance and fact/citation completeness)
  - `retrieval.top_k`: `5` -> `6` (Provide +1 chunk for multi-document synthesis)
