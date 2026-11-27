# Optimization Iteration Log

## Iteration 1 (wrong_refusal) — ACCEPTED
- **Diagnosis:** Failed to refuse out-of-scope questions related to medical advice, legal advice, and undocumented executive stock option vesting rules.
- **Failure IDs Targeted:** `['eval_21', 'eval_22', 'eval_23', 'eval_24', 'eval_25']`
- **Score:** `0.732` -> `0.745`
- **Pass Rate:** `70.4%` -> `74.1%`
- **Decision:** Accepted patch (+3.7pp pass rate, +0.013 composite score).
- **Targeted Diff:**
  - `retrieval.min_score_threshold`: `0.08` -> `0.12` (Raise score gate to enforce refusal on out-of-scope queries)
  - `system_prompt`: `(len=755)` -> `+ STRICT REFUSAL RULE: When asked about unsupported or missing policies, ALWAYS set refused=true and start answer with "I ` (Clarify answer-vs-refuse boundary)
