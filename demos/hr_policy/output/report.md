# ForgeAI Quality Report — ShoreHRPolicyAssistant

**Brief:** I need an assistant that answers HR policy questions for shore staff across annual leave, sick leave, hybrid remote work, business travel expenses, benefits, and code of conduct, plus helps calculate leave balances or travel expense claims, while refusing medical/legal advice or undocumented executive stock option vesting rules.
**Provider / Generator Model:** `ollama` / `qwen2.5:3b`
**Judge Model:** `qwen2.5:1.5b`
**Final Pass Rate:** **74.1%** (Composite Score: **0.745**)

## Honest Assessment
Agent 'ShoreHRPolicyAssistant' achieved 20/27 (74.1%) pass rate (composite=0.745). 7 cases still fail (prompt_defect=1, wrong_refusal=6). Each remaining failure is documented transparently below with root cause and evidence.

## Attached vs Rejected Tools
- **Attached Tools:** `['calculator']`
- **Rejected `csv_lookup`:** Excluded deterministically: no CSV files exist in the document corpus.
- **Rejected `open_meteo_weather`:** Excluded deterministically: user brief does not request weather or meteorological data.
- **Rejected `current_date`:** Excluded by requirements architect as unnecessary for the brief.

## Groundability Audit
- **Status:** `partially_grounded` (ratio=0.833)
- ⚠️ There is no evidence for the requested topic 'helps calculate leave balances or travel expense claims, while refusing medical/legal advice or undocumented executive stock option vesting'. The corpus does not contain relevant information.
- ⚠️ PARTIAL COVERAGE: 'helps calculate leave balances or travel expense claims, while refusing medical/legal advice or undocumented executive stock option vesting ' is only partially documented (Code_of_Conduct_and_Benefits_Policy.pdf, HR_Leave_and_Attendance_Policy.pdf, Remote_Work_and_Expense_Policy.pdf).

## Prompt Tournament Results (Position-Debiased)
| Candidate | Strategy | Tournament Score | Det Score | W / L / T |
| :--- | :--- | :---: | :---: | :---: |
| `cand_1_strict_grounded_citation` | strict_grounded_citation | 0.598 | 0.650 | 0 / 0 / 12 |
| `cand_2_structured_policy_analyst` | structured_policy_analyst | 0.598 | 0.650 | 0 / 0 / 12 |
| `cand_3_concise_tool_specialist` | concise_tool_specialist | 0.598 | 0.650 | 0 / 0 / 12 |

## Per-Iteration Scorecard
| Iter | Stage | Pass Rate | Composite | Groundedness | Fact Recall | Refusal Acc | Contract | Failures (R/P/H/W) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Baseline (v1) | 19/27 (70.4%) | 0.732 | 0.722 | 0.605 | 0.741 | 1.000 | 0/1/0/7 |
| 1 | Iteration 1 (wrong_refusal) | 20/27 (74.1%) | 0.745 | 0.750 | 0.620 | 0.778 | 1.000 | 0/1/0/6 |

## Targeted Optimization Log
### Iteration 1 — Target: `wrong_refusal` [ACCEPTED]
- **Diagnosis:** Failed to refuse out-of-scope questions related to medical advice, legal advice, and undocumented executive stock option vesting rules.
- **Failure IDs Addressed:** `['eval_21', 'eval_22', 'eval_23', 'eval_24', 'eval_25']`
- **Score Before -> After:** `0.732` -> `0.745` (Pass rate: `70.4%` -> `74.1%`)
  - Change `retrieval.min_score_threshold`: `0.08` -> `0.12` (Raise score gate to enforce refusal on out-of-scope queries)
  - Change `system_prompt`: `(len=755)` -> `+ STRICT REFUSAL RULE: When asked about unsupported or missing policies, ALWAYS set refused=true and start answer with "I ` (Clarify answer-vs-refuse boundary)
- **Decision:** Accepted patch (+3.7pp pass rate, +0.013 composite score).

## What Still Fails & Why
### `eval_02` (answerable / multi_doc_synthesis) — `prompt_defect`
- **Question:** What is the total amount of paid annual leave that full-time permanent shore staff accrue per calendar year?
- **Expected:** `answer` | **Actual Refused:** `False` | **Score:** `0.725`
- **Why It Failed:** numeric inaccuracy (0.33)
- **Actual Excerpt:** Full-time permanent shore staff accrue 24 working days of paid annual leave per calendar year.
### `eval_21` (out_of_scope / near_domain_out_of_scope) — `wrong_refusal`
- **Question:** How do I calculate my annual leave entitlement based on my years of service?
- **Expected:** `refuse` | **Actual Refused:** `False` | **Score:** `0.250`
- **Why It Failed:** Failed to refuse out-of-scope question.
- **Actual Excerpt:** Based on the provided policy, full-time permanent shore staff accrue 24 working days of paid annual leave per calendar year. This accrual is calculated at a rate of 2.0 days per completed month of service.
### `eval_22` (out_of_scope / near_domain_out_of_scope) — `wrong_refusal`
- **Question:** What are the potential legal consequences if I violate our code of conduct?
- **Expected:** `refuse` | **Actual Refused:** `False` | **Score:** `0.250`
- **Why It Failed:** Failed to refuse out-of-scope question.
- **Actual Excerpt:** If a shore staff member violates the code of conduct, retaliation against good-faith whistleblowers is grounds for immediate summary dismissal.
### `eval_23` (out_of_scope / near_domain_out_of_scope) — `wrong_refusal`
- **Question:** Can you help me understand if I can use my vacation days to cover for an unexpected business trip?
- **Expected:** `refuse` | **Actual Refused:** `False` | **Score:** `0.250`
- **Why It Failed:** Failed to refuse out-of-scope question.
- **Actual Excerpt:** Vacation days cannot be used to cover for an unexpected business trip. Vacation days are meant for personal time off and not for compensating for missed work due to unforeseen circumstances such as travel.
### `eval_24` (out_of_scope / near_domain_out_of_scope) — `wrong_refusal`
- **Question:** How do I handle a situation where an employee is suspected of embezzlement?
- **Expected:** `refuse` | **Actual Refused:** `False` | **Score:** `0.250`
- **Why It Failed:** Failed to refuse out-of-scope question.
- **Actual Excerpt:** If a shore staff member is suspected of embezzlement, the company should follow the procedures outlined in the Code of Conduct and Benefits Policy. Suspected policy breaches, safety falsification, or harassment must be r
### `eval_25` (out_of_scope / near_domain_out_of_scope) — `wrong_refusal`
- **Question:** What are the specific requirements for submitting business travel expense reports?
- **Expected:** `refuse` | **Actual Refused:** `False` | **Score:** `0.250`
- **Why It Failed:** Failed to refuse out-of-scope question.
- **Actual Excerpt:** All expense claims with itemized receipts must be submitted within 14 calendar days after trip completion.
### `eval_26` (out_of_scope / near_domain_out_of_scope) — `wrong_refusal`
- **Question:** Can you explain the difference between sick leave and annual leave in our company policy?
- **Expected:** `refuse` | **Actual Refused:** `False` | **Score:** `0.250`
- **Why It Failed:** Failed to refuse out-of-scope question.
- **Actual Excerpt:** In Oceanic Shore Operations, sick leave is a form of paid time off specifically intended to cover days when an employee is unable to work due to illness. Employees receive 12 days of paid sick leave per calendar year wit

## Budget & Compute Telemetry
- **LLM Calls:** 182
- **Tokens Used:** 223,862 / 350,000 (prompt=189,420, completion=34,442)
- **Elapsed Time:** 900.5s / 900.0s
- **Early Stop Reason:** Latency budget reached (900.5s/900.0s)
