# ForgeAI Quality Report — ShoreHRPolicyAssistant

**Brief:** I need an assistant that answers HR policy questions for shore staff across annual leave, sick leave, hybrid remote work, business travel expenses, benefits, and code of conduct, plus helps calculate leave balances or travel expense claims, while refusing medical/legal advice or undocumented executive stock option vesting rules.
**Provider / Generator Model:** `ollama` / `qwen2.5:3b`
**Judge Model:** `qwen2.5:1.5b`
**Final Pass Rate:** **79.2%** (Composite Score: **0.854**)

## Honest Assessment
Agent 'ShoreHRPolicyAssistant' achieved 19/24 (79.2%) pass rate (composite=0.854). 5 cases still fail (prompt_defect=3, hallucination=1, wrong_refusal=1). Each remaining failure is documented transparently below with root cause and evidence.

## Attached vs Rejected Tools
- **Attached Standard Tools:** `['calculator', 'current_date']`
- **Synthesized Custom Macros (AST-verified):** `['none']`
- **Rejected `csv_lookup`:** Excluded deterministically: no CSV/TSV/XLSX files exist in the document corpus.
- **Rejected `json_record_lookup`:** Excluded deterministically: no JSON/JSONL/YAML files exist in the document corpus.
- **Rejected `statistics_calculator`:** Excluded deterministically: user brief does not request statistical/variance/KPI calculations.
- **Rejected `unit_converter`:** Excluded deterministically: no engineering/logistics unit conversion intent detected.
- **Rejected `currency_converter`:** Excluded deterministically: user brief does not request currency conversion or FX rates.
- **Rejected `open_meteo_weather`:** Excluded deterministically: user brief does not request weather or meteorological data.
- **Rejected `http_public_country_info`:** Excluded deterministically: user brief does not request country metadata, ISO codes, or calling codes.
- **Rejected `text_stats_and_regex`:** Excluded deterministically: no regex/frequency text audit intent or log corpus detected.

## Groundability Audit
- **Status:** `grounded` (ratio=1.0)
- ⚠️ There is no evidence supporting the requested topic 'refuses medical/legal advice or undocumented executive stock option vesting rules'.

## Prompt Tournament Results (Position-Debiased)
| Candidate | Strategy | Tournament Score | Det Score | W / L / T |
| :--- | :--- | :---: | :---: | :---: |
| `cand_2_structured_policy_analyst` | structured_policy_analyst | 0.744 | 0.875 | 0 / 0 / 8 |
| `cand_3_concise_tool_specialist` | concise_tool_specialist | 0.701 | 0.809 | 0 / 0 / 8 |
| `cand_1_strict_grounded_citation` | strict_grounded_citation | 0.697 | 0.803 | 0 / 0 / 8 |

## Per-Iteration Scorecard
| Iter | Stage | Pass Rate | Composite | Groundedness | Fact Recall | Refusal Acc | Contract | Failures (R/P/H/W) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Baseline (v1) | 19/24 (79.2%) | 0.854 | 0.919 | 0.757 | 0.958 | 1.000 | 0/3/1/1 |
| 1 | Iteration 1 (prompt_defect) [ROLLED BACK] | 16/24 (66.7%) | 0.803 | 0.919 | 0.682 | 0.958 | 1.000 | 0/6/1/1 |

## Final Pass Rate by Question Category
| Category | Passed | Total | Pass Rate | Mean Composite |
| :--- | :---: | :---: | :---: | :---: |
| `answerable` | 10 | 13 | 76.9% | 0.856 |
| `out_of_scope` | 1 | 2 | 50.0% | 0.625 |
| `edge_case` | 8 | 9 | 88.9% | 0.901 |

## Targeted Optimization Log
### Iteration 1 — Target: `prompt_defect` [ROLLED BACK]
- **Diagnosis:** Dominant failure is numeric inaccuracy due to low min_score_threshold, and prompt defects related to retrieval misses.
- **Failure IDs Addressed:** `['eval_03', 'eval_20', 'eval_24']`
- **Score Before -> After:** `0.854` -> `0.803` (Pass rate: `79.2%` -> `66.7%`)
  - Change `system_prompt`: `(len=721)` -> `+ FORMAT & COMPLETENESS RULE: Include all key facts, exact numbers, and at least one valid filename in citations; return s` (Repair prompt compliance and fact/citation completeness)
  - Change `retrieval.top_k`: `5` -> `6` (Provide +1 chunk for multi-document synthesis)
- **Decision:** Rolled back patch (pass_delta=-0.125, score_delta=-0.051 did not improve over best).

## What Still Fails & Why
### `eval_03` (answerable / multi_doc_synthesis) — `prompt_defect`
- **Question:** What is the probation notice period for Job Band D and above shore employees?
- **Expected:** `answer` | **Actual Refused:** `False` | **Score:** `0.713`
- **Why It Failed:** numeric inaccuracy (0.25)
- **Actual Excerpt:** For Job Bands D and above, the standard written resignation notice period is 60 calendar days.
### `eval_17` (edge_case / tool_calculation) — `hallucination`
- **Question:** How many days of paid annual leave do full-time shore staff accrue per calendar year, including any carry-over days if they have 24 days already accrued?
- **Expected:** `answer` | **Actual Refused:** `False` | **Score:** `0.512`
- **Why It Failed:** Ungrounded claims/numbers detected: unverified numeric claim
- **Actual Excerpt:** Full-time shore staff accrue a total of up to 29 working days of paid annual leave per calendar year, including the possibility of carrying forward up to 5 unused annual leave days into the next calendar year.
### `eval_19` (out_of_scope / near_domain_out_of_scope | groundability_gap) — `wrong_refusal`
- **Question:** What is the legal requirement for sick leave in our country?
- **Expected:** `refuse` | **Actual Refused:** `False` | **Score:** `0.250`
- **Why It Failed:** Failed to refuse out-of-scope question.
- **Actual Excerpt:** The policy states that shore employees receive 12 days of paid sick leave per calendar year without carryover. A licensed physician's medical certificate is mandatory for any sick leave absence exceeding 3 consecutive wo
### `eval_20` (answerable / single_doc) — `prompt_defect`
- **Question:** According to Code_of_Conduct_and_Benefits_Policy.pdf, what are the details regarding 3.1 Gifts, Hospitality & Conflict of Interest?
- **Expected:** `answer` | **Actual Refused:** `False` | **Score:** `0.725`
- **Why It Failed:** numeric inaccuracy (0.33)
- **Actual Excerpt:** The policy prohibits cash or cash-equivalent gifts (including gift cards) of any value and requires disclosure in writing to Compliance for actual or potential conflicts of interest within 10 business days.
### `eval_24` (answerable / single_doc) — `prompt_defect`
- **Question:** According to HR_Leave_and_Attendance_Policy.pdf, what are the details regarding (Hr-Pol-01)?
- **Expected:** `answer` | **Actual Refused:** `False` | **Score:** `0.675`
- **Why It Failed:** numeric inaccuracy (0.00)
- **Actual Excerpt:** HR_Leave_and_Attendance_Policy.pdf contains details about leave and attendance policy for Oceanic Shore Operations shore staff including annual paid leave entitlements, sick leave, parental & compassionate leave, core wo

## Budget & Compute Telemetry
- **LLM Calls:** 182
- **Tokens Used:** 262,210 / 400,000 (prompt=210,253, completion=51,957)
- **Elapsed Time:** 1201.9s / 1200.0s
- **Early Stop Reason:** Latency budget reached (1201.9s/1200.0s)
