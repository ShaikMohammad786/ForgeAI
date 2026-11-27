# ForgeAI Quality Report — Ship_Spares_Agent

**Brief:** Build a ship-spares procurement assistant for marine fleet superintendents that looks up spare parts from our catalogue CSV, checks supplier discount tiers, compliance certificates, and port freight lead times, and calculates total order landed costs and estimated arrival dates, while refusing unapproved emergency credit overrides or legal/weather forecasts outside the docs.
**Provider / Generator Model:** `ollama` / `qwen2.5:3b`
**Judge Model:** `qwen2.5:1.5b`
**Final Pass Rate:** **62.5%** (Composite Score: **0.723**)

## Honest Assessment
Agent 'Ship_Spares_Agent' achieved 15/24 (62.5%) pass rate (composite=0.723). 9 cases still fail (prompt_defect=2, hallucination=1, wrong_refusal=6). Each remaining failure is documented transparently below with root cause and evidence.

## Attached vs Rejected Tools
- **Attached Tools:** `['csv_lookup', 'calculator']`
- **Rejected `open_meteo_weather`:** Excluded by requirements architect as unnecessary for the brief.
- **Rejected `current_date`:** Excluded by requirements architect as unnecessary for the brief.

## Groundability Audit
- **Status:** `partially_grounded` (ratio=0.875)
- ⚠️ PARTIAL COVERAGE: 'estimated arrival dates, while refusing unapproved emergency credit overrides or legal/weather forecasts outside the docs' is only partially documented (Supplier_SLA_and_Pricing_Terms.pdf, Port_Freight_and_Customs_Schedule.pdf).

## Prompt Tournament Results (Position-Debiased)
| Candidate | Strategy | Tournament Score | Det Score | W / L / T |
| :--- | :--- | :---: | :---: | :---: |
| `cand_1_strict_grounded_citation` | strict_grounded_citation | 0.631 | 0.702 | 0 / 0 / 12 |
| `cand_3_concise_tool_specialist` | concise_tool_specialist | 0.615 | 0.677 | 0 / 0 / 12 |
| `cand_2_structured_policy_analyst` | structured_policy_analyst | 0.607 | 0.665 | 0 / 0 / 12 |

## Per-Iteration Scorecard
| Iter | Stage | Pass Rate | Composite | Groundedness | Fact Recall | Refusal Acc | Contract | Failures (R/P/H/W) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Baseline (v1) | 14/24 (58.3%) | 0.703 | 0.731 | 0.528 | 0.750 | 1.000 | 0/3/1/6 |
| 1 | Iteration 1 (wrong_refusal) | 15/24 (62.5%) | 0.723 | 0.731 | 0.577 | 0.750 | 1.000 | 0/2/1/6 |

## Targeted Optimization Log
### Iteration 1 — Target: `wrong_refusal` [ACCEPTED]
- **Diagnosis:** Dominant failure is `wrong_refusal` for out-of-scope questions.
- **Failure IDs Addressed:** `['eval_17', 'eval_18', 'eval_19', 'eval_20', 'eval_21']`
- **Score Before -> After:** `0.703` -> `0.723` (Pass rate: `58.3%` -> `62.5%`)
  - Change `retrieval.min_score_threshold`: `0.08` -> `0.12` (Raise score gate to enforce refusal on out-of-scope queries)
  - Change `system_prompt`: `(len=733)` -> `+ If the question is out of scope, refuse it based on provided documents and current date.` (Clarify answer-vs-refuse boundary)
- **Decision:** Accepted patch (+4.2pp pass rate, +0.020 composite score).

## What Still Fails & Why
### `eval_06` (answerable / single_doc) — `prompt_defect`
- **Question:** What is the total delivery lead time from supplier and transit times?
- **Expected:** `answer` | **Actual Refused:** `False` | **Score:** `0.823`
- **Why It Failed:** numeric inaccuracy (0.40)
- **Actual Excerpt:** The total delivery lead time equals supplier lead_time_days (from spares_catalogue.csv) PLUS transit time: +2 days for air freight or +14 days for sea freight.
### `eval_13` (edge_case / multi_doc_synthesis) — `hallucination`
- **Question:** What is the total landed cost if 10 units of Main Seawater Cooling Pump Impeller (SP-MV-101) are ordered from SUP-NORDIC with a unit price of $640?
- **Expected:** `answer` | **Actual Refused:** `False` | **Score:** `0.562`
- **Why It Failed:** Ungrounded claims/numbers detected: unverified numeric claim
- **Actual Excerpt:** $13,136.84
### `eval_15` (out_of_scope / near_domain_out_of_scope) — `wrong_refusal`
- **Question:** Can you provide a discount estimate for an emergency credit override to expedite the delivery of critical spare parts?
- **Expected:** `refuse` | **Actual Refused:** `False` | **Score:** `0.250`
- **Why It Failed:** Failed to refuse out-of-scope question.
- **Actual Excerpt:** Emergency aircraft-on-ground (AOG) priority dispatch adds a flat handling surcharge of USD 380 per purchase order and cuts supplier warehouse lead time by 50% (minimum 1 day). No discount is offered for emergency credit 
### `eval_18` (out_of_scope / near_domain_out_of_scope) — `wrong_refusal`
- **Question:** Can you provide a comparison of our current supplier pricing with those from other known competitors?
- **Expected:** `refuse` | **Actual Refused:** `False` | **Score:** `0.250`
- **Why It Failed:** Failed to refuse out-of-scope question.
- **Actual Excerpt:** The provided documents do not contain information on the pricing or terms of other known competitors, only details about SUP-NORDIC, SUP-SINGAPORE, and SUP-ROTTERDAM. Therefore, a comparison with other suppliers cannot b
### `eval_19` (out_of_scope / near_domain_out_of_scope) — `wrong_refusal`
- **Question:** How would you handle a situation where a supplier's SLA exceeds our current contract terms?
- **Expected:** `refuse` | **Actual Refused:** `False` | **Score:** `0.250`
- **Why It Failed:** Failed to refuse out-of-scope question.
- **Actual Excerpt:** If the Supplier’s Service Level Agreement (SLA) exceeds the current contract terms, it is important to escalate this issue through the established escalation channels as outlined in the Proc-Doc-01 document. The document
### `eval_20` (out_of_scope / near_domain_out_of_scope) — `wrong_refusal`
- **Question:** What are the legal implications of using a supplier outside our current contract terms?
- **Expected:** `refuse` | **Actual Refused:** `False` | **Score:** `0.250`
- **Why It Failed:** Failed to refuse out-of-scope question.
- **Actual Excerpt:** Using a supplier outside the current contract terms could lead to legal implications such as non-compliance with warranty, return policy, and payment terms. Additionally, it may violate specific conditions related to res
### `eval_21` (out_of_scope / near_domain_out_of_scope) — `wrong_refusal`
- **Question:** How would you handle a situation where a supplier's compliance certificate expires before the order is placed?
- **Expected:** `refuse` | **Actual Refused:** `False` | **Score:** `0.250`
- **Why It Failed:** Failed to refuse out-of-scope question.
- **Actual Excerpt:** If a supplier's compliance certificate expires before an order is placed, it must be renewed or replaced to ensure that all critical spare parts ship with valid EN 10204 3.1 material certificates and class type-approval 
### `eval_22` (out_of_scope / near_domain_out_of_scope) — `wrong_refusal`
- **Question:** Can you provide a discount estimate for spare parts that are outside our current supplier discount tiers and do not have compliance certificates?
- **Expected:** `refuse` | **Actual Refused:** `False` | **Score:** `0.250`
- **Why It Failed:** Failed to refuse out-of-scope question.
- **Actual Excerpt:** Spare parts that are outside your current supplier discount tiers and do not have compliance certificates cannot be approved. Please ensure all orders meet the minimum requirements as outlined in the provided documents.
### `eval_24` (edge_case / ambiguous_phrasing) — `prompt_defect`
- **Question:** In Marine_Technical_Compliance_and_Specs.pdf (2.1 Mandatory Class Certification for Critical Parts), what rule or value is specified for Effective Year 2026 Classification?
- **Expected:** `answer` | **Actual Refused:** `False` | **Score:** `0.675`
- **Why It Failed:** numeric inaccuracy (0.00)
- **Actual Excerpt:** For items marked as CRITICAL in the spares_catalogue.csv, they must ship with an EN 10204 3.1 material certificate and class type-approval dossier.

## Budget & Compute Telemetry
- **LLM Calls:** 175
- **Tokens Used:** 241,738 / 350,000 (prompt=211,143, completion=30,595)
- **Elapsed Time:** 901.5s / 900.0s
- **Early Stop Reason:** Latency budget reached (901.5s/900.0s)
