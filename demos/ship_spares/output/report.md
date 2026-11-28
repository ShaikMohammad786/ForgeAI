# ForgeAI Quality Report — Ship_Spares_Agent

**Brief:** Build a ship-spares procurement assistant for marine fleet superintendents that looks up spare parts from our catalogue CSV, checks supplier discount tiers, compliance certificates, and port freight lead times, and calculates total order landed costs and estimated arrival dates, while refusing unapproved emergency credit overrides or legal/weather forecasts outside the docs.
**Provider / Generator Model:** `ollama` / `qwen2.5:3b`
**Judge Model:** `qwen2.5:1.5b`
**Final Pass Rate:** **70.8%** (Composite Score: **0.810**)

## Honest Assessment
Agent 'Ship_Spares_Agent' achieved 17/24 (70.8%) pass rate (composite=0.810). 7 cases still fail (prompt_defect=5, hallucination=1, wrong_refusal=1). Each remaining failure is documented transparently below with root cause and evidence.

## Attached vs Rejected Tools
- **Attached Standard Tools:** `['csv_lookup', 'calculator', 'unit_converter', 'current_date']`
- **Synthesized Custom Macros (AST-verified):** `['lookup_critical_spares (tabular_filter)', 'calculate_total_landed_cost (formula)']`
- **Custom Macro `lookup_critical_spares`:** Filter spares_catalogue.csv to find critical spare parts. — `spares_catalogue.csv` filters={'criticality': 'CRITICAL'}
- **Custom Macro `calculate_total_landed_cost`:** Calculate total landed cost based on unit prices, quantities, and discounts. — `unit_price * quantity * (1 - discount_pct / 100) + freight_usd`
- **Rejected `json_record_lookup`:** Excluded deterministically: no JSON/JSONL/YAML files exist in the document corpus.
- **Rejected `statistics_calculator`:** Excluded deterministically: user brief does not request statistical/variance/KPI calculations.
- **Rejected `currency_converter`:** Excluded deterministically: user brief does not request currency conversion or FX rates.
- **Rejected `open_meteo_weather`:** Excluded deterministically: user brief does not request weather or meteorological data.
- **Rejected `http_public_country_info`:** Excluded deterministically: user brief does not request country metadata, ISO codes, or calling codes.
- **Rejected `text_stats_and_regex`:** Excluded deterministically: no regex/frequency text audit intent or log corpus detected.

## Groundability Audit
- **Status:** `partially_grounded` (ratio=0.875)
- ⚠️ No explicit gaps were found in the corpus for any requested topic. All topics are supported by at least one relevant source.
- ⚠️ PARTIAL COVERAGE: 'estimated arrival dates' is only partially documented (Port_Freight_and_Customs_Schedule.pdf, spares_catalogue.csv).

## Prompt Tournament Results (Position-Debiased)
| Candidate | Strategy | Tournament Score | Det Score | W / L / T |
| :--- | :--- | :---: | :---: | :---: |
| `cand_2_structured_policy_analyst` | structured_policy_analyst | 0.677 | 0.772 | 0 / 0 / 8 |
| `cand_3_concise_tool_specialist` | concise_tool_specialist | 0.646 | 0.691 | 1 / 0 / 7 |
| `cand_1_strict_grounded_citation` | strict_grounded_citation | 0.603 | 0.691 | 0 / 1 / 7 |

## Per-Iteration Scorecard
| Iter | Stage | Pass Rate | Composite | Groundedness | Fact Recall | Refusal Acc | Contract | Failures (R/P/H/W) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Baseline (v1) | 14/24 (58.3%) | 0.791 | 0.940 | 0.616 | 0.958 | 1.000 | 0/8/1/1 |
| 1 | Iteration 1 (prompt_defect) | 17/24 (70.8%) | 0.810 | 0.940 | 0.634 | 0.958 | 1.000 | 0/5/1/1 |

## Final Pass Rate by Question Category
| Category | Passed | Total | Pass Rate | Mean Composite |
| :--- | :---: | :---: | :---: | :---: |
| `answerable` | 11 | 14 | 78.6% | 0.836 |
| `out_of_scope` | 1 | 2 | 50.0% | 0.625 |
| `edge_case` | 5 | 8 | 62.5% | 0.812 |

## Targeted Optimization Log
### Iteration 1 — Target: `prompt_defect` [ACCEPTED]
- **Diagnosis:** Dominant failure is retrieval misses due to low recall and score thresholds.
- **Failure IDs Addressed:** `['eval_10', 'eval_12', 'eval_13', 'eval_16', 'eval_18']`
- **Score Before -> After:** `0.791` -> `0.810` (Pass rate: `58.3%` -> `70.8%`)
  - Change `system_prompt`: `(len=923)` -> `+ FORMAT & COMPLETENESS RULE: Include all key facts, exact numbers, and at least one valid filename in citations; return s` (Repair prompt compliance and fact/citation completeness)
  - Change `retrieval.top_k`: `5` -> `6` (Provide +1 chunk for multi-document synthesis)
- **Decision:** Accepted patch (+12.5pp pass rate, +0.020 composite score).

## What Still Fails & Why
### `eval_10` (answerable / csv_lookup) — `prompt_defect`
- **Question:** What is the minimum order quantity for the Lube Oil Purifier Bowl Disc Stack?
- **Expected:** `answer` | **Actual Refused:** `False` | **Score:** `0.735`
- **Why It Failed:** numeric inaccuracy (0.40)
- **Actual Excerpt:** The Minimum Order Quantity (MOQ) for the Lube Oil Purifier Bowl Disc Stack (part_id: SP-PM-501) is 1.
### `eval_12` (edge_case / multi_doc_synthesis) — `hallucination`
- **Question:** What is the total landed cost for part SP-VL-204 if it's ordered with supplier SUP-NORDIC, has a quantity of 3 and a unit price of $1180?
- **Expected:** `answer` | **Actual Refused:** `False` | **Score:** `0.565`
- **Why It Failed:** Ungrounded claims/numbers detected: unverified numeric claim
- **Actual Excerpt:** $3458.00
### `eval_13` (edge_case / multi_doc_synthesis) — `prompt_defect`
- **Question:** What is the total landed cost for part SP-MV-104 if it's ordered with supplier SUP-SINGAPORE, has a quantity of 5 and a unit price of $920?
- **Expected:** `answer` | **Actual Refused:** `False` | **Score:** `0.574`
- **Why It Failed:** missed expected facts despite retrieval (recall=0.12); numeric inaccuracy (0.20)
- **Actual Excerpt:** $4372
### `eval_15` (out_of_scope / near_domain_out_of_scope | unrelated_domain) — `wrong_refusal`
- **Question:** Could you provide me with a list of all parts that are currently out of stock?
- **Expected:** `refuse` | **Actual Refused:** `False` | **Score:** `0.250`
- **Why It Failed:** Failed to refuse out-of-scope question.
- **Actual Excerpt:** The parts currently out of stock are: Main Seawater Cooling Pump Impeller (part_id: SP-MV-101), Bronze Butterfly Sea Chest Valve DN150 (part_id: SP-MV-104), and High-Pressure Fuel Injection Valve (part_id: SP-VL-204).
### `eval_16` (answerable / single_doc) — `prompt_defect`
- **Question:** According to Marine_Technical_Compliance_and_Specs.pdf, what are the details regarding 2.1 Mandatory Class Certification for Critical Parts?
- **Expected:** `answer` | **Actual Refused:** `False` | **Score:** `0.675`
- **Why It Failed:** numeric inaccuracy (0.00)
- **Actual Excerpt:** Every spare part marked criticality=CRITICAL in spares_catalogue.csv MUST ship with an EN 10204 3.1 material certificate and class type-approval dossier.
### `eval_18` (edge_case / ambiguous_phrasing) — `prompt_defect`
- **Question:** In Marine_Technical_Compliance_and_Specs.pdf (2.1 Mandatory Class Certification for Critical Parts), what rule or value is specified for Operating Limits Material Specifications?
- **Expected:** `answer` | **Actual Refused:** `False` | **Score:** `0.675`
- **Why It Failed:** numeric inaccuracy (0.00)
- **Actual Excerpt:** The document specifies that items marked STANDARD require only supplier Certificate of Conformity (CoC) unless ordered for dry-dock class renewal, and it also provides details on the operating limits and material specifi
### `eval_20` (answerable / single_doc) — `prompt_defect`
- **Question:** According to Port_Freight_and_Customs_Schedule.pdf, what are the details regarding 3.1 Air & Sea Freight Rate Structure?
- **Expected:** `answer` | **Actual Refused:** `False` | **Score:** `0.675`
- **Why It Failed:** numeric inaccuracy (0.00)
- **Actual Excerpt:** The air courier freight rate is USD 6.50 per kg for total shipment weight (unit_weight_kg * quantity from spares_catalogue.csv), with a minimum charge of USD 180 per shipment. Consolidated sea freight is charged at USD 1

## Budget & Compute Telemetry
- **LLM Calls:** 166
- **Tokens Used:** 289,716 / 400,000 (prompt=257,295, completion=32,421)
- **Elapsed Time:** 1176.2s / 1200.0s
- **Early Stop Reason:** None (completed cleanly)
