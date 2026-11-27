# Demo Corpus Synthesis Prompts

This file documents the committed prompts & deterministic specifications used to synthesize the two ForgeAI demonstration corpora (`demos/hr_policy/` and `demos/ship_spares/`), generated cleanly via `demos/scripts/build_demo_corpora.py`.

## Demo 1: Shore Staff HR Policy Corpus (`demos/hr_policy/docs/`)
> **Synthesis Prompt**:
> Author three realistic, multi-page corporate maritime shore-staff HR policy documents formatted as PDFs with numbered headings, concrete numeric entitlements, waiting periods, receipt caps, approval workflows, and deliberate cross-document dependencies:
> 1. `HR_Leave_and_Attendance_Policy.pdf` (Annual leave, carry-forward, probation, sick leave medical-note thresholds, parental & compassionate leave, core working hours, compensatory holiday time off).
> 2. `Remote_Work_and_Expense_Policy.pdf` (Hybrid remote eligibility linked to probation, home-office stipend, domestic vs international travel meal caps, lodging limits, mileage rates, itemized receipt deadlines, explicitly non-reimbursable items, cross-reference to leave rules during business travel).
> 3. `Code_of_Conduct_and_Benefits_Policy.pdf` (Ethics hotline, quarterly gift cap, conflict-of-interest disclosure deadline, employer/employee medical insurance co-pay split, annual training allowance, grade-based resignation notice periods).
> Do NOT include executive stock-option/equity vesting or cryptocurrency compensation so groundability and refusal guardrails can be tested authentically.

## Demo 2: Ship-Spares Procurement Corpus (`demos/ship_spares/docs/`)
> **Synthesis Prompt**:
> Author a maritime fleet spare-parts procurement dataset containing one structured CSV catalogue plus three supporting supplier/technical/freight PDFs with cross-table SKUs, supplier IDs, pricing discounts, lead times, and compliance limits:
> 1. `spares_catalogue.csv` (18 marine engine, pump, valve, turbocharger, and seal SKUs with `part_id`, `part_name`, `category`, `engine_model`, `supplier_id`, `unit_price_usd`, `moq`, `stock_qty`, `lead_time_days`, `weight_kg`, `criticality`).
> 2. `Supplier_SLA_and_Pricing_Terms.pdf` (Supplier volume discount tiers per `supplier_id`, warranty hours/months, RMA return window, payment net terms, AOG expedite fee).
> 3. `Marine_Technical_Compliance_and_Specs.pdf` (Class 3.1 certification requirement for `CRITICAL` parts in CSV, pressure/temperature limits for key SKUs, elastomer shelf life).
> 4. `Port_Freight_and_Customs_Schedule.pdf` (Air vs sea per-kg freight rates, minimum air fee, port customs handling fees by hub, transit lead-time adders combined with `lead_time_days`, dangerous goods restrictions).
