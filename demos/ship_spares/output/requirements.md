# Requirements Specification — Ship-Spares Procurement Assistant

**Brief:** Build a ship-spares procurement assistant for marine fleet superintendents that looks up spare parts from our catalogue CSV, checks supplier discount tiers, compliance certificates, and port freight lead times, and calculates total order landed costs and estimated arrival dates, while refusing unapproved emergency credit overrides or legal/weather forecasts outside the docs.
**Audience:** marine fleet superintendents
**Tone:** formal, policy-accurate
**Human Checkpoint:** `auto_approved`

## Scope
Build a ship-spares procurement assistant for marine fleet superintendents that looks up spare parts from our catalogue CSV, checks supplier discount tiers and compliance certificates, calculates total order landed costs based on unit prices and quantities, refuses unapproved emergency credit overrides or legal/weather forecasts outside the provided documents, and estimates arrival dates based on port freight lead times.

## Must-Answer Topics
- Supplier Discount Tiers
- Compliance Certificates
- Total Order Landed Costs Calculation
- Emergency Credit Overrides Refusal
- Legal/Weather Forecasts Outside Provided Documents
- Port Freight Lead Times

## Must-Refuse Topics
- External Legal or Weather Forecast Sources
- Unapproved Emergency Credit Overrides
- Legal/Weather Forecasts Outside Provided Documents
- unapproved emergency credit overrides
- legal/weather forecasts outside the docs
- Questions outside the provided document corpus

## Tool Selection
- **Recommended:** `['csv_lookup', 'calculator', 'unit_converter', 'current_date']`
- **Rejected `json_record_lookup`:** Excluded deterministically: no JSON/JSONL/YAML files exist in the document corpus.
- **Rejected `statistics_calculator`:** Excluded deterministically: user brief does not request statistical/variance/KPI calculations.
- **Rejected `currency_converter`:** Excluded deterministically: user brief does not request currency conversion or FX rates.
- **Rejected `open_meteo_weather`:** Excluded deterministically: user brief does not request weather or meteorological data.
- **Rejected `http_public_country_info`:** Excluded deterministically: user brief does not request country metadata, ISO codes, or calling codes.
- **Rejected `text_stats_and_regex`:** Excluded deterministically: no regex/frequency text audit intent or log corpus detected.

## Clarifying Q&A
1. **Q:** What is the scope of audience for this ship-spares procurement assistant? Are we limited to marine fleet superintendents or should it also include other stakeholders?
   **A (auto_inferred):** The audience for this ship-spares procurement assistant is limited to marine fleet superintendents.
2. **Q:** What specific emergency credit overrides should the system refuse? Are there certain types of emergencies or situations where we can allow exceptions?
   **A (auto_inferred):** The system should refuse all unapproved emergency credit overrides.
3. **Q:** Should the system strictly refuse any legal or weather forecasts that are outside of the provided documents? Or is there a need to consider external sources as well?
   **A (auto_inferred):** The system should strictly refuse any legal or weather forecasts that are outside of the provided documents.
4. **Q:** What is the expected behavior for port freight lead times if they fall outside the documented schedule in Port_Freight_and_Customs_Schedule.pdf? Should we assume a default value or refuse such orders?
   **A (auto_inferred):** The system should refuse any port freight lead times that are outside of the documented schedule.
5. **Q:** Should the system calculate total order landed costs based on unit prices and quantities from spares_catalogue.csv, or should we also consider any additional fees or taxes that might apply?
   **A (auto_inferred):** The system should calculate total order landed costs based solely on unit prices and quantities from spares_catalogue.csv.

## Groundability Audit
- **Status:** `partially_grounded` (ratio=0.875)
- **Build a ship-spares procurement assistant for marine fleet superintendents that looks up spare parts from our catalogue CSV, checks supplier** -> `supported` (score=0.96, sources=['Marine_Technical_Compliance_and_Specs.pdf', 'Supplier_SLA_and_Pricing_Terms.pdf', 'spares_catalogue.csv']): The provided documents cover the technical compliance, supplier SLA and pricing terms, port freight lead times, and spare parts catalogue, which are all necessary for building a ship-spares procurement assistant.
- **port freight lead times** -> `supported` (score=1.00, sources=['Port_Freight_and_Customs_Schedule.pdf', 'Supplier_SLA_and_Pricing_Terms.pdf']): The provided document contains specific details about port freight rates and customs handling fees, which are relevant to calculating port freight lead times.
- **calculates total order landed costs** -> `supported` (score=0.71, sources=['Port_Freight_and_Customs_Schedule.pdf', 'Supplier_SLA_and_Pricing_Terms.pdf']): The provided documents contain information about port freight rates, customs handling fees, and supplier SLA and pricing terms, which are essential for calculating total order landed costs.
- **estimated arrival dates** -> `partial` (score=0.31, sources=['Port_Freight_and_Customs_Schedule.pdf', 'spares_catalogue.csv']): The provided document contains details about port freight rates and customs handling fees, which can be used to estimate the arrival date of orders.

### Warnings
- ⚠️ No explicit gaps were found in the corpus for any requested topic. All topics are supported by at least one relevant source.
- ⚠️ PARTIAL COVERAGE: 'estimated arrival dates' is only partially documented (Port_Freight_and_Customs_Schedule.pdf, spares_catalogue.csv).
