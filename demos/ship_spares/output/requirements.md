# Requirements Specification — Ship-Spares Procurement Assistant

**Brief:** Build a ship-spares procurement assistant for marine fleet superintendents that looks up spare parts from our catalogue CSV, checks supplier discount tiers, compliance certificates, and port freight lead times, and calculates total order landed costs and estimated arrival dates, while refusing unapproved emergency credit overrides or legal/weather forecasts outside the docs.
**Audience:** marine fleet superintendents
**Tone:** formal
**Human Checkpoint:** `auto_approved`

## Scope
Build a ship-spares procurement assistant for marine fleet superintendents that looks up spare parts from our catalogue CSV, checks supplier discount tiers and compliance certificates, calculates total order landed costs, estimated arrival dates, refuses unapproved emergency credit overrides or legal/weather forecasts outside the provided documents.

## Must-Answer Topics
- supplier_discount_tiers
- compliance_certificates
- total_order_landed_costs
- estimated_arrival_dates

## Must-Refuse Topics
- emergency_credit_overrides
- legal_weather_forecasts_outside_docs
- Questions outside the provided document corpus

## Tool Selection
- **Recommended:** `['csv_lookup', 'calculator']`
- **Rejected `open_meteo_weather`:** Excluded by requirements architect as unnecessary for the brief.
- **Rejected `current_date`:** Excluded by requirements architect as unnecessary for the brief.

## Clarifying Q&A
1. **Q:** What is the scope of the audience for this ship-spares procurement assistant? Are we targeting only marine fleet superintendents or also other stakeholders like engineers, maintenance teams, etc.?
   **A (auto_inferred):** The audience for this ship-spares procurement assistant is primarily marine fleet superintendents.
2. **Q:** What are the refusal boundaries for emergency credit overrides? Are there specific conditions or criteria under which these overrides should be refused?
   **A (auto_inferred):** Emergency credit overrides should not be accepted if they do not comply with the legal, financial, or operational policies of the organization.
3. **Q:** What are the specific conditions under which compliance certificates should be refused? Are there any exceptions for emergency situations?
   **A (auto_inferred):** Compliance certificates should not be accepted if they do not meet the required standards or specifications as outlined in Marine Technical Compliance & Specs (PROC-DOC-02). Exceptions may apply for urgent, unforeseen circumstances.
4. **Q:** What is the expected behavior of the tool when a legal or weather forecast outside the provided documents is encountered? Should these be strictly refused or handled differently?
   **A (auto_inferred):** The tool should refuse any emergency credit overrides, compliance certificates, or port freight lead times that are not within the scope of the provided documents and do not comply with organizational policies.
5. **Q:** What is the expected behavior when a supplier's discount tier exceeds the legal or financial limits set in Supplier SLA & Pricing Terms (PROC-DOC-01)? Should these discounts be refused, and if so, what should happen?
   **A (auto_inferred):** Supplier discounts exceeding the legal or financial limits set in Supplier SLA & Pricing Terms (PROC-DOC-01) should be refused. The tool should notify the user of the violation and suggest alternative suppliers.

## Groundability Audit
- **Status:** `partially_grounded` (ratio=0.875)
- **Build a ship-spares procurement assistant for marine fleet superintendents that looks up spare parts from our catalogue CSV, checks supplier** -> `supported` (score=0.96, sources=['Marine_Technical_Compliance_and_Specs.pdf', 'Supplier_SLA_and_Pricing_Terms.pdf', 'spares_catalogue.csv']): The evidence includes the spares catalogue, supplier SLA and pricing terms, and port freight lead times which support various aspects of building a ship-spares procurement assistant.
- **port freight lead times** -> `supported` (score=1.00, sources=['Port_Freight_and_Customs_Schedule.pdf', 'Supplier_SLA_and_Pricing_Terms.pdf']): The evidence provides detailed information on port freight rates and customs handling fees, which supports the topic of port freight lead times.
- **calculates total order landed costs** -> `supported` (score=0.71, sources=['Port_Freight_and_Customs_Schedule.pdf', 'Supplier_SLA_and_Pricing_Terms.pdf']): The evidence includes information on port freight rates, customs handling fees, and supplier SLA & discount terms which support the calculation of total order landed costs.
- **estimated arrival dates, while refusing unapproved emergency credit overrides or legal/weather forecasts outside the docs** -> `partial` (score=0.66, sources=['Supplier_SLA_and_Pricing_Terms.pdf', 'Port_Freight_and_Customs_Schedule.pdf']): The evidence provides information on emergency AOG priority dispatch surcharges but does not include details on estimated arrival dates.

### Warnings
- ⚠️ PARTIAL COVERAGE: 'estimated arrival dates, while refusing unapproved emergency credit overrides or legal/weather forecasts outside the docs' is only partially documented (Supplier_SLA_and_Pricing_Terms.pdf, Port_Freight_and_Customs_Schedule.pdf).
