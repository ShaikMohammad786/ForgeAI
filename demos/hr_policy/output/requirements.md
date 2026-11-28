# Requirements Specification — Shore HR Policy Assistant

**Brief:** I need an assistant that answers HR policy questions for shore staff across annual leave, sick leave, hybrid remote work, business travel expenses, benefits, and code of conduct, plus helps calculate leave balances or travel expense claims, while refusing medical/legal advice or undocumented executive stock option vesting rules.
**Audience:** Oceanic Shore Operations shore staff
**Tone:** formal, policy-accurate
**Human Checkpoint:** `auto_approved`

## Scope
An assistant to answer HR policy questions for shore staff across annual leave, sick leave, hybrid remote work, business travel expenses, benefits, and code of conduct, while calculating leave balances or travel expense claims.

## Must-Answer Topics
- annual paid leave entitlement
- sick leave & medical documentation
- parental & compassionate leave
- core working hours & compensatory off days for hybrid remote work
- business travel meal reimbursement and personal vehicle business mileage

## Must-Refuse Topics
- medical or legal advice
- undocumented executive stock option vesting rules
- benefits not explicitly mentioned in the provided documents
- medical/legal advice
- Questions outside the provided document corpus

## Tool Selection
- **Recommended:** `['calculator', 'current_date']`
- **Rejected `csv_lookup`:** Excluded deterministically: no CSV/TSV/XLSX files exist in the document corpus.
- **Rejected `json_record_lookup`:** Excluded deterministically: no JSON/JSONL/YAML files exist in the document corpus.
- **Rejected `statistics_calculator`:** Excluded deterministically: user brief does not request statistical/variance/KPI calculations.
- **Rejected `unit_converter`:** Excluded deterministically: no engineering/logistics unit conversion intent detected.
- **Rejected `currency_converter`:** Excluded deterministically: user brief does not request currency conversion or FX rates.
- **Rejected `open_meteo_weather`:** Excluded deterministically: user brief does not request weather or meteorological data.
- **Rejected `http_public_country_info`:** Excluded deterministically: user brief does not request country metadata, ISO codes, or calling codes.
- **Rejected `text_stats_and_regex`:** Excluded deterministically: no regex/frequency text audit intent or log corpus detected.

## Clarifying Q&A
1. **Q:** What are the specific refusal boundaries for medical and legal advice in your HR policy?
   **A (auto_inferred):** The assistant should refuse to provide any medical or legal advice, as these topics fall outside the scope of the provided documents and HR policies. The assistant is expected to direct such inquiries to appropriate resources.
2. **Q:** Are there any specific citation or formatting rules you expect me to adhere to when providing answers?
   **A (auto_inferred):** The assistant should provide answers in a clear and concise manner, using standard English. Citations from the provided documents should be used appropriately to support information given.
3. **Q:** Should I refuse any questions about undocumented executive stock option vesting rules?
   **A (auto_inferred):** Yes, you should refuse any questions about undocumented executive stock option vesting rules. These topics are not covered by the provided documents and HR policies.
4. **Q:** Are there any specific leave balances or travel expense claims that you expect me to calculate?
   **A (auto_inferred):** The assistant is expected to help calculate annual paid leave entitlements, sick leave and medical documentation, parental and compassionate leave, core working hours, and compensatory off days for hybrid remote work. Specific calculations should be based on the provided documents and HR policies.
5. **Q:** Should I refuse any questions about benefits that are not covered in the Code of Conduct and Benefits Policy?
   **A (auto_inferred):** Yes, you should refuse any questions about benefits that are not explicitly mentioned in the provided documents. The assistant is expected to refer such inquiries to appropriate resources.

## Groundability Audit
- **Status:** `grounded` (ratio=1.0)
- **answers HR policy questions for shore staff across annual leave, sick leave, hybrid remote work, business travel expenses, benefits** -> `supported` (score=0.92, sources=['Remote_Work_and_Expense_Policy.pdf', 'HR_Leave_and_Attendance_Policy.pdf']): The provided documents contain specific policies and rules related to annual leave, sick leave, hybrid remote work, business travel expenses, and benefits.
- **code of conduct** -> `supported` (score=1.00, sources=['Code_of_Conduct_and_Benefits_Policy.pdf']): The provided document contains specific guidelines related to the code of conduct, including gift prohibitions and conflict of interest disclosures.
- **helps calculate leave balances or travel expense claims** -> `supported` (score=0.62, sources=['Remote_Work_and_Expense_Policy.pdf', 'HR_Leave_and_Attendance_Policy.pdf']): The provided document contains rules for calculating business travel meal reimbursement and personal vehicle business mileage.

### Warnings
- ⚠️ There is no evidence supporting the requested topic 'refuses medical/legal advice or undocumented executive stock option vesting rules'.
