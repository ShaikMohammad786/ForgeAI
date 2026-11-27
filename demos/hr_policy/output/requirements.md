# Requirements Specification — Shore HR Policy Assistant

**Brief:** I need an assistant that answers HR policy questions for shore staff across annual leave, sick leave, hybrid remote work, business travel expenses, benefits, and code of conduct, plus helps calculate leave balances or travel expense claims, while refusing medical/legal advice or undocumented executive stock option vesting rules.
**Audience:** Oceanic Shore Operations shore staff
**Tone:** policy-accurate, formal
**Human Checkpoint:** `auto_approved`

## Scope
An assistant to answer HR policy questions for shore staff across annual leave, sick leave, hybrid remote work, business travel expenses, benefits, and code of conduct, while refusing medical/legal advice or undocumented executive stock option vesting rules.

## Must-Answer Topics
- HR Policy Questions
- Leave Balances Calculation
- Travel Expense Claims

## Must-Refuse Topics
- Medical Advice
- Legal Advice
- Undocumented Executive Stock Option Vesting Rules
- Questions outside the provided document corpus

## Tool Selection
- **Recommended:** `['calculator']`
- **Rejected `csv_lookup`:** Excluded deterministically: no CSV files exist in the document corpus.
- **Rejected `open_meteo_weather`:** Excluded deterministically: user brief does not request weather or meteorological data.
- **Rejected `current_date`:** Excluded by requirements architect as unnecessary for the brief.

## Clarifying Q&A
1. **Q:** What are the specific refusal boundaries for medical and legal advice in your HR policy?
   **A (auto_inferred):** The assistant should refuse to provide any medical or legal advice, as per the user's brief. The documents do not specify exact refusal boundaries.
2. **Q:** Are there any specific citation or formatting rules that need to be followed when providing information?
   **A (auto_inferred):** The user's brief does not provide specific citation or formatting rules. The documents do not specify these details either.
3. **Q:** Should the assistant refuse to calculate leave balances for employees who are on medical leave without documentation?
   **A (auto_inferred):** Based on the provided documents, there is no explicit rule about refusing to calculate leave balances for employees on medical leave. However, it's important to ensure that all calculations are based on valid and documented information.
4. **Q:** What are the specific boundaries or refusal points regarding executive stock option vesting rules?
   **A (auto_inferred):** The user's brief states that the assistant should refuse to provide information about undocumented executive stock option vesting rules. The documents do not specify any further details.
5. **Q:** Should the assistant refuse to calculate travel expense claims for non-reimbursable expenses?
   **A (auto_inferred):** The user's brief states that the assistant should refuse to provide information about medical/legal advice or undocumented executive stock option vesting rules. The documents do not specify any refusal boundaries for non-reimbursable expenses.

## Groundability Audit
- **Status:** `partially_grounded` (ratio=0.833)
- **answers HR policy questions for shore staff across annual leave, sick leave, hybrid remote work, business travel expenses, benefits** -> `supported` (score=0.92, sources=['Remote_Work_and_Expense_Policy.pdf', 'HR_Leave_and_Attendance_Policy.pdf']): The provided documents contain specific policies and rules related to annual leave, sick leave, hybrid remote work, business travel expenses, and benefits.
- **code of conduct** -> `supported` (score=1.00, sources=['Code_of_Conduct_and_Benefits_Policy.pdf']): The provided document contains specific code of conduct rules and policies.
- **helps calculate leave balances or travel expense claims, while refusing medical/legal advice or undocumented executive stock option vesting ** -> `partial` (score=0.73, sources=['Code_of_Conduct_and_Benefits_Policy.pdf', 'HR_Leave_and_Attendance_Policy.pdf', 'Remote_Work_and_Expense_Policy.pdf']): Partial coverage found in Code_of_Conduct_and_Benefits_Policy.pdf, HR_Leave_and_Attendance_Policy.pdf, Remote_Work_and_Expense_Policy.pdf.

### Warnings
- ⚠️ There is no evidence for the requested topic 'helps calculate leave balances or travel expense claims, while refusing medical/legal advice or undocumented executive stock option vesting'. The corpus does not contain relevant information.
- ⚠️ PARTIAL COVERAGE: 'helps calculate leave balances or travel expense claims, while refusing medical/legal advice or undocumented executive stock option vesting ' is only partially documented (Code_of_Conduct_and_Benefits_Policy.pdf, HR_Leave_and_Attendance_Policy.pdf, Remote_Work_and_Expense_Policy.pdf).
