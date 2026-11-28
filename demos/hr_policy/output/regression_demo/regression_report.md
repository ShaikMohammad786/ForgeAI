# ForgeAI Regression Report — ShoreHRPolicyAssistant

- **Added Files:** `[]`
- **Modified Files:** `['HR_Leave_and_Attendance_Policy.pdf', 'Remote_Work_and_Expense_Policy.pdf']`
- **Removed Files:** `[]`
- **Pass Rate Change:** `66.7%` -> `33.3%`

## Newly Broken (8)
- `eval_01` (prompt_defect): What is the maximum number of unused annual leave days that shore staff can carry forward into the next calendar year? — missed expected facts despite retrieval (recall=0.25); numeric inaccuracy (0.00)
- `eval_02` (prompt_defect): What is the total amount of paid sick leave that full-time permanent shore staff receive per calendar year? — numeric inaccuracy (0.33)
- `eval_05` (retrieval_miss): What is the maximum number of unused annual leave days that can be carried forward into the next calendar year for new hires during their probation period? — Required facts/sources (['Code_of_Conduct_and_Benefits_Policy.pdf', 'HR_Leave_and_Attendance_Policy.pdf', 'Remote_Work_and_Expense_Policy.pdf']) missing from top-5 retrieved chunks (top_score=0.83).
- `eval_06` (wrong_refusal): How many days of fully paid parental leave are available to primary caregivers after 6 months of continuous shore employment? — False refusal on answerable question.
- `eval_07` (wrong_refusal): What is the standard shore office hours, including mandatory core collaboration hours? — False refusal on answerable question.
- `eval_08` (wrong_refusal): How many days of paid leave do secondary caregivers receive for parental leave? — False refusal on answerable question.
- `eval_11` (wrong_refusal): What is the reimbursement rate for personal vehicle business mileage? — False refusal on answerable question.
- `eval_12` (hallucination): How many days of paid sick leave does a shore employee receive per calendar year, including any carry-over days if they have 12 days already used? — Ungrounded claims/numbers detected: unverified numeric claim

## Now Ungroundable After Document Changes (4)
- `eval_06`: How many days of fully paid parental leave are available to primary caregivers after 6 months of continuous shore employment? (old docs: ['HR_Leave_and_Attendance_Policy.pdf'])
- `eval_07`: What is the standard shore office hours, including mandatory core collaboration hours? (old docs: ['HR_Leave_and_Attendance_Policy.pdf'])
- `eval_09`: What is the maximum amount for home-office ergonomic equipment stipend per 24 months? (old docs: ['Remote_Work_and_Expense_Policy.pdf'])
- `eval_11`: What is the reimbursement rate for personal vehicle business mileage? (old docs: ['Remote_Work_and_Expense_Policy.pdf'])

## Newly Fixed (0)

## Still Failing (8)
- `eval_03` (prompt_defect): What is the probation notice period for Job Band D and above shore employees?
- `eval_04` (wrong_refusal): What is the total amount of paid sick leave that new hires can receive during their probation period?
- `eval_09` (hallucination): What is the maximum amount for home-office ergonomic equipment stipend per 24 months?
- `eval_14` (hallucination): How much of the annual professional training allowance (USD 1,500) must shore staff pay out if they receive full-time group medical insurance coverage?
- `eval_17` (retrieval_miss): How many days of paid annual leave do full-time shore staff accrue per calendar year, including any carry-over days if they have 24 days already accrued?
- `eval_19` (wrong_refusal): What is the legal requirement for sick leave in our country?
- `eval_20` (prompt_defect): According to Code_of_Conduct_and_Benefits_Policy.pdf, what are the details regarding 3.1 Gifts, Hospitality & Conflict of Interest?
- `eval_24` (retrieval_miss): According to HR_Leave_and_Attendance_Policy.pdf, what are the details regarding (Hr-Pol-01)?
