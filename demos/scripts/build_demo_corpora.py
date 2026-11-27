"""Generate realistic multi-page PDFs and CSV datasets for Demo 1 & Demo 2."""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_pdf(out_path: Path, title: str, doc_meta: str, sections: list[tuple[str, list[str]]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=6,
        textColor=colors.HexColor("#0F2942"),
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=14,
    )
    h_style = ParagraphStyle(
        "SecHeading",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.HexColor("#1E3A8A"),
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )

    story = [Paragraph(title, title_style), Paragraph(doc_meta, meta_style)]
    for heading, paragraphs in sections:
        story.append(Paragraph(heading, h_style))
        for p in paragraphs:
            story.append(Paragraph(p, body_style))
        story.append(Spacer(1, 6))

    doc.build(story)


def build_hr_demo() -> None:
    base = REPO_ROOT / "demos" / "hr_policy"
    docs = base / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    (base / "brief.txt").write_text(
        "I need an assistant that answers HR policy questions for shore staff across annual leave, "
        "sick leave, hybrid remote work, business travel expenses, benefits, and code of conduct, "
        "plus helps calculate leave balances or travel expense claims, while refusing medical/legal "
        "advice or undocumented executive stock option vesting rules.",
        encoding="utf-8",
    )

    _write_pdf(
        docs / "HR_Leave_and_Attendance_Policy.pdf",
        "Oceanic Shore Operations — Leave & Attendance Policy (HR-POL-01)",
        "Effective Date: January 1, 2026 | Version 3.2 | Scope: Full-Time & Fixed-Term Shore Staff",
        [
            (
                "1.1 Annual Paid Leave Entitlement",
                [
                    "Full-time permanent shore staff accrue 24 working days of paid annual leave per calendar year (2.0 days per completed month of service).",
                    "During the initial 90-day probationary period, new hires accrue leave but may only take paid annual leave after completing 60 calendar days unless pre-approved in writing by HR.",
                    "Employees may carry forward a maximum of 5 unused annual leave days into the next calendar year. Any carried-forward days not used by March 31 are forfeited without cash payout.",
                ],
            ),
            (
                "1.2 Sick Leave & Medical Documentation",
                [
                    "Shore employees receive 12 days of paid sick leave per calendar year without carryover.",
                    "A licensed physician's medical certificate is mandatory for any sick leave absence exceeding 3 consecutive working days, or if sick leave occurs immediately before or after a public holiday.",
                    "For absences of 1 to 3 days, self-certification in the HR portal before 10:00 local time on the first day is sufficient.",
                ],
            ),
            (
                "1.3 Parental & Compassionate Leave",
                [
                    "Primary caregivers are entitled to 16 weeks of fully paid parental leave after 6 months of continuous shore employment. Secondary caregivers receive 4 weeks of paid leave.",
                    "Compassionate (bereavement) leave provides 5 consecutive paid working days for the loss of an immediate family member (spouse, child, parent, or sibling).",
                ],
            ),
            (
                "1.4 Core Working Hours & Travel Compensatory Off",
                [
                    "Standard shore office hours are 09:00 to 17:30 Monday through Friday with mandatory core collaboration hours from 10:00 to 15:30.",
                    "When shore staff perform mandatory weekend business travel under Remote_Work_and_Expense_Policy.pdf, 1 compensatory day off is granted for each full weekend day travelled, usable within 45 days.",
                ],
            ),
        ],
    )

    _write_pdf(
        docs / "Remote_Work_and_Expense_Policy.pdf",
        "Oceanic Shore Operations — Hybrid Remote Work & Travel Expense Policy (HR-POL-02)",
        "Effective Date: January 1, 2026 | Version 2.4 | Scope: Shore Staff",
        [
            (
                "2.1 Hybrid Remote Work Eligibility",
                [
                    "Permanent shore employees who have passed the 90-day probation period defined in HR_Leave_and_Attendance_Policy.pdf may work remotely up to 2 days per week with line manager approval.",
                    "Tuesdays and Thursdays are designated anchor office days; remote days are normally scheduled on Mondays, Wednesdays, or Fridays.",
                    "Working remotely from outside the country of employment requires prior written VP HR and tax compliance clearance at least 21 days in advance.",
                ],
            ),
            (
                "2.2 Home Office Equipment Stipend",
                [
                    "Eligible hybrid shore staff may claim a one-time home-office ergonomic equipment stipend of up to USD 450 once every 24 months (chair, external monitor, desk lamp, or headset).",
                    "Items purchased under the stipend remain company property if employment ends within 12 months of reimbursement.",
                ],
            ),
            (
                "2.3 Business Travel Meal, Lodging & Mileage Limits",
                [
                    "Business travel meal reimbursement is capped at USD 55 per day for domestic travel and USD 75 per day for international travel. Per-diem advances are not issued; actual itemized receipts are mandatory.",
                    "Standard hotel accommodation is capped at USD 220 per night excluding taxes unless written CFO exception is obtained for peak port conferences.",
                    "Personal vehicle business mileage is reimbursed at USD 0.65 per kilometer.",
                ],
            ),
            (
                "2.4 Submission Deadlines & Non-Reimbursable Expenses",
                [
                    "All expense claims with itemized receipts must be submitted within 14 calendar days after trip completion.",
                    "Strictly non-reimbursable items include alcohol, hotel mini-bar charges, in-room movies, traffic fines, personal laundry on trips under 4 nights, and business/first-class flight upgrades without VP approval.",
                ],
            ),
        ],
    )

    _write_pdf(
        docs / "Code_of_Conduct_and_Benefits_Policy.pdf",
        "Oceanic Shore Operations — Code of Conduct, Ethics & Benefits (HR-POL-03)",
        "Effective Date: January 1, 2026 | Version 4.0 | Scope: All Shore Employees",
        [
            (
                "3.1 Gifts, Hospitality & Conflict of Interest",
                [
                    "Shore staff must never solicit gifts from suppliers or port agents. Unsolicited nominal gifts or hospitality exceeding USD 50 in cumulative value per vendor per calendar quarter must be declined or surrendered to Compliance.",
                    "Cash or cash-equivalent gifts (including gift cards) of any value are strictly prohibited.",
                    "Actual or potential conflicts of interest (e.g., family ownership in a marine supplier) must be disclosed in writing to Compliance within 10 business days.",
                ],
            ),
            (
                "3.2 Ethics Reporting & Anti-Retaliation",
                [
                    "Suspected policy breaches, safety falsification, or harassment must be reported via the confidential Ethics Hotline at ethics@oceanic-shore.example or directly to the Chief People Officer.",
                    "Retaliation against good-faith whistleblowers is grounds for immediate summary dismissal.",
                ],
            ),
            (
                "3.3 Medical Insurance & Annual Learning Allowance",
                [
                    "Company group medical insurance covers shore staff and eligible dependents with an 80% employer / 20% employee premium co-pay split.",
                    "Full-time shore staff receive an annual professional training and certification allowance of USD 1,500 per calendar year, subject to line manager pre-approval and course completion proof.",
                ],
            ),
            (
                "3.4 Resignation Notice Periods",
                [
                    "After probation, standard written resignation notice is 30 calendar days for Job Bands A through C and 60 calendar days for Job Band D and above.",
                    "Unused accrued annual leave (excluding forfeited prior-year days past March 31 per HR_Leave_and_Attendance_Policy.pdf) is paid out on final settlement.",
                ],
            ),
        ],
    )


def build_ship_spares_demo() -> None:
    base = REPO_ROOT / "demos" / "ship_spares"
    docs = base / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    (base / "brief.txt").write_text(
        "Build a ship-spares procurement assistant for marine fleet superintendents that looks up "
        "spare parts from our catalogue CSV, checks supplier discount tiers, compliance certificates, "
        "and port freight lead times, and calculates total order landed costs and estimated arrival dates, "
        "while refusing unapproved emergency credit overrides or legal/weather forecasts outside the docs.",
        encoding="utf-8",
    )

    rows = [
        ["part_id", "part_name", "category", "engine_model", "supplier_id", "unit_price_usd", "moq", "stock_qty", "lead_time_days", "weight_kg", "criticality"],
        ["SP-MV-101", "Main Seawater Cooling Pump Impeller", "Pumps", "MAN-6S50MC", "SUP-NORDIC", "640", "2", "14", "5", "8.5", "CRITICAL"],
        ["SP-MV-104", "Bronze Butterfly Sea Chest Valve DN150", "Valves", "AUX-ALL", "SUP-ROTTERDAM", "920", "1", "6", "7", "22.0", "CRITICAL"],
        ["SP-VL-204", "High-Pressure Fuel Injection Valve", "Fuel System", "MAN-6S50MC", "SUP-NORDIC", "1180", "4", "20", "4", "4.2", "CRITICAL"],
        ["SP-VL-209", "Starting Air Non-Return Valve", "Valves", "WARTSILA-6L46", "SUP-SINGAPORE", "490", "2", "11", "6", "6.0", "STANDARD"],
        ["SP-GS-309", "Turbocharger Rotor Bearing Kit", "Turbocharger", "ABB-TPL77", "SUP-SINGAPORE", "3450", "1", "3", "9", "14.0", "CRITICAL"],
        ["SP-GS-312", "Cylinder Head Gasket Set (6-Pack)", "Seals", "MAN-6S50MC", "SUP-NORDIC", "420", "2", "28", "3", "5.5", "STANDARD"],
        ["SP-EL-402", "Viton O-Ring Seal Kit High-Temp", "Seals", "WARTSILA-6L46", "SUP-ROTTERDAM", "165", "5", "45", "2", "1.2", "STANDARD"],
        ["SP-EL-415", "Main Switchboard AVR Voltage Regulator", "Electrical", "HYUNDAI-HIMSEN", "SUP-SINGAPORE", "2250", "1", "4", "8", "3.8", "CRITICAL"],
        ["SP-PM-501", "Lube Oil Purifier Bowl Disc Stack", "Separators", "ALFA-S835", "SUP-NORDIC", "2890", "1", "2", "12", "19.5", "CRITICAL"],
        ["SP-PM-508", "Bilge Oily Water Separator Filter Element", "Filters", "AUX-ALL", "SUP-ROTTERDAM", "210", "6", "36", "3", "3.0", "STANDARD"],
        ["SP-EX-603", "Exhaust Manifold Expansion Bellows", "Exhaust", "WARTSILA-6L46", "SUP-ROTTERDAM", "1340", "1", "5", "10", "16.5", "STANDARD"],
        ["SP-COOL-702", "Plate Heat Exchanger Titanium Plate", "Cooling", "ALFA-M15", "SUP-SINGAPORE", "310", "10", "60", "5", "2.4", "STANDARD"],
    ]
    with (docs / "spares_catalogue.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    _write_pdf(
        docs / "Supplier_SLA_and_Pricing_Terms.pdf",
        "Marine Fleet Procurement — Supplier SLA & Discount Terms (PROC-DOC-01)",
        "Effective Year: 2026 | Approved Vendors: SUP-NORDIC, SUP-SINGAPORE, SUP-ROTTERDAM",
        [
            (
                "1.1 Volume Discounts & Minimum Order Quantities",
                [
                    "Orders placed with SUP-NORDIC receive a 5% unit price discount when total quantity on a single part line is 10 units or more.",
                    "Orders placed with SUP-SINGAPORE receive an 8% order discount whenever the pre-freight merchandise subtotal reaches USD 15,000 or higher.",
                    "SUP-ROTTERDAM offers fixed contract pricing (0% volume discount) but guarantees 48-hour dispatch readiness for in-stock STANDARD items.",
                    "All orders must satisfy the Minimum Order Quantity (MOQ) listed in spares_catalogue.csv.",
                ],
            ),
            (
                "1.2 Warranty, RMA & Expedited AOG Surcharge",
                [
                    "All OEM spare parts carry a warranty of 12 months from installation or 4,000 engine running hours, whichever occurs first.",
                    "Defective or wrong-spec items must be reported via an RMA ticket within 21 calendar days of port delivery.",
                    "Emergency Aircraft-On-Ground (AOG) priority dispatch adds a flat handling surcharge of USD 380 per purchase order and cuts supplier warehouse lead time by 50% (minimum 1 day).",
                ],
            ),
            (
                "1.3 Payment Terms & Credit Limits",
                [
                    "Standard payment terms are Net-30 days for purchase orders up to USD 25,000 and Net-45 days above USD 25,000.",
                    "Fleet superintendents are authorized to approve purchase orders up to USD 20,000; orders exceeding USD 20,000 require Technical Director sign-off.",
                ],
            ),
        ],
    )

    _write_pdf(
        docs / "Marine_Technical_Compliance_and_Specs.pdf",
        "Marine Engineering — Technical Compliance & Material Certification (PROC-DOC-02)",
        "Effective Year: 2026 | Classification Compliance: IACS / DNV / Lloyd's Register",
        [
            (
                "2.1 Mandatory Class Certification for Critical Parts",
                [
                    "Every spare part marked criticality=CRITICAL in spares_catalogue.csv MUST ship with an EN 10204 3.1 material certificate and class type-approval dossier.",
                    "Items marked STANDARD require only supplier Certificate of Conformity (CoC) unless ordered for dry-dock class renewal.",
                ],
            ),
            (
                "2.2 Operating Limits & Material Specifications",
                [
                    "SP-VL-204 (High-Pressure Fuel Injection Valve) is rated up to 1,200 bar injection pressure and 180 C fuel preheat temperature; nozzle lift tolerance is 0.45 mm.",
                    "SP-MV-101 (Main Seawater Cooling Pump Impeller) uses nickel-aluminum-bronze alloy (CuAl10Fe5Ni5) rated for 3,600 RPM continuous duty.",
                    "SP-EL-402 (Viton O-Ring Seal Kit) has a strict maximum shelf life of 36 months from cure date and must be stored below 25 C away from UV light.",
                ],
            ),
        ],
    )

    _write_pdf(
        docs / "Port_Freight_and_Customs_Schedule.pdf",
        "Global Marine Logistics — Port Freight Rates & Customs Lead Times (PROC-DOC-03)",
        "Effective Year: 2026 | Hub Ports: Singapore, Rotterdam, Fujairah",
        [
            (
                "3.1 Air & Sea Freight Rate Structure",
                [
                    "Air courier freight is billed at USD 6.50 per kg (total shipment weight = unit weight_kg * quantity from spares_catalogue.csv), subject to a minimum air freight charge of USD 180 per shipment.",
                    "Consolidated sea freight is billed at USD 1.90 per kg with a minimum charge of USD 95 per shipment.",
                ],
            ),
            (
                "3.2 Port Customs Handling Fees & Total Lead Time",
                [
                    "Flat port customs clearance & launch delivery fees are: Singapore USD 140, Rotterdam USD 195, and Fujairah USD 210.",
                    "Total delivery lead time equals supplier lead_time_days (from spares_catalogue.csv) PLUS transit time: +2 days for air freight or +14 days for sea freight.",
                    "Example formula: Landed Cost = (unit_price_usd * qty * (1 - discount)) + freight_cost + customs_fee.",
                ],
            ),
            (
                "3.3 Restricted & Hazardous Cargo Rules",
                [
                    "Lithium batteries, pressurized nitrogen accumulators, and solvent cleaning chemicals are strictly prohibited on standard passenger-deck air courier consignments and must move via dedicated IMDG dangerous-goods sea freight.",
                ],
            ),
        ],
    )


if __name__ == "__main__":
    build_hr_demo()
    build_ship_spares_demo()
    print("Built HR Policy & Ship-Spares demo corpora cleanly.")
