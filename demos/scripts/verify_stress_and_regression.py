"""Run unseen stress-tests (0% halt + unseen port/weather/JSON domain) and Regression Mode proof."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from demos.scripts.build_demo_corpora import _write_pdf
from forgeai.factory.pipeline import run_forge_pipeline
from forgeai.factory.regression import run_regression_check


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_regression_demo() -> None:
    """Create a realistic updated HR policy corpus snapshot and run ForgeAI Regression Mode."""
    hr_agent_dir = REPO_ROOT / "demos" / "hr_policy" / "output"
    reg_out = REPO_ROOT / "demos" / "hr_policy" / "output" / "regression_demo"
    if reg_out.exists():
        shutil.rmtree(reg_out)

    with tempfile.TemporaryDirectory(prefix="forgeai_hr_v2_") as td:
        docs_v2 = Path(td)
        # Copy Code of Conduct unchanged, update Leave (24 -> 20 days), remove $450 home-office stipend from Remote Policy
        shutil.copy2(
            REPO_ROOT / "demos" / "hr_policy" / "docs" / "Code_of_Conduct_and_Benefits_Policy.pdf",
            docs_v2 / "Code_of_Conduct_and_Benefits_Policy.pdf",
        )
        _write_pdf(
            docs_v2 / "HR_Leave_and_Attendance_Policy.pdf",
            "Oceanic Shore Operations — Updated Leave & Attendance Policy (HR-POL-01 v4.0)",
            "Effective Date: July 1, 2026 | Version 4.0 | Scope: Full-Time Shore Staff",
            [
                (
                    "1.1 Revised Annual Paid Leave Entitlement",
                    [
                        "Effective July 2026, full-time permanent shore staff accrue 20 working days of paid annual leave per calendar year.",
                        "Carry forward is reduced to a maximum of 3 unused annual leave days into Q1 before February 28.",
                    ],
                ),
                (
                    "1.2 Sick Leave & Medical Documentation",
                    [
                        "Shore employees receive 12 days of paid sick leave per calendar year without carryover.",
                        "A licensed physician's medical certificate is mandatory for any sick leave absence exceeding 3 consecutive working days.",
                    ],
                ),
            ],
        )
        _write_pdf(
            docs_v2 / "Remote_Work_and_Expense_Policy.pdf",
            "Oceanic Shore Operations — Revised Hybrid & Travel Policy (HR-POL-02 v3.0)",
            "Effective Date: July 1, 2026 | Version 3.0 | Scope: Shore Staff",
            [
                (
                    "2.1 Hybrid Remote Work Eligibility",
                    [
                        "Permanent shore employees who passed probation may work remotely 1 day per week with manager approval.",
                    ],
                ),
                (
                    "2.2 Travel Meal & Lodging Caps (Stipend Discontinued)",
                    [
                        "The prior home-office equipment stipend program has been discontinued for 2026.",
                        "Domestic travel meal reimbursement is capped at USD 55 per day and hotel stays at USD 220 per night.",
                    ],
                ),
            ],
        )

        diff, new_card = run_regression_check(
            agent_dir=hr_agent_dir,
            new_docs_dir=docs_v2,
            out_dir=reg_out,
            provider="ollama",
            model_name="qwen2.5:3b",
            judge_model_name="qwen2.5:1.5b",
            use_llm_judge=True,
        )
        print(
            f"[REGRESSION DEMO] modified={diff.modified_files} | "
            f"pass_rate: {diff.old_pass_rate:.3f} -> {diff.new_pass_rate:.3f} | "
            f"newly_broken={len(diff.newly_broken)} | now_ungroundable={len(diff.now_ungroundable)}"
        )


def run_unseen_stress_tests() -> None:
    """Verify unseen brief behavior: (A) 0% ungroundable halt, (B) JSON+Weather+Unit+Partial Gap."""
    out_root = REPO_ROOT / "runs" / "unseen_stress"
    if out_root.exists():
        shutil.rmtree(out_root)

    with tempfile.TemporaryDirectory(prefix="forgeai_unseen_") as td:
        docs = Path(td)
        (docs / "berths.json").write_text(
            json.dumps(
                [
                    {"port": "Singapore", "berth": "B-04", "max_draft_m": 14.5, "tug_fee_usd": 1200},
                    {"port": "Rotterdam", "berth": "R-12", "max_draft_m": 16.0, "tug_fee_usd": 1450},
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        (docs / "port_dispatch_rules.md").write_text(
            "# Port Dispatch Rules\n"
            "Vessels must give 24 hours advance berth notice. Operations suspend if sustained wind exceeds 28 knots.\n"
            "Convert vessel draft from meters to feet when requested by US-flagged masters.\n",
            encoding="utf-8",
        )

        # Trap A: 100% ungroundable brief
        res_a = run_forge_pipeline(
            brief="Provide spacecraft fusion reactor diagnostics and lunar orbit radiation shielding checklists.",
            docs_dir=docs,
            out_dir=out_root / "trap_ungroundable",
            provider="ollama",
            model_name="qwen2.5:3b",
            judge_model_name="qwen2.5:1.5b",
            unattended=True,
            max_iterations=0,
            run_tournament=False,
        )
        assert res_a.halted_ungroundable is True, "Expected 0% grounded brief to halt safely!"
        print("[UNSEEN TRAP A] Halted safely on 0% ungroundable brief:", res_a.requirements.groundability.overall_status)

        # Trap B: Unseen Port Dispatch + JSON + Weather + Unit Conversion + Unsupported prediction gap
        res_b = run_forge_pipeline(
            brief=(
                "Build a port dispatch assistant that looks up berth JSON records, checks live port weather "
                "forecasts via Open-Meteo, converts wind knots to kmh or meters to feet, calculates tug fees, "
                "and forecasts 10-year ocean temperature anomalies."
            ),
            docs_dir=docs,
            out_dir=out_root / "unseen_port_dispatch",
            provider="ollama",
            model_name="qwen2.5:3b",
            judge_model_name="qwen2.5:1.5b",
            unattended=True,
            max_iterations=1,
            eval_target_cases=20,
            run_tournament=False,
            max_seconds_budget=900.0,
        )
        assert not res_b.halted_ungroundable
        tools = [t.name for t in res_b.agent.tools] if res_b.agent else []
        print(
            f"[UNSEEN TRAP B] agent={res_b.agent.name if res_b.agent else None} | "
            f"tools={tools} | pass_rate={res_b.report.final_pass_rate if res_b.report else None} | "
            f"unsupported={res_b.requirements.groundability.unsupported_requests}"
        )


if __name__ == "__main__":
    run_regression_demo()
    run_unseen_stress_tests()
