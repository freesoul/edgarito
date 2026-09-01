"""Focused audit coverage for the frozen Coca-Cola experiment fixture."""

import json
import subprocess
from pathlib import Path

from edgarito.evaluation import (
    audit_case,
    canonical_information_content_hash,
    evaluation_information_set_hash,
    load_fixture,
)
from edgarito.services.forecasting.reasoning import build_evidence_catalog
from edgarito.services.forecasting.reasoning.contracts import ForecastReasoningResponse
from edgarito.services.forecasting.reasoning.evidence import content_hash
from edgarito.services.forecasting.reasoning.reasoner import (
    CONTEXT_VERSION,
    FORECAST_REASONER_INSTRUCTIONS,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    VALIDATOR_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]


def test_ko_fixture_has_partial_company_fact_alternatives_and_exact_filings():
    fixture = load_fixture("ko")
    assert fixture.manifest.completeness == "partial"
    assert fixture.manifest.operating_evidence_available is False
    assert fixture.case.reasoning_input.segments == ()
    assert fixture.case.reasoning_input.definitions == ()

    required_concepts = {
        "revenue",
        "operating_income",
        "pretax_income",
        "income_tax_expense",
        "depreciation_and_amortization",
        "capital_expenditures",
        "accounts_receivable",
        "inventory",
        "prepaid_and_other_current_assets",
        "current_assets",
        "accounts_payable",
        "accrued_liabilities",
        "current_liabilities",
        "cash_and_equivalents",
    }
    rows_by_year = {
        year: {row.concept: row for row in fixture.manifest.financial_rows if row.fiscal_year == year}
        for year in (2022, 2023)
    }
    assert all(set(rows) == required_concepts for rows in rows_by_year.values())

    filing_by_year = {
        2022: ("0000021344-23-000011", "2023-02-21"),
        2023: ("0000021344-24-000009", "2024-02-20"),
    }
    expected_values = {
        2022: {"revenue": "43004", "operating_income": "10909", "pretax_income": "11686", "income_tax_expense": "2115", "depreciation_and_amortization": "1260", "capital_expenditures": "1484", "accounts_receivable": "3487", "inventory": "4233", "prepaid_and_other_current_assets": "3240", "current_assets": "22591", "accounts_payable": "5307", "accrued_liabilities": "5643", "current_liabilities": "19724", "cash_and_equivalents": "9519"},
        2023: {"revenue": "45754", "operating_income": "11311", "pretax_income": "12952", "income_tax_expense": "2249", "depreciation_and_amortization": "1128", "capital_expenditures": "1852", "accounts_receivable": "3410", "inventory": "4424", "prepaid_and_other_current_assets": "5235", "current_assets": "26732", "accounts_payable": "5590", "accrued_liabilities": "5631", "current_liabilities": "23571", "cash_and_equivalents": "9366"},
    }
    for year, rows in rows_by_year.items():
        accession, filed = filing_by_year[year]
        assert {concept: row.value for concept, row in rows.items()} == expected_values[year]
        assert all(row.source_id == accession and str(row.filed) == filed for row in rows.values())
        assert int(rows["capital_expenditures"].value) > 0

    assert audit_case(fixture.case).valid


def test_ko_availability_hashes_are_loader_valid_and_actuals_stay_separate():
    fixture = load_fixture("ko")
    historical = {
        (item.fiscal_year, item.metric): item
        for item in fixture.case.reasoning_input.historical_facts
    }
    for record in fixture.case.availability_manifest:
        item = historical[(2022, "revenue")] if "2022" in record.identity else historical[(2023, "revenue")]
        assert record.content_hash == canonical_information_content_hash(item)

    assert [(item.metric, str(item.value)) for item in fixture.actual_outcomes.observations] == [
        ("revenue", "47061"),
        ("ebit", "9992"),
    ]
    assert all(item.fiscal_year == 2024 for item in fixture.actual_outcomes.observations)
    assert all(item.source_id == "0000021344-25-000011" for item in fixture.actual_outcomes.observations)


def test_ko_experiment_artifact_is_truthful_and_deterministic():
    fixture = load_fixture("ko")
    artifact_path = ROOT / "tests" / "fixtures" / "evaluation" / "ko_experiment.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert artifact["status"] == "blocked"
    assert artifact["completion"] == "incomplete"
    assert artifact["successful"] is False
    assert artifact["current_commit_sha"] == commit
    assert artifact["as_of"] == fixture.case.as_of.isoformat()
    assert artifact["horizon"]["fiscal_years"] == list(fixture.case.fiscal_years)
    assert artifact["fixture_case_id"] == fixture.case.case_id
    assert artifact["cutoff"]["historical_cutoff"] == "2023-12-31"
    assert "2024-02-20" in artifact["cutoff"]["rationale"]
    assert "2024-02-15" in artifact["cutoff"]["rationale"]

    assert artifact["hashes"] == {
        "case_id": fixture.case.case_id,
        "evidence_bundle_hash": build_evidence_catalog(
            fixture.case.reasoning_input
        ).bundle_hash,
        "information_set_hash": evaluation_information_set_hash(fixture.case),
    }
    assert artifact["fixture_information_scope"]["segments"]["count"] == 0
    assert artifact["fixture_information_scope"]["driver_definitions"]["count"] == 0
    assert artifact["fixture_information_scope"]["management_guidance"]["count"] == 0
    assert artifact["fixture_information_scope"]["research_evidence"]["count"] == 0
    assert artifact["missing_requirements"] == [
        "Gross profit/gross margin evidence and R&D, SG&A, and other operating item facts are missing; they are required to support an auditable FY2024 operating expense forecast."
    ]

    runs = {item["run"]: item for item in artifact["runs"]}
    assert runs[1]["status"] == "proposal_available"
    assert runs[2]["status"] == "structured_response_failure"
    assert runs[2]["error_type"] == "OpenAIExtractionError"
    assert runs[2]["assumption_table"] == []
    assert runs[3]["status"] == "proposal_available"
    assert runs[1]["assumption_table"] and runs[3]["assumption_table"]
    assert [
        (item["metric"], item["low"], item["base"], item["high"])
        for item in runs[1]["assumption_table"]
    ] == [
        ("revenue", ["45000"], ["47000"], ["49000"]),
        (
            "depreciation_and_amortization",
            ["1050"],
            ["1150"],
            ["1300"],
        ),
        ("capital_expenditures", ["1650"], ["1850"], ["2050"]),
    ]
    assert [
        (item["metric"], item["low"], item["base"], item["high"])
        for item in runs[3]["assumption_table"]
    ] == [("revenue", ["46500"], ["47125"], ["47750"])]
    for run in (1, 3):
        for assumption in runs[run]["assumption_table"]:
            assert {
                "low",
                "base",
                "high",
                "kind",
                "confidence",
                "evidence_ids",
                "rationale",
            } <= assumption.keys()
    assert artifact["predeclared_policy"]["primary_run"] == 1
    assert artifact["stability"]["excluded_failed_runs"] == [2]
    assert artifact["stability"]["successful_runs"] == [1, 3]
    assert set(artifact["stability"]["targets"]) == {
        "forecast_metric:company:company:revenue",
        "forecast_metric:company:company:depreciation_and_amortization",
        "forecast_metric:company:company:capital_expenditures",
    }
    assert (
        artifact["stability"]["targets"]["forecast_metric:company:company:revenue"]
        ["base_statistics"]["dispersion"]
        == "125"
    )
    assert artifact["stability"]["targets"]["forecast_metric:company:company:revenue"]["values"] == {
        "run_1": {"low": "45000", "base": "47000", "high": "49000"},
        "run_3": {"low": "46500", "base": "47125", "high": "47750"},
    }
    assert all(
        "run_3" not in artifact["stability"]["targets"][target]["values"]
        for target in (
            "forecast_metric:company:company:depreciation_and_amortization",
            "forecast_metric:company:company:capital_expenditures",
        )
    )
    route_statuses = {
        (item["route"], item.get("run")): item["status"]
        for item in artifact["route_attempts"]
    }
    assert route_statuses[("ForecastReasoner proposal", 2)] == (
        "structured_response_failure"
    )
    assert route_statuses[("ForecastReasoner -> DriverBasedFcffForecastService", 1)] == (
        "blocked"
    )
    assert route_statuses[("FcffForecastService normalized baseline", None)] == (
        "available"
    )
    assert route_statuses[("hybrid baseline", None)] == "unavailable"

    assert artifact["driver_executor"]["status"] == "blocked"
    assert artifact["driver_executor"]["readiness"]["ready"] is False
    assert "no fallback forecast was used" in artifact["driver_executor"]["error"]
    assert artifact["driver_executor"]["compiler"]["plan"]["resolved"] == "driver_based"
    economics = artifact["driver_executor"]["economics"]
    assert economics["fiscal_years"] == [2024]
    assert economics["consolidated_gross_profit"] == [None]
    assert economics["reinvestment_seed"]["value"] == "-10502"
    assert artifact["normalized_baseline"]["status"] == "available"
    normalized = artifact["normalized_baseline"]["forecast_values"]["2024"]
    assert normalized["revenue"] == "48679.85573435029299600037204"
    assert normalized["ebit"] == "12191.56404294981102099069922"
    assert artifact["hybrid"]["status"] == "unavailable"
    assert artifact["hybrid"]["output"] is None
    assert artifact["reasoned_output"] is None
    assert all(
        item["code"] == "UNSUPPORTED_METHOD"
        for item in artifact["validation_readiness"]["run_1"]["rejected_assumptions"]
    )
    assert "missing evidence" not in artifact["primary_outputs"]["reason"].casefold()
    assert artifact["primary_outputs"] == {
        "status": "not_produced",
        "reason": artifact["error_comparison"]["reason"],
        "reasoned_forecast": None,
        "error_comparison": None,
        "valuation": None,
    }
    assert artifact["error_comparison"]["reasoned_errors"] is None
    assert artifact["error_comparison"]["delta_reasoned_minus_normalized"] is None
    assert [(item["metric"], item["value"]) for item in artifact["actual_fy2024"]] == [
        ("revenue", "47061"),
        ("ebit", "9992"),
    ]

    identities = artifact["identities"]
    assert identities["prompt_version"] == PROMPT_VERSION
    assert identities["prompt_hash"] == content_hash(FORECAST_REASONER_INSTRUCTIONS)
    assert identities["schema_version"] == SCHEMA_VERSION
    assert identities["schema_hash"] == content_hash(ForecastReasoningResponse.model_json_schema())
    assert identities["validator_version"] == VALIDATOR_VERSION
    assert identities["validator_hash"] == content_hash(VALIDATOR_VERSION)
    assert identities["context_version"] == CONTEXT_VERSION
    assert identities["evidence_bundle_hash"] == build_evidence_catalog(
        fixture.case.reasoning_input
    ).bundle_hash
