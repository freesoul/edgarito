"""Focused audit coverage for the captured Visa evaluation experiment."""

import json
from pathlib import Path

from edgarito.evaluation import (
    audit_actual_outcomes,
    audit_case,
    evaluation_information_set_hash,
    load_fixture,
)
from edgarito.services.forecasting.reasoning import build_evidence_catalog
from edgarito.services.forecasting.reasoning.contracts import ForecastReasoningResponse
from edgarito.services.forecasting.reasoning.evidence import (
    content_hash,
    manual_inputs_hash,
    research_hash,
)
from edgarito.services.forecasting.reasoning.reasoner import (
    CONTEXT_VERSION,
    FORECAST_REASONER_INSTRUCTIONS,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    VALIDATOR_VERSION,
    build_reasoning_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "tests" / "fixtures" / "evaluation" / "v_experiment.json"


def _artifact():
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_v_experiment_freezes_current_fixture_hashes_and_cache_bypass_identities():
    fixture = load_fixture("v")
    artifact = _artifact()
    reasoning = fixture.case.reasoning_input

    assert artifact["current_commit_sha"] == "342b8f7f003b1259448b9dcf6b145f4a521bd7df"
    assert artifact["fixture_case_id"] == fixture.case.case_id
    assert artifact["as_of"] == fixture.case.as_of.isoformat()
    assert artifact["horizon"]["fiscal_years"] == list(fixture.case.fiscal_years)
    assert artifact["hashes"] == {
        "case_id": fixture.case.case_id,
        "evidence_bundle_hash": build_evidence_catalog(reasoning).bundle_hash,
        "information_set_hash": evaluation_information_set_hash(fixture.case),
        "normalized_history_hash": content_hash(reasoning.historical_facts),
        "operating_evidence_hash": content_hash(reasoning.observations),
        "operating_history_hash": content_hash(reasoning.historical_facts),
        "manual_inputs_hash": manual_inputs_hash(reasoning),
        "research_hash": research_hash(reasoning),
    }

    assert artifact["hashes"]["normalized_history_hash"] == (
        "f020ada1d9b385e0e6b4c01d9d4d83a55e6dc8fd9a562beb3ed903f210b87b57"
    )
    assert artifact["hashes"]["operating_evidence_hash"] == (
        "264e2f01b29b55059c0fe0103a5da468904540500087dc077444e0666ff6b5ce"
    )
    assert artifact["hashes"]["operating_history_hash"] == artifact["hashes"][
        "normalized_history_hash"
    ]

    identities = artifact["identities"]
    assert identities["model"] == "gpt-5.6-luna"
    assert identities["reasoning_effort"] == "medium"
    assert identities["context_version"] == CONTEXT_VERSION
    assert identities["prompt_version"] == PROMPT_VERSION
    assert identities["prompt_hash"] == content_hash(build_reasoning_prompt())
    assert identities["prompt_hash"] == content_hash(FORECAST_REASONER_INSTRUCTIONS)
    assert identities["schema_version"] == SCHEMA_VERSION
    assert identities["schema_hash"] == content_hash(ForecastReasoningResponse.model_json_schema())
    assert identities["validator_version"] == VALIDATOR_VERSION
    assert identities["validator_hash"] == content_hash(VALIDATOR_VERSION)

    bypasses = artifact["cache_bypass_identities"]
    assert len(bypasses) == 5
    assert all(not item["cache_hit"] and item["force_refresh"] for item in bypasses)
    assert len({item["cache_key"] for item in bypasses}) == 1
    assert [item["run"] for item in bypasses] == [1, 2, 3, 4, 5]


def test_v_experiment_preserves_all_structured_runs_and_primary_policy():
    artifact = _artifact()

    captured = artifact["captured_runs"]
    assert len(captured) == 5
    assert artifact["raw_structured_proposals"] == [
        item["proposal"]["response"] for item in captured
    ]
    assert artifact["validation_results"] == [
        item["validation"] for item in captured
    ]
    assert artifact["compilation_outputs"] == [
        item["compilation"] for item in captured
    ]
    assert all(item["structured_success"] for item in captured)
    assert all(item["compile_success"] for item in captured)
    assert artifact["structured_success"] == {
        "successful_runs": [1, 2, 3, 4, 5],
        "count": 5,
        "denominator": 5,
    }
    assert artifact["fully_clean_deterministic_validation"] == {
        "successful_runs": [1, 3],
        "count": 2,
        "denominator": 5,
    }
    assert artifact["compiler_execution"] == {
        "successful_runs": [1, 2, 3, 4, 5],
        "count": 5,
        "denominator": 5,
    }
    assert artifact["predeclared_policy"]["primary_run"] == 1
    assert artifact["primary"]["run"] == 1
    assert artifact["primary"]["assumption"] == {
        "assumption_id": "A-2024-company-revenue",
        "assumption_type": "model_assumption",
        "base": ["36000"],
        "basis": "absolute",
        "confidence": "low",
        "driver_id": None,
        "evidence_based": False,
        "evidence_ids": ["HIST-06a5898e4fceedc1", "HIST-602668dad7cdef13"],
        "fiscal_years": [2024],
        "high": ["37000"],
        "low": ["35000"],
        "metric": "revenue",
        "model_assumption": True,
        "rationale": "Model assumption for FY2024 consolidated revenue, anchored to the accepted FY2022 and FY2023 historical revenue observations. No FY2024 management guidance or consensus evidence was supplied, so the range is intentionally conservative and remains subject to revision when near-term guidance becomes available.",
        "scope": "company",
        "scope_id": "company",
        "target_type": "forecast_metric",
        "unit": "USD millions",
    }
    assert artifact["primary"]["operating_driver_assumptions"] == []
    assert artifact["primary"]["reasoned_canonical_forecast"] is None
    assert artifact["primary"]["reasoned_validation_output"] is None
    assert artifact["primary"]["no_silent_fallback"] is True
    assert artifact["primary"]["unresolved_items"] == captured[0]["proposal"]["response"][
        "unresolved_items"
    ]
    assert "unresolved_management_guidance" not in artifact["primary"]


def test_v_experiment_preserves_route_values_errors_and_unmodeled_registry():
    fixture = load_fixture("v")
    artifact = _artifact()
    route = artifact["routes"]
    normalized = artifact["normalized_baseline"]["forecast_values"]["2024"]

    assert artifact["pre_reasoner_quality_gate"] == {
        "status": "rejected",
        "rejected_definitions": 0,
        "definitions": 0,
        "error_type": "OperatingForecastQualityError",
        "error": "Operating forecast quality rejected: definitions=0 (requires non-empty definitions)",
    }
    assert artifact["driver_executor"]["status"] == "blocked"
    assert artifact["driver_executor"]["error_type"] == "DriverBasedForecastIncompleteError"
    assert artifact["driver_executor"]["error"] == route["driver"]["error"]
    assert artifact["driver_executor"]["readiness"] == route["driver"]["readiness"]
    assert artifact["driver_executor"]["fallback_used"] is False

    assert normalized == {
        "revenue": "36377.29133401569430228590924",
        "ebit": "23372.21054020534385719728438",
        "tax_rate": "17.71047863785442075831732940",
        "nopat": "19232.88018528791711890282225",
        "depreciation_and_amortization": "1059.580805844208679533745613",
        "capex": "1191.83724998285949289009437",
        "operating_working_capital": "-18785.83466245016011969137542",
        "delta_owc": "-562.83466245016011969137542",
        "fcff": "19663.45840359942642523784891",
    }
    assert route["normalized"]["status"] == "available"
    assert artifact["reasoned_output"] is None
    assert artifact["hybrid"]["status"] == "unavailable"
    assert artifact["primary_outputs"]["reasoned_forecast"] is None
    assert artifact["primary_outputs"]["error_comparison"] is None
    assert artifact["error_comparison"]["reasoned_errors"] is None
    assert artifact["error_comparison"]["delta_reasoned_minus_normalized"] is None
    assert artifact["error_comparison"]["claim"] is None
    assert {
        metric: {
            "absolute": values["absolute"],
            "percentage": values["percentage"],
        }
        for metric, values in artifact["error_comparison"]["normalized_errors"].items()
    } == {
        "revenue": {
            "absolute": "451.29133401569430228590924",
            "percentage": "1.256169164437160558609111062%",
        },
        "ebit": {
            "absolute": "222.78945979465614280271562",
            "percentage": "0.9442231820074428599394601399%",
        },
        "depreciation_and_amortization": {
            "absolute": "25.580805844208679533745613",
            "percentage": "2.473965748956351985855475145%",
        },
        "capex": {
            "absolute": "65.16275001714050710990563",
            "percentage": "5.183989659279276619722007160%",
        },
        "tax_rate": {
            "absolute": "0.26190864287198222344527722",
            "percentage": "1.501032135855817602664090581%",
        },
    }

    reasoning = fixture.case.reasoning_input
    assert len(reasoning.segments) == 0
    assert len(reasoning.definitions) == 0
    assert len(reasoning.observations) == 8
    assert all(item.driver_id != "take_rate" for item in reasoning.observations)
    assert "transactions_take_rate" not in str(artifact["discovered_model"])


def test_v_experiment_keeps_actuals_separate_and_diagnostics_explicit():
    fixture = load_fixture("v")
    artifact = _artifact()
    actual = {item["metric"]: item for item in artifact["actual_fy2024"]}
    fixture_actual = {
        item.metric: item.model_dump(mode="json")
        for item in fixture.actual_outcomes.observations
    }
    assert actual == fixture_actual
    assert set(actual) == {
        "revenue",
        "ebit",
        "pretax_income",
        "income_tax_expense",
        "depreciation_and_amortization",
        "capex",
        "tax_rate",
    }
    assert artifact["actuals_policy"] == {
        "actual_financials_are_separate": True,
        "safe_derived_actuals": ["tax_rate"],
        "owc_actual": None,
        "fcff_actual": None,
        "primary_driver_assumptions_scored": False,
        "driver_assumption_scoring_reason": "Primary run 1 produced no operating driver assumptions.",
    }
    assert not any("owc" in metric.casefold() for metric in actual)
    assert not any("fcff" in metric.casefold() for metric in actual)
    assert artifact["actual_audit"] == audit_actual_outcomes(
        fixture.actual_outcomes, as_of=fixture.case.as_of
    ).model_dump(mode="json")
    assert artifact["information_set_audit"] == audit_case(fixture.case).model_dump(
        mode="json"
    )

    assert artifact["model_coverage"] == {
        "driver_coverage": None,
        "modeled_revenue_share": None,
        "accepted_definitions": 0,
        "material_revenue_components": {
            "service_revenue": "unmodeled",
            "data_processing": "unmodeled",
            "international_transaction": "unmodeled",
            "other_revenue": "unmodeled",
            "client_incentives": "unmodeled",
        },
    }

    assert artifact["stability"]["successful_runs"] == [1, 2, 3, 4, 5]
    assert artifact["accepted_executable_stability"]["successful_runs"] == [1, 2, 3, 4, 5]
    assert set(artifact["stability"]["targets"]) == {
        "forecast_metric:company:company:revenue",
        "forecast_metric:company:company:depreciation_and_amortization",
        "forecast_metric:company:company:capital_expenditures",
        "forecast_metric:company:company:tax_rate",
    }
    assert set(artifact["accepted_executable_stability"]["targets"]) == {
        "forecast_metric:company:company:revenue",
        "forecast_metric:company:company:depreciation_and_amortization",
        "forecast_metric:company:company:capital_expenditures",
    }
    for report in (
        artifact["stability"],
        artifact["accepted_executable_stability"],
    ):
        assert all("mean" not in item["statistics"] for item in report["targets"].values())
        assert all("average" not in item["statistics"] for item in report["targets"].values())

    diagnostics = artifact["diagnostic"]
    assert diagnostics["primary_bottleneck"] == "business-model discovery"
    assert diagnostics["classification"] == "Visa economics unmodeled by current registry"
    assert diagnostics["unsafe_shortcut"] == (
        "transactions_take_rate would invent a blended take rate"
    )
    assert diagnostics["next_action"] == (
        "Design/evaluate a provider-neutral multi-component Visa-like business-model discovery abstraction, not production company-specific logic, before retrying Visa."
    )
    assert "OWC" in diagnostics["owc_asymmetry_warning"]
    assert {
        letter: verdict["category"]
        for letter, verdict in artifact["diagnostic_verdicts"].items()
    } == {
        "A": "BUSINESS-MODEL DISCOVERY",
        "B": "REASONING QUALITY",
        "C": "DRIVER ACCURACY",
        "D": "FINANCIAL ACCURACY",
        "E": "CALIBRATION",
        "F": "STABILITY",
        "G": "MAIN BOTTLENECK",
        "H": "NEXT ENGINEERING ACTION",
    }
    assert artifact["diagnostic_verdicts"]["A"]["verdict"] == "no driver model"
    assert artifact["diagnostic_verdicts"]["B"]["verdict"] == (
        "primary one low-confidence revenue model assumption accepted but no driver assumptions"
    )
    assert artifact["diagnostic_verdicts"]["C"]["verdict"] == "unavailable"
    assert artifact["diagnostic_verdicts"]["D"]["verdict"] == (
        "financial reasoned accuracy unavailable while normalized errors available"
    )
    assert artifact["diagnostic_verdicts"]["E"]["verdict"] == (
        "primary revenue interval 35000-37000 contains actual 35926 but only proposal-level, not executed"
    )
    assert artifact["diagnostic_verdicts"]["F"]["verdict"] == (
        "5/5 structured, 2/5 fully clean validation, accepted revenue base dispersion 1000 across runs1/3 and all-proposal dispersion1500"
    )
    assert artifact["diagnostic_verdicts"]["G"]["verdict"] == (
        "business-model discovery"
    )
    assert artifact["diagnostic_verdicts"]["H"]["verdict"] == (
        "provider-neutral multi-component business-model discovery design/evaluation"
    )
