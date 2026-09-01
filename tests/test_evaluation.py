"""Offline contract and scoring coverage for isolated evaluation."""

import asyncio
import datetime
import os
import subprocess
import sys
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.evaluation import (
    HUMAN_SECTIONS,
    ActualAssumptionOutcome,
    ActualFinancialObservation,
    ActualOutcomeData,
    ForecastBacktestCase,
    ForecastBacktestRunner,
    InformationAvailabilityRecord,
    LeakageError,
    StabilityEvaluator,
    aggregate_calibration,
    audit_case,
    build_evaluation_evidence_catalog,
    build_evaluation_reasoning_content,
    canonical_information_content_hash,
    canonical_information_identity,
    compare_forecasts,
    complexity_diagnostics,
    cutoff_financials,
    deterministic_machine_report,
    human_report,
    load_fixture,
    reconstruct_actual_outcomes,
    score_assumptions,
    score_financials,
)
from edgarito.schemas.forecasting import FcffForecastParameters
from edgarito.schemas.guidance.management import ManagementGuidance
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.operating import (
    EvidenceReference,
    OperatingDriverObservation,
    OperatingEvidenceExtractionResult,
    OperatingSegment,
)
from edgarito.services.forecasting._fcff.service import FcffForecastService
from edgarito.services.forecasting.reasoning import (
    ForecastReasoningInput,
    ReasonedForecastAssumption,
)
from edgarito.services.forecasting.reasoning.reasoner import (
    FORECAST_REASONER_INSTRUCTIONS,
)
from edgarito.services.research.contracts import (
    EvidenceProvenance,
    MarketGrowthEvidence,
)

D = Decimal
DEFAULT_LOW = (D("90"), D("90"))
DEFAULT_BASE = (D("100"), D("100"))
DEFAULT_HIGH = (D("110"), D("110"))


def _fixture():
    return load_fixture("ko")


def _actuals(*, values=None, assumptions=()):
    values = values or {}
    observations = tuple(
        ActualFinancialObservation(
            fiscal_year=year,
            period_end=datetime.date(year, 12, 31),
            metric=metric,
            value=value,
            unit="USD",
            source="test actual subset",
            source_date=datetime.date(year + 1, 2, 1),
            provenance="typed test fixture",
        )
        for year, year_values in values.items()
        for metric, value in year_values.items()
    )
    return ActualOutcomeData(observations=observations, assumption_outcomes=assumptions)


def _forecast(values, *, unit="USD"):
    rows = [
        SimpleNamespace(fiscal_year=year, unit=unit, **year_values)
        for year, year_values in values.items()
    ]
    return SimpleNamespace(observations=rows)


def _assumption(*, low=DEFAULT_LOW, base=DEFAULT_BASE, high=DEFAULT_HIGH):
    return ReasonedForecastAssumption(
        assumption_id="revenue-assumption",
        scope="company",
        target_type="forecast_metric",
        metric="revenue",
        unit="USD",
        basis="absolute",
        fiscal_years=(2025, 2026),
        low=low,
        base=base,
        high=high,
        rationale="supported test path",
        confidence="high",
        evidence_based=True,
        evidence_ids=("HIST-1",),
    )


def _financial_values(first, second=None):
    second = second or first
    metrics = {
        "revenue": (100, 120),
        "ebit": (20, 25),
        "tax_rate": (25, 25),
        "nopat": (15, 18.75),
        "depreciation_and_amortization": (10, 11),
        "capex": (8, 9),
        "operating_working_capital": (20, 22),
        "delta_nwc": (2, 2),
        "fcff": (15, 18.75),
    }
    return _forecast(
        {
            2025: {metric: D(str(values[0])) for metric, values in metrics.items()},
            2026: {metric: D(str(values[1])) for metric, values in metrics.items()},
        }
    )


def _reconstruction_financials(current_end=datetime.date(2024, 9, 28)):
    values = {
        "revenue": (100, 120),
        "operating_income": (20, 24),
        "pretax_income": (20, 24),
        "income_tax_expense": (5, 6),
        "depreciation_and_amortization": (10, 12),
        "capital_expenditures": (-8, -9),
        "accounts_receivable": (20, 24),
        "inventory": (10, 11),
        "prepaid_and_other_current_assets": (5, 6),
        "accounts_payable": (12, 13),
        "accrued_liabilities": (4, 5),
        "deferred_revenue_current": (2, 3),
    }
    observations = []
    for year, period_end, index in (
        (2023, datetime.date(2023, 9, 30), 0),
        (2024, current_end, 1),
    ):
        for concept_name, pair in values.items():
            concept = FinancialConcept(concept_name)
            observations.append(
                FinancialObservation(
                    concept=concept,
                    statement=concept.statement,
                    value=pair[index],
                    unit="USD",
                    granularity=Granularity.ANNUAL,
                    fiscal_year=year,
                    fiscal_period=FiscalPeriod.FY,
                    period_end=period_end,
                    provider="test",
                    taxonomy="test",
                    source_concept=(
                        "PaymentsToAcquirePropertyPlantAndEquipment"
                        if concept_name == "capital_expenditures"
                        else concept_name
                    ),
                    filed=datetime.date(year + 1, 1, 10),
                )
            )
    return NormalizedCompanyFinancials(
        provider="test",
        company_id="X",
        company_name="X",
        ticker="X",
        observations=observations,
    )


def test_future_facts_reject_before_fake_reasoner():
    fixture = _fixture()
    future = FinancialObservation(
        concept=FinancialConcept.REVENUE,
        statement=FinancialConcept.REVENUE.statement,
        value=1,
        unit="USD",
        granularity=Granularity.ANNUAL,
        fiscal_year=2025,
        fiscal_period=FiscalPeriod.FY,
        period_end=datetime.date(2025, 12, 31),
        provider="fixture",
        taxonomy="fixture",
        source_concept="Revenue",
        filed=datetime.date(2026, 2, 1),
    )
    case = fixture.case.model_copy(
        update={
            "point_in_time_financials": fixture.case.point_in_time_financials.model_copy(
                update={"observations": [*fixture.case.point_in_time_financials.observations, future]}
            )
        }
    )
    calls = []

    class Fake:
        async def forecast(self, *_args, **_kwargs):
            calls.append(True)
            return _forecast({2025: {"revenue": D(1)}})

    with pytest.raises(LeakageError):
        asyncio.run(ForecastBacktestRunner(reasoned_service=Fake()).run(case, fixture.actual_outcomes))
    assert calls == []


def test_undated_forward_evidence_rejects_closed():
    fixture = _fixture()
    forward = OperatingDriverObservation(
        segment_id="company",
        driver_id="growth",
        fiscal_year=2025,
        value=1,
        unit="percent",
        origin="forward_evidence",
        confidence="medium",
    )
    input_value = fixture.case.reasoning_input.model_copy(
        update={"manual_forward_driver_observations": (forward,)}
    )
    case = fixture.case.model_copy(update={"reasoning_input": input_value})
    audit = audit_case(case)
    assert not audit.valid
    assert any("no availability date" in item for item in audit.issues)


def test_operating_evidence_mapping_contract_is_recursively_audited():
    fixture = _fixture()
    evidence = {
        "segments": [
            {
                "segment_id": "consumer",
                "name": "Consumer",
                "currency": "USD",
                "evidence": {
                    "provider": "sec",
                    "accession": "consumer-2023-10k",
                    "filing_date": "2024-02-15",
                },
            }
        ],
        "definitions": [
            {
                "driver_id": "consumer-revenue",
                "archetype": "generic_segment_growth",
                "segment_id": "consumer",
                "output_metric": "revenue",
                "input_metrics": ["growth"],
                "units": {"growth": "percent"},
                "formula_id": "generic_segment_growth",
                "required_inputs": ["growth"],
                "evidence": {
                    "provider": "sec",
                    "accession": "consumer-2023-10k",
                    "filing_date": "2024-02-15",
                },
            }
        ],
        "observations": [
            {
                "segment_id": "consumer",
                "driver_id": "growth",
                "fiscal_year": 2024,
                "value": 5,
                "unit": "percent",
                "origin": "forward_evidence",
                "confidence": "medium",
                "evidence": {
                    "provider": "sec",
                    "accession": "consumer-2023-10k",
                    "filing_date": "2024-02-15",
                },
            }
        ],
    }
    case = fixture.case.model_copy(update={"operating_evidence": evidence})
    assert audit_case(case).valid

    future_evidence = {
        **evidence,
        "segments": [
            {
                **evidence["segments"][0],
                "evidence": {
                    **evidence["segments"][0]["evidence"],
                    "filing_date": "2025-02-15",
                },
            }
        ],
    }
    future_audit = audit_case(
        fixture.case.model_copy(update={"operating_evidence": future_evidence})
    )
    assert not future_audit.valid
    assert any("after as_of" in issue for issue in future_audit.issues)


def test_typed_operating_evidence_result_is_recursively_audited():
    fixture = _fixture()
    reference = EvidenceReference(
        provider="sec",
        accession="consumer-2023-10k",
        filing_date=datetime.date(2024, 2, 15),
    )
    result = OperatingEvidenceExtractionResult(
        segments=(
            OperatingSegment(
                segment_id="consumer",
                name="Consumer",
                currency="USD",
                evidence=reference,
            ),
        ),
    )
    assert audit_case(fixture.case.model_copy(update={"operating_evidence": result})).valid

    future_reference = reference.model_copy(update={"filing_date": datetime.date(2025, 2, 15)})
    future_result = result.model_copy(
        update={
            "segments": (
                result.segments[0].model_copy(update={"evidence": future_reference}),
            )
        }
    )
    future_audit = audit_case(
        fixture.case.model_copy(update={"operating_evidence": future_result})
    )
    assert not future_audit.valid
    assert any("after as_of" in issue for issue in future_audit.issues)


def test_known_research_and_guidance_mapping_contracts_are_accepted():
    fixture = _fixture()
    research = MarketGrowthEvidence(
        market="test market",
        source_date=datetime.date(2024, 2, 15),
        source_type="analyst_estimate",
        low=1,
        base=2,
        high=3,
        provenance=EvidenceProvenance(source="test source"),
        unit="percent",
    )
    guidance = ManagementGuidance(
        metric="revenue",
        fiscal_year=2024,
        period_type="fiscal_year",
        point=1,
        value_kind="monetary",
        unit="millions",
        filing_date=datetime.date(2024, 2, 15),
        accession_number="test-2023-10k",
        filing_form="10-K",
        source_document="test filing",
        source_document_type="sec",
        supporting_text="test guidance",
        evidence_verified=True,
        extraction_model="test",
    )
    for field, value in (
        ("research_evidence", (research.model_dump(mode="python"),)),
        ("management_guidance", (guidance.model_dump(mode="python"),)),
    ):
        audit = audit_case(
            fixture.case.model_copy(update={"evidence_snapshot": {field: value}})
        )
        assert audit.valid


def test_actual_data_is_not_a_reasoning_input():
    actuals = _actuals(values={2025: {"revenue": D(1)}})
    with pytest.raises(ValidationError):
        ForecastReasoningInput.model_validate(actuals.model_dump())
    fixture = _fixture()
    with pytest.raises(ValueError):
        ForecastBacktestCase(
            ticker=fixture.case.ticker,
            company=fixture.case.company,
            as_of=fixture.case.as_of,
            fiscal_years=fixture.case.fiscal_years,
            point_in_time_financials=fixture.case.point_in_time_financials,
            reasoning_input=actuals,
        )


def test_actuals_absent_from_prompt_catalog_and_information_hash():
    fixture = _fixture()
    content = build_evaluation_reasoning_content(fixture.case)
    catalog = build_evaluation_evidence_catalog(fixture.case)
    assert "subsequent actual" not in content
    assert all("actual" not in item.category.casefold() for item in catalog.items)
    assert fixture.case.case_id == fixture.case.case_id


def test_assumption_base_interval_zero_and_sign_scoring():
    assumption = _assumption(low=(D(0), D(0)), base=(D(10), D(10)), high=(D(20), D(20)))
    actuals = ActualOutcomeData(
        assumption_outcomes=(
                ActualAssumptionOutcome(
                    target_type="forecast_metric", metric="revenue", fiscal_year=2025, actual=15, unit="USD"
                ),
                ActualAssumptionOutcome(
                    target_type="forecast_metric", metric="revenue", fiscal_year=2026, actual=0, unit="USD"
            ),
        )
    )
    report = score_assumptions(SimpleNamespace(accepted_assumptions=(assumption,)), actuals)
    assert report.scores[0].interval_hit is True
    assert report.scores[0].normalized_interval_position == D("0.75")
    assert report.scores[1].percentage_error is None
    financial = score_financials(
        _forecast({2025: {"revenue": D(0), "ebit": D(-1)}, 2026: {"revenue": D(2), "ebit": D(1)}}),
        _actuals(values={2025: {"revenue": D(0), "ebit": D(1)}, 2026: {"revenue": D(1), "ebit": D(2)}}),
    )
    zero = next(item for item in financial.scores if item.metric == "revenue" and item.fiscal_year == 2025)
    signed = next(item for item in financial.scores if item.metric == "ebit" and item.fiscal_year == 2025)
    assert zero.percentage_error is None
    assert signed.sign_error is True and signed.percentage_error is None


def test_financial_direction_and_multi_year_independence():
    actuals = _actuals(values={2025: {"revenue": D(100)}, 2026: {"revenue": D(90)}, 2027: {"revenue": D(110)}})
    forecast = _forecast({2025: {"revenue": D(100)}, 2026: {"revenue": D(110)}, 2027: {"revenue": D(100)}})
    report = score_financials(forecast, actuals)
    scores = [item for item in report.scores if item.metric == "revenue"]
    assert [item.absolute_error for item in scores] == [D(0), D(20), D(10)]
    assert scores[1].yoy_direction_error is True
    assert scores[2].yoy_direction_error is True


def test_normalized_fake_comparison_and_method_scores():
    reasoned = _forecast({2025: {"revenue": D(100)}})
    normalized = _forecast({2025: {"revenue": D(90)}})
    actuals = _actuals(values={2025: {"revenue": D(95)}})
    comparison = compare_forecasts(reasoned, normalized, actuals, route="normalized", method="normalized")
    assert comparison.route.available
    assert comparison.metric_deltas["revenue"]["status"] == "tied"
    assert comparison.financial_scores.per_method["normalized"]


def test_optional_hybrid_is_explicitly_unavailable():
    comparison = compare_forecasts(_forecast({2025: {"revenue": D(1)}}), None, _actuals(values={2025: {"revenue": D(1)}}), route="hybrid")
    assert comparison.route.available is False
    assert comparison.unavailable_reason


def test_reasoned_comparison_reports_status_and_delta():
    comparison = compare_forecasts(
        _forecast({2025: {"revenue": D(100)}}),
        _forecast({2025: {"revenue": D(105)}}),
        _actuals(values={2025: {"revenue": D(110)}}),
        route="normalized",
    )
    assert comparison.metric_deltas["revenue"]["status"] == "worsened"
    assert comparison.metric_deltas["revenue"]["delta_reasoned_minus_baseline"] > 0


def test_calibration_counts_and_small_sample_warning():
    assumption = _assumption()
    actuals = ActualOutcomeData(
        assumption_outcomes=tuple(
            ActualAssumptionOutcome(target_type="forecast_metric", metric="revenue", fiscal_year=year, actual=value, unit="USD")
            for year, value in ((2025, 100), (2026, 200))
        )
    )
    summary = aggregate_calibration(score_assumptions(SimpleNamespace(accepted_assumptions=(assumption,)), actuals))
    assert summary.sample_size == 2 and summary.hit_count == 1
    assert "Small sample" in summary.strata[0].warning
    assert "no statistical significance" in summary.warning.casefold()


def test_complexity_counts_are_descriptive():
    fixture = _fixture()
    assumption = _assumption()
    result = SimpleNamespace(
        reasoning=SimpleNamespace(
            accepted_assumptions=(assumption,),
            rejected_assumptions=("rejected",),
            unresolved_items=("missing",),
        )
    )
    diagnostics = complexity_diagnostics(fixture.case.reasoning_input, result)
    assert diagnostics.segments == 0
    assert diagnostics.financial_assumptions == 1
    assert diagnostics.evidence_based == 1
    assert diagnostics.unresolved == 1 and diagnostics.rejected == 1
    assert "financial_assumptions=1" in diagnostics.over_modeling_indicators


def test_stability_force_refresh_and_variance():
    fixture = _fixture()

    class FakeReasoner:
        def __init__(self):
            self.calls = 0

        async def reason(self, _input, *, force_refresh=False):
            assert force_refresh is True
            self.calls += 1
            value = D(100 + self.calls)
            return SimpleNamespace(
                proposal_identity=f"proposal-{self.calls}",
                assumptions=(_assumption(low=(D(90), D(90)), base=(value, value), high=(D(110), D(110))),),
            )

    reasoner = FakeReasoner()
    report = asyncio.run(StabilityEvaluator(reasoner).evaluate(fixture.case, runs=3))
    assert reasoner.calls == 3
    assert report.run_count == 3
    assert len(report.observations) == 6
    assert report.variance["forecast_metric:company:company:revenue:2025"] > 0


def test_case_and_scores_are_immutable():
    fixture = _fixture()
    with pytest.raises(ValidationError):
        fixture.case.ticker = "OTHER"
    forecast = _financial_values(1)
    original = tuple(forecast.observations[0].__dict__.items())
    score_financials(forecast, _actuals(values={2025: {"revenue": D(1)}, 2026: {"revenue": D(1)}}))
    assert tuple(forecast.observations[0].__dict__.items()) == original


def test_frozen_manifests_load_deterministically():
    first = load_fixture("ko")
    second = load_fixture("ko.json")
    assert first.case.case_id == second.case.case_id
    assert {load_fixture(name).case.ticker for name in ("ko", "v", "tsla")} == {"KO", "V", "TSLA"}
    assert load_fixture("tsla").case.reasoning_input.segments[1].name == "Energy Generation and Storage"


def test_actual_reconstruction_identity_sign_and_noncalendar_period():
    concepts = {
        "revenue": (100, 120),
        "operating_income": (20, 24),
        "pretax_income": (20, 24),
        "income_tax_expense": (5, 6),
        "depreciation_and_amortization": (10, 12),
        "capital_expenditures": (-8, -9),
        "accounts_receivable": (20, 24),
        "inventory": (10, 11),
        "prepaid_and_other_current_assets": (5, 6),
        "accounts_payable": (12, 13),
        "accrued_liabilities": (4, 5),
        "deferred_revenue_current": (2, 3),
    }
    observations = []
    for year, end in ((2023, datetime.date(2023, 9, 30)), (2024, datetime.date(2024, 9, 28))):
        index = 0 if year == 2023 else 1
        for concept_name, values in concepts.items():
            concept = FinancialConcept(concept_name)
            observations.append(
                FinancialObservation(
                    concept=concept,
                    statement=concept.statement,
                    value=values[index],
                    unit="USD",
                    granularity=Granularity.ANNUAL,
                    fiscal_year=year,
                    fiscal_period=FiscalPeriod.FY,
                    period_end=end,
                    provider="test",
                    taxonomy="test",
                    source_concept=(
                        "PaymentsToAcquirePropertyPlantAndEquipment"
                        if concept_name == "capital_expenditures"
                        else concept_name
                    ),
                    filed=datetime.date(year + 1, 1, 10),
                )
            )
    financials = NormalizedCompanyFinancials(provider="test", company_id="X", company_name="X", ticker="X", observations=observations)
    actuals = reconstruct_actual_outcomes(financials, fiscal_years=(2024,))
    by_metric = {item.metric: item for item in actuals.observations}
    assert by_metric["capex"].value == D(9)
    assert by_metric["operating_working_capital"].value == D(20)
    assert by_metric["delta_nwc"].value == D(3)
    assert by_metric["fcff"].value == by_metric["nopat"].value + D(12) - D(9) - D(3)
    assert by_metric["revenue"].period_end == datetime.date(2024, 9, 28)
    assert by_metric["capex"].value_kind == "reported"
    assert by_metric["capex"].sign_normalization == "normalized_from_explicit_negative_cash_flow_convention"
    assert by_metric["fcff"].value_kind == "derived"
    assert by_metric["fcff"].reconstruction_provenance


def test_derived_actual_requires_reconstruction_provenance():
    with pytest.raises(ValidationError, match="reconstruction provenance"):
        ActualFinancialObservation(
            fiscal_year=2024,
            period_end=datetime.date(2024, 12, 31),
            metric="fcff",
            value=1,
            unit="USD millions",
            value_kind="derived",
        )


def test_machine_and_human_reports_are_deterministic_fixed_sections():
    fixture = _fixture()

    class Fake:
        async def forecast(self, *_args, **_kwargs):
            return _forecast(
                {fixture.case.fiscal_years[0]: {"revenue": D(1)}},
                unit="USD millions",
            )

    result = asyncio.run(
        ForecastBacktestRunner(reasoned_service=Fake(), normalized_baseline=Fake()).run(
            fixture.case, fixture.actual_outcomes
        )
    )
    assert deterministic_machine_report(result) == deterministic_machine_report(result)
    human = human_report(result)
    assert tuple(line for line in human.splitlines() if line in HUMAN_SECTIONS) == HUMAN_SECTIONS
    assert "generated_at" not in human


def test_runner_calls_normalized_and_hybrid_with_same_cutoff_and_explicit_routes():
    fixture = _fixture()
    calls = []

    class Reasoned:
        async def forecast(self, financials, _input, parameters, **kwargs):
            calls.append(("reasoned", financials, parameters, kwargs))
            return _forecast(
                {fixture.case.fiscal_years[0]: {"revenue": D(1)}},
                unit="USD millions",
            )

    class Baseline:
        async def forecast(self, financials, parameters, **kwargs):
            calls.append(("baseline", financials, parameters, kwargs))
            return _forecast(
                {fixture.case.fiscal_years[0]: {"revenue": D(2)}},
                unit="USD millions",
            )

    result = asyncio.run(
        ForecastBacktestRunner(
            reasoned_service=Reasoned(), normalized_baseline=Baseline(), hybrid_baseline=Baseline()
        ).run(fixture.case, fixture.actual_outcomes)
    )
    assert [item[0] for item in calls] == ["reasoned", "baseline", "baseline"]
    assert calls[1][1] == calls[2][1]
    assert calls[1][2] == calls[2][2] == FcffForecastParameters(forecast_years=1)
    assert calls[1][3]["availability_mode"].value == "point_in_time"
    assert result.normalized_comparison.route.route == "normalized"
    assert result.hybrid_comparison.route.route == "hybrid"


def test_production_reasoner_prompt_and_no_provider_are_untouched():
    assert "Never browse" in FORECAST_REASONER_INSTRUCTIONS
    assert "scenario" not in FORECAST_REASONER_INSTRUCTIONS.casefold()


def test_normalized_fact_requires_filed_or_linked_availability_before_call():
    fixture = _fixture()
    first = fixture.case.point_in_time_financials.observations[0]
    changed = first.model_copy(update={"filed": None})
    financials = fixture.case.point_in_time_financials.model_copy(
        update={"observations": [changed, *fixture.case.point_in_time_financials.observations[1:]]}
    )
    case = fixture.case.model_copy(update={"point_in_time_financials": financials})
    calls = []

    class Fake:
        async def forecast(self, *_args, **_kwargs):
            calls.append(True)
            return _forecast({2024: {"revenue": D(1)}})

    audit = audit_case(case)
    assert not audit.valid
    assert any("no availability date" in issue for issue in audit.issues)
    with pytest.raises(LeakageError):
        asyncio.run(ForecastBacktestRunner(reasoned_service=Fake()).run(case, fixture.actual_outcomes))
    assert calls == []


def test_explicit_availability_manifest_replaces_period_end_fallback():
    fixture = _fixture()
    first = next(
        item
        for item in fixture.case.point_in_time_financials.observations
        if item.fiscal_year == 2023 and item.concept == FinancialConcept.REVENUE
    )
    changed = first.model_copy(update={"filed": None})
    financials = fixture.case.point_in_time_financials.model_copy(
        update={
            "observations": [
                changed,
                *(
                    item
                    for item in fixture.case.point_in_time_financials.observations
                    if item != first
                ),
            ]
        }
    )
    identity = canonical_information_identity(changed, "normalized_fact")
    record = InformationAvailabilityRecord(
        identity=identity,
        category="normalized_fact",
        available_on=datetime.date(2024, 2, 15),
        source="KO 2023 10-K",
        source_id="0000021344-24-000009",
        provenance="explicit test availability link",
        content_hash=canonical_information_content_hash(changed),
    )
    case = fixture.case.model_copy(
        update={"point_in_time_financials": financials, "availability_manifest": (record,)}
    )
    assert audit_case(case).valid
    assert len(cutoff_financials(financials, as_of=case.as_of, availability_manifest=(record,)).observations) == len(financials.observations)
    stale_financials = financials.model_copy(
        update={
            "observations": [
                changed.model_copy(update={"value": changed.value + 1}),
                *financials.observations[1:],
            ]
        }
    )
    stale_case = case.model_copy(update={"point_in_time_financials": stale_financials})
    assert not audit_case(stale_case).valid
    assert any("content hash" in issue for issue in audit_case(stale_case).issues)
    with pytest.raises(ValueError, match="content hash"):
        cutoff_financials(
            stale_financials,
            as_of=case.as_of,
            availability_manifest=(record,),
        )
    with pytest.raises(ValueError, match="category metadata"):
        cutoff_financials(
            financials,
            as_of=case.as_of,
            availability_manifest=(record.model_copy(update={"category": "research"}),),
        )
    with pytest.raises(ValueError, match="source metadata"):
        cutoff_financials(
            financials,
            as_of=case.as_of,
            availability_manifest=(record.model_copy(update={"source_id": "other"}),),
        )


def test_default_normalized_baseline_is_real_service():
    assert isinstance(ForecastBacktestRunner().normalized_baseline, FcffForecastService)


def test_normalized_empty_baseline_fails_clearly():
    fixture = _fixture()

    class Empty:
        async def forecast(self, *_args, **_kwargs):
            return SimpleNamespace(observations=())

    class Reasoned:
        async def forecast(self, *_args, **_kwargs):
            return _forecast({2024: {"revenue": D(1)}})

    with pytest.raises(ValueError, match="empty forecast rows"):
        asyncio.run(ForecastBacktestRunner(reasoned_service=Reasoned(), normalized_baseline=Empty()).run(fixture.case, fixture.actual_outcomes))


def test_normalized_wrong_horizon_fails_before_comparison():
    fixture = _fixture()

    class Wrong:
        async def forecast(self, *_args, **_kwargs):
            return _forecast({2025: {"revenue": D(1)}})

    with pytest.raises(ValueError, match="does not match required years"):
        asyncio.run(ForecastBacktestRunner(reasoned_service=Wrong(), normalized_baseline=Wrong()).run(fixture.case, fixture.actual_outcomes))


def test_normalized_without_scoreable_required_metric_fails():
    fixture = _fixture()

    class Fake:
        async def forecast(self, *_args, **_kwargs):
            return _forecast({2024: {"revenue": D(1)}})

    actuals = ActualOutcomeData(company=fixture.case.company, ticker=fixture.case.ticker)
    with pytest.raises(ValueError, match="no scoreable required metrics"):
        asyncio.run(ForecastBacktestRunner(reasoned_service=Fake(), normalized_baseline=Fake()).run(fixture.case, actuals))


def test_reconstruction_rejects_mixed_monetary_units_before_arithmetic():
    financials = _reconstruction_financials()
    changed = financials.observations[1].model_copy(update={"unit": "USD millions"})
    mixed = financials.model_copy(
        update={
            "observations": [
                changed,
                *financials.observations[:1],
                *financials.observations[2:],
            ]
        }
    )
    with pytest.raises(ValueError, match="incompatible monetary units"):
        reconstruct_actual_outcomes(mixed, fiscal_years=(2024,))


def test_reconstruction_does_not_blindly_abs_unknown_negative_capex():
    financials = _reconstruction_financials()
    capex_index = next(index for index, item in enumerate(financials.observations) if item.concept == FinancialConcept.CAPITAL_EXPENDITURES and item.fiscal_year == 2024)
    changed = financials.observations[capex_index].model_copy(update={"source_concept": "UnknownCashFlowConcept"})
    observations = list(financials.observations)
    observations[capex_index] = changed
    actuals = reconstruct_actual_outcomes(financials.model_copy(update={"observations": observations}), fiscal_years=(2024,))
    capex = next(item for item in actuals.observations if item.metric == "capex")
    assert capex.value is None
    assert capex.sign_normalization == "unavailable_unknown_sign_convention"


def test_sep_to_dec_fiscal_pair_has_no_delta_but_52_week_pair_does():
    sep_to_dec = reconstruct_actual_outcomes(
        _reconstruction_financials(datetime.date(2024, 12, 31)), fiscal_years=(2024,)
    )
    assert next(item for item in sep_to_dec.observations if item.metric == "delta_nwc").value is None
    assert next(item for item in sep_to_dec.observations if item.metric == "fcff").value is None
    noncalendar = reconstruct_actual_outcomes(_reconstruction_financials(), fiscal_years=(2024,))
    assert next(item for item in noncalendar.observations if item.metric == "delta_nwc").value == D(3)


def test_actual_assumption_duplicate_and_incompatible_units_are_explicit():
    first = ActualAssumptionOutcome(target_type="forecast_metric", metric="revenue", fiscal_year=2024, actual=1, unit="USD")
    with pytest.raises(ValidationError, match="unique"):
        ActualOutcomeData(assumption_outcomes=(first, first))
    assumption = _assumption(low=(D(0), D(0)), base=(D(1), D(1)), high=(D(2), D(2)))
    incompatible = ActualOutcomeData(
        assumption_outcomes=(
            ActualAssumptionOutcome(target_type="forecast_metric", metric="revenue", fiscal_year=2025, actual=1, unit="EUR"),
            ActualAssumptionOutcome(target_type="forecast_metric", metric="revenue", fiscal_year=2026, actual=1, unit="EUR"),
        )
    )
    scored = score_assumptions(SimpleNamespace(accepted_assumptions=(assumption,)), incompatible)
    assert all(not item.scored and "incompatible unit" in item.unmatched_reason for item in scored.scores)


@pytest.mark.parametrize(
    ("first_metric", "second_metric"),
    (("EBIT", "operating_income"), ("CAPEX", "capital_expenditures")),
)
def test_actual_financial_aliases_are_unique_by_canonical_metric(
    first_metric, second_metric
):
    observation_kwargs = {
        "fiscal_year": 2024,
        "period_end": datetime.date(2024, 12, 31),
        "value": 1,
        "unit": "USD",
    }
    with pytest.raises(ValidationError, match="unique"):
        ActualOutcomeData(
            observations=(
                ActualFinancialObservation(metric=first_metric, **observation_kwargs),
                ActualFinancialObservation(metric=second_metric, **observation_kwargs),
            )
        )


def test_required_years_emit_unscored_baseline_rows():
    report = score_financials(
        _forecast({2024: {"revenue": D(1)}}),
        _actuals(values={2024: {"revenue": D(1)}, 2025: {"revenue": D(1)}}),
        required_years=(2024, 2025),
    )
    missing = next(item for item in report.scores if item.metric == "revenue" and item.fiscal_year == 2025)
    assert missing.scored is False and missing.forecast is None


def test_financial_scores_require_matching_currency_unit_and_scale():
    actual_usd_millions = _actuals(values={2025: {"revenue": D(100)}})
    actual_eur = ActualOutcomeData(
        observations=(
            ActualFinancialObservation(
                fiscal_year=2025,
                period_end=datetime.date(2025, 12, 31),
                metric="revenue",
                value=100,
                unit="EUR",
                currency="EUR",
                scale=1,
                source="typed source",
            ),
        )
    )
    usd = score_financials(
        _forecast({2025: {"revenue": D(100)}}), actual_usd_millions
    )
    euros = score_financials(_forecast({2025: {"revenue": D(100)}}), actual_eur)
    assert next(item for item in usd.scores if item.metric == "revenue").scored
    eur_score = next(item for item in euros.scores if item.metric == "revenue")
    assert not eur_score.scored and "incompatible" in eur_score.unmatched_reason
    scale_score = next(
        item
        for item in score_financials(
            _forecast({2025: {"revenue": D(100)}}),
            ActualOutcomeData(
                observations=(
                    ActualFinancialObservation(
                        fiscal_year=2025,
                        period_end=datetime.date(2025, 12, 31),
                        metric="revenue",
                        value=100,
                        unit="USD millions",
                        currency="USD",
                        scale=1,
                        source="typed source",
                    ),
                )
            ),
        ).scores
        if item.metric == "revenue"
    )
    assert not scale_score.scored and "incompatible" in scale_score.unmatched_reason
    million_forecast = _forecast(
        {2025: {"revenue": D(100)}}, unit="USD millions"
    )
    million_actual = ActualOutcomeData(
        observations=(
            ActualFinancialObservation(
                fiscal_year=2025,
                period_end=datetime.date(2025, 12, 31),
                metric="revenue",
                value=100,
                unit="USD millions",
                currency="USD",
                scale=1,
                source="typed source",
            ),
        )
    )
    assert next(
        item
        for item in score_financials(million_forecast, million_actual).scores
        if item.metric == "revenue"
    ).scored


def test_stability_rejects_raw_input_and_future_research_before_reasoner():
    fixture = _fixture()

    class Fake:
        def __init__(self):
            self.calls = 0

        async def reason(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(assumptions=())

    fake = Fake()
    with pytest.raises(TypeError, match="complete ForecastBacktestCase"):
        asyncio.run(StabilityEvaluator(fake).evaluate(fixture.case.reasoning_input))
    future = MarketGrowthEvidence(
        market="future market",
        source_date=datetime.date(2025, 1, 1),
        source_type="analyst_estimate",
        low=1,
        base=2,
        high=3,
        provenance=EvidenceProvenance(source="future source"),
        unit="percent",
    )
    case = fixture.case.model_copy(
        update={"reasoning_input": fixture.case.reasoning_input.model_copy(update={"research_evidence": (future,)})}
    )
    with pytest.raises(LeakageError):
        asyncio.run(StabilityEvaluator(fake).evaluate(case))
    assert fake.calls == 0


def test_stability_without_refresh_support_fails_before_any_repeat_call():
    fixture = _fixture()

    class CachedOnly:
        def __init__(self):
            self.calls = 0

        async def reason(self, _input):
            self.calls += 1
            return SimpleNamespace(assumptions=())

    reasoner = CachedOnly()
    with pytest.raises(ValueError, match="force_refresh"):
        asyncio.run(StabilityEvaluator(reasoner).evaluate(fixture.case, runs=2))
    assert reasoner.calls == 0


def test_hybrid_different_evidence_is_rejected_before_collaborator():
    fixture = _fixture()
    calls = []

    class Fake:
        async def forecast(self, *_args, **_kwargs):
            calls.append(True)
            return _forecast({2024: {"revenue": D(1)}})

    case = fixture.case.model_copy(update={"hybrid_evidence": {"observations": []}})
    with pytest.raises(ValueError, match="hybrid evidence identity"):
        asyncio.run(ForecastBacktestRunner(reasoned_service=Fake(), normalized_baseline=Fake()).run(case, fixture.actual_outcomes))
    assert calls == []


def test_metric_period_end_evidence_is_not_classified_as_actual_by_heuristic():
    fixture = _fixture()
    case = fixture.case.model_copy(
        update={
            "evidence_snapshot": {
                "evidence_id": "throughput-2023",
                "metric": "throughput",
                "period_end": "2023-12-31",
                "source_date": "2024-02-15",
            }
        }
    )
    assert audit_case(case).valid


def test_unknown_evidence_container_and_generic_as_of_are_not_dates():
    fixture = _fixture()
    unknown = fixture.case.model_copy(update={"evidence_snapshot": {"mystery": {"value": 1}}})
    assert not audit_case(unknown).valid
    as_of_only = fixture.case.model_copy(
        update={"evidence_snapshot": {"as_of": "2024-03-01"}}
    )
    assert not audit_case(as_of_only).valid


def test_case_identity_is_stable_across_hash_seeds_for_sets():
    code = (
        "from edgarito.evaluation import load_fixture; "
        "f=load_fixture('ko'); "
        "c=f.case.model_copy(update={'evidence_snapshot': {'tags': {'b', 'a'}}}); "
        "print(c.case_id)"
    )
    values = []
    for seed in ("1", "37"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        values.append(subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip())
    assert values[0] == values[1]


def test_fixtures_use_exact_fy2024_reported_subset_after_fy2023_cutoff():
    ko = load_fixture("ko")
    visa = load_fixture("v")
    tesla = load_fixture("tsla")
    assert ko.case.as_of == datetime.date(2024, 3, 1)
    assert ko.case.fiscal_years == (2024,)
    assert {item.fiscal_year for item in ko.case.point_in_time_financials.observations} == {2022, 2023}
    assert next(
        item
        for item in ko.case.point_in_time_financials.observations
        if item.fiscal_year == 2023 and item.concept == FinancialConcept.REVENUE
    ).value == D("45754")
    assert next(
        item
        for item in ko.case.point_in_time_financials.observations
        if item.fiscal_year == 2023 and item.concept == FinancialConcept.OPERATING_INCOME
    ).value == D("11311")
    assert next(item for item in ko.actual_outcomes.observations if item.metric == "revenue").value == D("47061")
    assert next(item for item in ko.actual_outcomes.observations if item.metric == "ebit").value == D("9992")
    assert next(item for item in visa.actual_outcomes.observations if item.metric == "revenue").value == D("35926")
    assert next(item for item in visa.actual_outcomes.observations if item.metric == "ebit").value == D("23595")
    assert next(item for item in tesla.actual_outcomes.observations if item.metric == "revenue").value == D("97690")
    assert next(item for item in tesla.actual_outcomes.observations if item.metric == "ebit").value == D("7076")
    for fixture in (ko, visa, tesla):
        assert all(item.source_id and item.source_concept and item.source_date for item in fixture.actual_outcomes.observations)
