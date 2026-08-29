import asyncio
import importlib
from decimal import Decimal
from types import SimpleNamespace

import pytest

from edgarito.cli import main
from edgarito.cli.parser import build_parser
from edgarito.config.valuation import ForecastMethod
from edgarito.schemas.forecasting import (
    FcffForecast,
    FcffForecastDriver,
    FcffForecastMethod,
    FcffForecastParameters,
    ForecastAssumptionSource,
    ForecastDecision,
    ForecastMetric,
    ForecastOverride,
    ForecastPlan,
    ForecastScope,
    ForecastStrategy,
)
from edgarito.services.forecasting._fcff.paths import effective_tax_rate
from edgarito.services.forecasting.orchestration import (
    DriverBasedForecastIncompleteError,
    FcffForecastOrchestrationService,
)
from edgarito.services.forecasting.plan import FcffForecastPlanService
from edgarito.services.operating.contracts import (
    OperatingForecastQualityError,
    OperatingForecastQualityResult,
)


def _canonical_fcff_forecast() -> FcffForecast:
    return FcffForecast(
        provider="test",
        company_id="test-company",
        company_name="Test Company",
        base_fiscal_year=2024,
        base_period_end="2024-12-31",
        base_revenue=Decimal("100"),
        base_operating_income=Decimal("20"),
        base_tax_rate=Decimal("20"),
        base_nopat=Decimal("16"),
        base_depreciation_and_amortization=Decimal("4"),
        base_capital_expenditures=Decimal("5"),
        base_operating_working_capital=Decimal("10"),
        unit="USD",
        parameters=FcffForecastParameters(forecast_years=1),
        historical_fiscal_years=(2024,),
        assumption_sources={
            driver: ForecastAssumptionSource.TRAILING_AVERAGE
            for driver in FcffForecastDriver
        },
    )


def test_forecast_metric_covers_future_driver_based_financial_lines():
    assert {
        ForecastMetric.GROSS_MARGIN.value,
        ForecastMetric.GROSS_PROFIT.value,
        ForecastMetric.R_AND_D.value,
        ForecastMetric.SG_AND_A.value,
        ForecastMetric.TAX.value,
        ForecastMetric.DEPRECIATION_AND_AMORTIZATION.value,
        ForecastMetric.CAPEX.value,
        ForecastMetric.OPERATING_WORKING_CAPITAL.value,
        ForecastMetric.DELTA_NWC.value,
    } <= {metric.value for metric in ForecastMetric}


@pytest.mark.parametrize(
    ("pretax", "tax", "expected"),
    [
        (Decimal("100"), Decimal("25"), Decimal("25")),
        (Decimal("100"), Decimal("0"), Decimal("0")),
        (Decimal("0"), Decimal("0"), None),
        (Decimal("-100"), Decimal("25"), None),
        (Decimal("100"), Decimal("-1"), None),
        (Decimal("100"), Decimal("101"), None),
    ],
)
def test_fcff_effective_tax_rate_retains_strict_policy(pretax, tax, expected):
    period = SimpleNamespace(
        pretax_income=pretax,
        income_tax_expense=tax,
    )
    assert effective_tax_rate(period) == expected


def test_plan_contracts_are_immutable_unique_and_stably_serialized():
    decisions = (
        ForecastDecision(
            scope="segment",
            scope_id="zeta",
            metric="revenue",
            strategy="driver",
        ),
        ForecastDecision(
            scope="company",
            metric="revenue",
            strategy="consolidated",
        ),
    )
    plan = ForecastPlan(
        requested="AUTO",
        resolved="HYBRID",
        decisions=decisions,
        audit_records=("quality passed",),
    )

    assert plan.decisions[0].scope == ForecastScope.COMPANY
    assert plan.model_dump_json() == plan.model_copy().model_dump_json()
    with pytest.raises((TypeError, ValueError)):
        plan.resolved = FcffForecastMethod.NORMALIZED

    with pytest.raises(ValueError, match="unique by scope and metric"):
        ForecastPlan(
            requested="normalized",
            resolved="normalized",
            decisions=(
                ForecastDecision(
                    scope="company", metric="revenue", strategy="consolidated"
                ),
                ForecastDecision(scope="company", metric="revenue", strategy="ratio"),
            ),
        )


@pytest.mark.parametrize(
    "method",
    [
        FcffForecastMethod.NORMALIZED,
        FcffForecastMethod.HYBRID,
        FcffForecastMethod.DRIVER_BASED,
    ],
)
def test_explicit_plan_methods_must_resolve_to_themselves(method):
    assert ForecastPlan(requested=method, resolved=method).resolved == method
    invalid_resolution = (
        FcffForecastMethod.HYBRID
        if method != FcffForecastMethod.HYBRID
        else FcffForecastMethod.NORMALIZED
    )
    with pytest.raises(ValueError, match="resolve to their requested method"):
        ForecastPlan(requested=method, resolved=invalid_resolution)


@pytest.mark.parametrize("resolved", ["auto", "driver_based"])
def test_auto_plan_can_only_resolve_to_executable_method(resolved):
    with pytest.raises(ValueError, match="AUTO forecast plans"):
        ForecastPlan(requested="auto", resolved=resolved)


def test_orchestration_revalidates_custom_plan_method_resolution():
    class InvalidPlanner:
        def plan(self, *args, **kwargs):
            return SimpleNamespace(
                requested=FcffForecastMethod.DRIVER_BASED,
                resolved=FcffForecastMethod.NORMALIZED,
            )

    with pytest.raises(ValueError, match="resolve to their requested method"):
        FcffForecastOrchestrationService(plan_service=InvalidPlanner()).forecast(
            "financials", "parameters", "driver_based"
        )


def test_company_and_segment_scope_ids_are_unambiguous_for_decisions_and_overrides():
    with pytest.raises(ValueError, match="Company forecast decisions"):
        ForecastDecision(
            scope="company",
            scope_id="another-company",
            metric="revenue",
            strategy="consolidated",
        )
    with pytest.raises(ValueError, match="Segment forecast decisions"):
        ForecastDecision(scope="segment", metric="revenue", strategy="driver")
    with pytest.raises(ValueError, match="Company forecast overrides"):
        ForecastOverride(
            scope="company",
            scope_id="another-company",
            metric="revenue",
            strategy="explicit",
        )
    with pytest.raises(ValueError, match="Segment forecast overrides"):
        ForecastOverride(scope="segment", metric="revenue", strategy="driver")


def test_override_cannot_create_an_ambiguous_second_company_scope():
    override = ForecastOverride(
        scope="company",
        metric="revenue",
        strategy="explicit",
    )
    assert override.scope_id == "company"
    with pytest.raises(ValueError, match="Company forecast overrides"):
        ForecastOverride(
            scope="company",
            scope_id="issuer-alias",
            metric="revenue",
            strategy="explicit",
        )


def test_manual_override_replaces_a_decision_and_preserves_explicit_path():
    override = ForecastOverride(
        scope=ForecastScope.COMPANY,
        metric=ForecastMetric.REVENUE_GROWTH,
        strategy=ForecastStrategy.EXPLICIT,
        explicit_path=(Decimal("4"), Decimal("3")),
        provenance="manual scenario",
    )
    plan = FcffForecastPlanService().plan(
        FcffForecastMethod.NORMALIZED,
        overrides=(override,),
    )

    decision = plan.decision(ForecastScope.COMPANY, ForecastMetric.REVENUE_GROWTH)
    assert decision is not None
    assert decision.strategy == ForecastStrategy.EXPLICIT
    assert decision.explicit_path == (Decimal("4"), Decimal("3"))
    assert decision.provenance == "manual scenario"
    assert plan.overrides == (override,)


@pytest.mark.parametrize("method", ["normalized", "hybrid"])
def test_consolidated_company_decisions_describe_only_modeled_material_metrics(method):
    plan = FcffForecastPlanService().plan(
        method,
        evidence={"segments": ({"segment_id": "cloud"},)},
    )

    company_metrics = {
        item.metric for item in plan.decisions if item.scope == ForecastScope.COMPANY
    }
    assert company_metrics == {
        ForecastMetric.REVENUE,
        ForecastMetric.OPERATING_MARGIN,
        ForecastMetric.TAX,
        ForecastMetric.DEPRECIATION_AND_AMORTIZATION,
        ForecastMetric.CAPEX,
        ForecastMetric.OPERATING_WORKING_CAPITAL,
        ForecastMetric.DELTA_NWC,
    }
    assert all(
        item.strategy == ForecastStrategy.CONSOLIDATED for item in plan.decisions
        if item.scope == ForecastScope.COMPANY
    )
    if method == "normalized":
        assert not any(item.scope == ForecastScope.SEGMENT for item in plan.decisions)
    else:
        segment = plan.decision(ForecastScope.SEGMENT, ForecastMetric.REVENUE, "cloud")
        assert segment is not None
        assert segment.strategy == ForecastStrategy.DRIVER


def test_driver_based_segment_plan_represents_revenue_and_gross_profit_drivers():
    plan = FcffForecastPlanService().plan(
        "driver_based",
        evidence={"segments": ({"segment_id": "cloud"},)},
    )

    segment_decisions = {
        item.metric for item in plan.decisions if item.scope == ForecastScope.SEGMENT
    }
    assert segment_decisions == {
        ForecastMetric.REVENUE,
        ForecastMetric.GROSS_MARGIN,
        ForecastMetric.GROSS_PROFIT,
    }
    assert all(
        item.strategy == ForecastStrategy.DRIVER
        for item in plan.decisions
        if item.scope == ForecastScope.SEGMENT
    )
    company_costs = {
        ForecastMetric.R_AND_D,
        ForecastMetric.SG_AND_A,
        ForecastMetric.TAX,
        ForecastMetric.CAPEX,
        ForecastMetric.DEPRECIATION_AND_AMORTIZATION,
        ForecastMetric.OPERATING_WORKING_CAPITAL,
        ForecastMetric.DELTA_NWC,
    }
    company_metrics = {
        item.metric for item in plan.decisions if item.scope == ForecastScope.COMPANY
    }
    assert company_metrics == company_costs | {
        ForecastMetric.REVENUE,
        ForecastMetric.OPERATING_MARGIN,
        ForecastMetric.GROSS_MARGIN,
        ForecastMetric.GROSS_PROFIT,
    }
    assert all(
        item.strategy == ForecastStrategy.DRIVER
        for item in plan.decisions
        if item.scope == ForecastScope.COMPANY and item.metric in company_costs
    )
    assert all(
        item.strategy == ForecastStrategy.DRIVER
        for item in plan.decisions
        if item.scope == ForecastScope.COMPANY
        and item.metric
        in {ForecastMetric.GROSS_MARGIN, ForecastMetric.GROSS_PROFIT}
    )
    assert not {
        ForecastMetric.REVENUE_GROWTH,
        ForecastMetric.TAX_RATE,
        ForecastMetric.DEPRECIATION_TO_REVENUE,
        ForecastMetric.CAPEX_TO_REVENUE,
        ForecastMetric.OPERATING_WORKING_CAPITAL_TO_REVENUE,
    } & {
        item.metric for item in plan.decisions if item.scope == ForecastScope.COMPANY
    }


@pytest.mark.parametrize(
    ("quality", "expected", "reason"),
    [
        (
            SimpleNamespace(accepted=True, reason="coverage passed"),
            "hybrid",
            "coverage passed",
        ),
        (
            SimpleNamespace(accepted=False, reason="coverage failed"),
            "normalized",
            "coverage failed",
        ),
    ],
)
def test_auto_reuses_quality_assessment_and_records_rationale(
    quality, expected, reason
):
    plan = FcffForecastPlanService().plan(
        "auto",
        operating_quality=quality,
        evidence={"segments": ({"segment_id": "cloud"},)},
    )

    assert plan.resolved == FcffForecastMethod(expected)
    assert reason in plan.rationale
    assert plan.requested == FcffForecastMethod.AUTO
    if expected == "hybrid":
        segment = plan.decision("segment", "revenue", "cloud")
        assert segment is not None
        assert segment.strategy == ForecastStrategy.DRIVER
    else:
        assert not any(item.scope == ForecastScope.SEGMENT for item in plan.decisions)


def test_supplied_rejected_quality_cannot_be_overridden_by_capability():
    capability_calls = []

    def capability(_evidence):
        capability_calls.append(True)
        return SimpleNamespace(accepted=True, reason="capability passed")

    plan = FcffForecastPlanService().plan(
        "auto",
        operating_quality=SimpleNamespace(
            accepted=False, reason="quality gate rejected"
        ),
        operating_capability=capability,
        evidence={"segments": ({"segment_id": "cloud"},)},
    )

    assert plan.resolved == FcffForecastMethod.NORMALIZED
    assert capability_calls == []
    assert "quality gate rejected" in plan.rationale


def test_auto_without_an_operating_quality_assessment_is_normalized():
    plan = FcffForecastPlanService().plan(
        "auto", evidence={"segments": ({"segment_id": "cloud"},)}
    )

    assert plan.resolved == FcffForecastMethod.NORMALIZED
    assert "no operating capability" in plan.rationale


def test_planner_does_not_swallow_quality_gate_errors():
    def failing_gate(_evidence):
        raise LookupError("quality assessment failed")

    with pytest.raises(LookupError, match="quality assessment failed"):
        FcffForecastPlanService(quality_gate=failing_gate).plan(
            "auto", evidence={"segments": ()}
        )


def test_driver_based_is_plannable_but_never_falls_back():
    plan = FcffForecastPlanService().plan("driver_based")
    assert plan.resolved == FcffForecastMethod.DRIVER_BASED
    assert any(item.strategy == ForecastStrategy.DRIVER for item in plan.decisions)

    with pytest.raises(DriverBasedForecastIncompleteError, match="no fallback"):
        FcffForecastOrchestrationService().forecast(None, None, "driver_based")


def test_orchestration_normalized_delegates_without_changing_canonical_result():
    canonical = _canonical_fcff_forecast()

    class FakeFcff:
        def forecast(self, financials, parameters, **kwargs):
            assert financials == "financials"
            assert parameters == "parameters"
            return canonical

    result = FcffForecastOrchestrationService(fcff_service=FakeFcff()).forecast(
        "financials", "parameters", "normalized"
    )

    assert result.forecast is canonical
    assert result.plan.resolved == FcffForecastMethod.NORMALIZED


def test_orchestration_hybrid_returns_pipeline_forecast_without_recalculation():
    canonical = _canonical_fcff_forecast()

    class FakePipeline:
        def forecast(self, financials, evidence, parameters, **kwargs):
            assert financials == "financials"
            assert evidence == "evidence"
            assert parameters == "parameters"
            return SimpleNamespace(forecast=canonical)

    result = FcffForecastOrchestrationService(
        operating_pipeline=FakePipeline()
    ).forecast(
        "financials",
        "parameters",
        "hybrid",
        evidence="evidence",
    )

    assert result.forecast is canonical
    assert result.plan.resolved == FcffForecastMethod.HYBRID


def test_auto_falls_back_after_final_hybrid_quality_gate_rejection():
    canonical = _canonical_fcff_forecast()
    accepted = SimpleNamespace(accepted=True, reason="preliminary quality passed")
    rejected = OperatingForecastQualityResult(
        accepted=False,
        reason="reconstruction error=0.25 (maximum 0.10)",
    )
    calls = []

    class FakeFcff:
        def forecast(self, financials, parameters, **kwargs):
            calls.append((financials, parameters))
            return canonical

    class RejectingPipeline:
        def forecast(self, financials, evidence, parameters, **kwargs):
            calls.append(("pipeline", financials, evidence, parameters))
            raise OperatingForecastQualityError(rejected)

    result = FcffForecastOrchestrationService(
        fcff_service=FakeFcff(),
        operating_pipeline=RejectingPipeline(),
    ).forecast(
        "financials",
        "parameters",
        "auto",
        evidence="evidence",
        operating_quality=accepted,
    )

    assert result.forecast is canonical
    assert result.plan.requested == FcffForecastMethod.AUTO
    assert result.plan.resolved == FcffForecastMethod.NORMALIZED
    assert rejected.reason in result.plan.rationale
    assert any(rejected.reason in record for record in result.plan.audit)
    assert calls[0][0] == "pipeline"
    assert len(calls) == 2


def test_explicit_hybrid_propagates_final_quality_rejection():
    rejection = OperatingForecastQualityResult(
        accepted=False, reason="driver coverage=0.4 (minimum 0.60)"
    )

    class RejectingPipeline:
        def forecast(self, financials, evidence, parameters, **kwargs):
            raise OperatingForecastQualityError(rejection)

    with pytest.raises(OperatingForecastQualityError, match="driver coverage=0.4"):
        FcffForecastOrchestrationService(
            operating_pipeline=RejectingPipeline()
        ).forecast(
            "financials",
            "parameters",
            "hybrid",
            evidence="evidence",
            operating_quality=SimpleNamespace(accepted=True),
        )


def test_default_hybrid_pipeline_reuses_orchestration_fcff_service(monkeypatch):
    orchestration = importlib.import_module(
        "edgarito.services.forecasting.orchestration"
    )
    integration = importlib.import_module("edgarito.services.operating.integration")
    canonical = _canonical_fcff_forecast()
    fcff_service = object()
    seen = {}

    class SpyPipeline:
        @staticmethod
        def quality_gate(_evidence):
            return SimpleNamespace(accepted=True)

        def __init__(self, *, fcff_service):
            seen["fcff_service"] = fcff_service

        def forecast(self, financials, evidence, parameters, **kwargs):
            return SimpleNamespace(forecast=canonical)

    monkeypatch.setattr(integration, "OperatingForecastPipelineService", SpyPipeline)
    result = orchestration.FcffForecastOrchestrationService(
        fcff_service=fcff_service
    ).forecast("financials", "parameters", "hybrid", evidence="evidence")

    assert result.forecast is canonical
    assert seen["fcff_service"] is fcff_service


def test_forecast_cli_auto_propagates_unrelated_value_error(monkeypatch):
    forecast_cli = importlib.import_module("edgarito.cli.use_cases.forecast")
    profile = SimpleNamespace(
        forecast=SimpleNamespace(
            default_method=ForecastMethod.FCFF,
            fcff=SimpleNamespace(
                forecast_years=1,
                revenue_growth=None,
                operating_margin=None,
                tax_rate=None,
                depreciation_to_revenue=None,
                capex_to_revenue=None,
                operating_working_capital_to_revenue=None,
                revenue_anchors={},
                assumption_source_overrides={},
                historical_window=3,
            ),
        )
    )
    args = SimpleNamespace(
        ticker="TEST",
        profile=None,
        forecast_method=None,
        fcff_forecast_method="auto",
        operating_margin=None,
        tax_rate=None,
        depreciation_to_revenue=None,
        capex_to_revenue=None,
        operating_working_capital_to_revenue=None,
        years=None,
        revenue_growth=None,
        fcf_margin=None,
        historical_window=None,
    )

    class FakeFcff:
        def required_concepts(self):
            return {"revenue"}

        def forecast(self, financials, parameters):
            return _canonical_fcff_forecast()

    async def retrieve(args, granularity, concepts):
        return "financials"

    async def unrelated(*args, **kwargs):
        raise ValueError("unrelated orchestration failure")

    monkeypatch.setattr(
        forecast_cli, "load_selected_valuation_profile", lambda args: profile
    )
    monkeypatch.setattr(
        forecast_cli,
        "fcff_parameters",
        lambda args, configured: FcffForecastParameters(forecast_years=1),
    )
    monkeypatch.setattr(forecast_cli, "retrieve_financials", retrieve)
    monkeypatch.setattr(forecast_cli, "FcffForecastService", FakeFcff)
    monkeypatch.setattr(forecast_cli, "_retrieve_hybrid_evidence", unrelated)

    with pytest.raises(ValueError, match="unrelated orchestration failure"):
        asyncio.run(forecast_cli.run_forecast(args))


def test_cli_keeps_legacy_selectors_distinct_and_defaults_unset():
    parser = build_parser()
    forecast = parser.parse_args(["forecast", "--ticker", "TEST"])
    valuation = parser.parse_args(["valuation", "--ticker", "TEST"])

    assert forecast.forecast_method is None
    assert forecast.fcff_forecast_method is None
    assert valuation.fcff_forecast_method is None
    assert valuation.projection_method is None
    assert (
        parser.parse_args(
            ["forecast", "--ticker", "TEST", "--fcff-forecast-method", "hybrid"]
        ).fcff_forecast_method
        == "hybrid"
    )


def test_cli_driver_based_method_reports_the_dedicated_error(capsys):
    with pytest.raises(SystemExit):
        main(
            [
                "forecast",
                "--ticker",
                "TEST",
                "--fcff-forecast-method",
                "driver_based",
            ]
        )

    assert "driver_based" in capsys.readouterr().err
