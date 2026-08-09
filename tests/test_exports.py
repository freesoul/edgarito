import datetime
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.enums.provider import ProviderName
from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.red_flags import (
    RedFlag,
    RedFlagCategory,
    RedFlagEvidence,
    RedFlagSeverity,
    RedFlagSourceObservation,
    RedFlagsReport,
    RedFlagWarning,
)
from edgarito.schemas.valuation.intrinsic import (
    ExecutedValuation,
    InputProvenance,
    IntrinsicValuationResult,
    ModelWarning,
    ResolvedModelAssumption,
    SkippedValuation,
    ValuationConfidence,
    ValuationRunResult,
    WarningSeverity,
)
from edgarito.services.export import (
    AdaptiveMultistagePlanExport,
    CanonicalSecurityIdentifiers,
    CompanyAnalysisReportService,
    ExportSection,
    FcffForecastExport,
    FinancialDataExportService,
    ForecastExport,
    ForecastExportService,
    ForecastKind,
    MetricsExportService,
    RedFlagsExportService,
    ValuationExportService,
)
from edgarito.services.forecasting import (
    AdaptiveMultistagePlan,
    FcffForecast,
    FcffForecastDriver,
    FcffForecastObservation,
    FcffForecastParameters,
    ForecastAssumptionSource,
    SimplifiedFcfForecast,
    SimplifiedFcfForecastObservation,
    SimplifiedFcfForecastParameters,
)
from edgarito.services.metrics import CompanyMetrics, FinancialMetric, MetricObservation
from edgarito.services.valuation import (
    BusinessArchetype,
    CompanyLifecycle,
    Cyclicality,
    DataReadiness,
    ModelRole,
    ModelSuitability,
    ValuationModel,
    ValuationProfile,
    ValuationSelection,
)

PERIOD_END = datetime.date(2024, 12, 31)


def _financial_observation(
    concept: FinancialConcept = FinancialConcept.REVENUE,
) -> FinancialObservation:
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal("100"),
        unit="USD",
        granularity=Granularity.ANNUAL,
        fiscal_year=2024,
        fiscal_period=FiscalPeriod.FY,
        period_end=PERIOD_END,
        provider="sec",
        taxonomy="us-gaap",
        source_concept="Revenues",
        accession_number="0000000000-24-000001",
        form="10-K",
        filed=datetime.date(2025, 2, 1),
        derivation="reported source observation",
    )


def _financials() -> NormalizedCompanyFinancials:
    return NormalizedCompanyFinancials(
        provider="normalized-sec",
        company_id="1",
        company_name="Example Co",
        ticker="EX",
        retrieved_at=datetime.datetime(2025, 2, 2, 12, 0, tzinfo=datetime.timezone.utc),
        observations=[_financial_observation()],
    )


def test_financial_export_is_versioned_json_serializable_and_keeps_provenance():
    source = _financials()
    exported = FinancialDataExportService().export(source)

    assert exported.section == ExportSection.FINANCIAL_DATA
    assert exported.schema_version == 1
    assert exported.company_id == source.company_id
    assert exported.generated_at == source.retrieved_at
    assert exported.observations[0].accession_number == "0000000000-24-000001"
    assert exported.observations[0].source_concept == "Revenues"
    assert exported.observations[0].derivation == "reported source observation"
    assert "0000000000-24-000001" in exported.model_dump_json()
    with pytest.raises((TypeError, ValueError)):
        exported.observations = ()

    source.observations[0].source_concept = "changed"
    assert exported.observations[0].source_concept == "Revenues"


def test_identifier_mappings_are_detached_immutable_and_json_objects():
    source = _financials().model_copy(
        update={
            "identifiers": SecurityIdentifiers(
                ticker="EX",
                isin="US0378331005",
                cik=320193,
                exchange="NYSE",
                exchange_symbols={"nyse": "EX"},
                provider_symbols={ProviderName.SEC: "EX"},
            )
        }
    )

    exported = FinancialDataExportService().export(source)

    assert isinstance(exported.identifiers, CanonicalSecurityIdentifiers)
    assert exported.identifiers.provider_symbols == ((ProviderName.SEC, "EX"),)
    assert exported.identifiers.exchange_symbols == (("NYSE", "EX"),)
    payload = json.loads(exported.model_dump_json())["identifiers"]
    assert payload["provider_symbols"] == {"sec": "EX"}
    assert payload["exchange_symbols"] == {"NYSE": "EX"}

    with pytest.raises(TypeError):
        exported.identifiers.provider_symbols[0] = (ProviderName.SEC, "OTHER")
    with pytest.raises(TypeError):
        exported.identifiers.exchange_symbols[0] = ("NYSE", "OTHER")


def test_metrics_and_forecasts_copy_formulas_inputs_and_adaptive_plan():
    metrics = CompanyMetrics(
        provider="normalized-sec",
        company_id="1",
        company_name="Example Co",
        ticker="EX",
        observations=[
            MetricObservation(
                metric=FinancialMetric.OPERATING_MARGIN,
                value=Decimal("25"),
                unit="%",
                granularity=Granularity.ANNUAL,
                fiscal_year=2024,
                fiscal_period=FiscalPeriod.FY,
                period_end=PERIOD_END,
                provider="normalized-sec",
                formula="100 × operating income / revenue",
                input_concepts=(
                    FinancialConcept.OPERATING_INCOME,
                    FinancialConcept.REVENUE,
                ),
            )
        ],
    )
    metric_export = MetricsExportService().export(metrics)
    assert metric_export.observations[0].formula.startswith("100")
    assert metric_export.observations[0].input_concepts == (
        FinancialConcept.OPERATING_INCOME,
        FinancialConcept.REVENUE,
    )

    forecast = SimplifiedFcfForecast(
        provider="normalized-sec",
        company_id="1",
        company_name="Example Co",
        ticker="EX",
        base_fiscal_year=2024,
        base_period_end=PERIOD_END,
        base_revenue=Decimal("100"),
        base_free_cash_flow=Decimal("10"),
        unit="USD",
        parameters=SimplifiedFcfForecastParameters(
            forecast_years=1,
            revenue_growth=(Decimal("5"),),
            free_cash_flow_margin=(Decimal("11"),),
        ),
        historical_fiscal_years=(2023, 2024),
        revenue_growth_source=ForecastAssumptionSource.EXPLICIT,
        free_cash_flow_margin_source=ForecastAssumptionSource.EXPLICIT,
        observations=[
            SimplifiedFcfForecastObservation(
                forecast_year=1,
                fiscal_year=2025,
                period_end=datetime.date(2025, 12, 31),
                revenue_growth=Decimal("5"),
                revenue=Decimal("105"),
                free_cash_flow_margin=Decimal("11"),
                free_cash_flow=Decimal("11.55"),
                unit="USD",
            )
        ],
    )
    plan = AdaptiveMultistagePlan(
        requested_years=1,
        effective_years=1,
        high_growth_years=1,
        transition_years=0,
        stable_years=0,
        initial_growth_rate=Decimal("5"),
        terminal_growth_rate=Decimal("3"),
        max_annual_growth_fade=Decimal("1"),
        forward_evidence_summary=("guidance",),
    )
    forecast_export = ForecastExportService().export(forecast, adaptive_plan=plan)
    assert forecast_export.observations[0].formula == (
        "revenue × free cash flow margin"
    )
    assert isinstance(forecast_export.adaptive_plan, AdaptiveMultistagePlanExport)
    assert forecast_export.adaptive_plan.forward_evidence_summary == ("guidance",)

    fcff = FcffForecast(
        provider="normalized-sec",
        company_id="1",
        company_name="Example Co",
        base_fiscal_year=2024,
        base_period_end=PERIOD_END,
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
        observations=[
            FcffForecastObservation(
                forecast_year=1,
                fiscal_year=2025,
                period_end=datetime.date(2025, 12, 31),
                revenue_growth=Decimal("5"),
                revenue=Decimal("105"),
                operating_margin=Decimal("20"),
                operating_income=Decimal("21"),
                tax_rate=Decimal("20"),
                nopat=Decimal("16.8"),
                depreciation_to_revenue=Decimal("4"),
                depreciation_and_amortization=Decimal("4.2"),
                capex_to_revenue=Decimal("5"),
                capital_expenditures=Decimal("5.25"),
                operating_working_capital_to_revenue=Decimal("10"),
                operating_working_capital=Decimal("10.5"),
                change_in_operating_working_capital=Decimal(".5"),
                fcff=Decimal("15.25"),
                unit="USD",
            )
        ],
    )
    fcff_export = ForecastExportService().export(fcff)
    assert isinstance(fcff_export.forecast, FcffForecastExport)
    assert fcff_export.forecast.assumption_sources[0][1] == (
        ForecastAssumptionSource.TRAILING_AVERAGE
    )


def _valuation_run() -> ValuationRunResult:
    profile = ValuationProfile(
        provider="normalized-sec",
        company_id="1",
        company_name="Example Co",
        ticker="EX",
        business_archetype=BusinessArchetype.GENERAL_OPERATING,
        lifecycle=CompanyLifecycle.MATURE,
        cyclicality=Cyclicality.LOW,
    )
    suitability = ModelSuitability(
        model=ValuationModel.FCFF_DCF,
        role=ModelRole.PRIMARY,
        suitability_score=90,
        data_readiness=DataReadiness.READY,
    )
    selection = ValuationSelection(profile=profile, models=[suitability])
    result = IntrinsicValuationResult(
        model=ValuationModel.FCFF_DCF,
        adapter="test adapter",
        company_id="1",
        company_name="Example Co",
        ticker="EX",
        valuation_date=datetime.date(2025, 2, 2),
        currency="USD",
        equity_value=Decimal("1000"),
        diluted_shares=Decimal("10"),
        value_per_share=Decimal("100"),
        assumptions=(
            ResolvedModelAssumption(
                name="WACC", value=Decimal("8"), unit="percent", source="explicit"
            ),
        ),
        forecast_summary=(),
        confidence=ValuationConfidence.HIGH,
        warnings=(
            ModelWarning(
                code="test_warning", severity=WarningSeverity.LOW, summary="test"
            ),
        ),
        provenance=(InputProvenance(field="net_debt", source="test"),),
        details=SimpleNamespace(raw_value=Decimal("123"), nested={"values": [1, 2]}),
    )
    return ValuationRunResult(
        economic_profile=profile,
        selection=selection,
        executed_models=(
            ExecutedValuation(
                role=ModelRole.PRIMARY, suitability=suitability, result=result
            ),
        ),
        skipped_models=(
            SkippedValuation(
                model=ValuationModel.EQUITY_DCF,
                role=ModelRole.CROSSCHECK,
                readiness=DataReadiness.BLOCKED,
                missing_inputs=frozenset({"wacc"}),
                reasons=("not ready",),
            ),
        ),
        relative_cross_checks=(
            SimpleNamespace(kind="peer", metadata={"labels": ["core"]}),
        ),
    )


def test_valuation_export_retains_executed_skipped_assumptions_details_and_generic_values():
    exported = ValuationExportService().export(_valuation_run())

    assert exported.executed_models[0].result.assumptions[0].name == "WACC"
    assert exported.executed_models[0].result.provenance[0].field == "net_debt"
    assert exported.executed_models[0].result.details == {
        "nested": {"values": (1, 2)},
        "raw_value": Decimal("123"),
    }
    assert exported.skipped_models[0].missing_inputs == ("wacc",)
    assert exported.relative_cross_checks == (
        {"kind": "peer", "metadata": {"labels": ("core",)}},
    )
    payload = json.loads(exported.model_dump_json())
    assert payload["executed_models"][0]["result"]["details"] == {
        "nested": {"values": [1, 2]},
        "raw_value": "123",
    }

    with pytest.raises(TypeError):
        exported.executed_models[0].result.details["nested"] = {}
    with pytest.raises(TypeError):
        exported.executed_models[0].result.details["nested"]["values"] = ()
    with pytest.raises(TypeError):
        exported.relative_cross_checks[0]["metadata"]["labels"] = ()


def test_forecast_export_rejects_a_forecast_type_payload_mismatch():
    forecast = SimplifiedFcfForecast(
        provider="normalized-sec",
        company_id="1",
        company_name="Example Co",
        base_fiscal_year=2024,
        base_period_end=PERIOD_END,
        base_revenue=Decimal("100"),
        base_free_cash_flow=Decimal("10"),
        unit="USD",
        parameters=SimplifiedFcfForecastParameters(forecast_years=1),
        historical_fiscal_years=(2024,),
        revenue_growth_source=ForecastAssumptionSource.TRAILING_AVERAGE,
        free_cash_flow_margin_source=ForecastAssumptionSource.TRAILING_AVERAGE,
    )
    exported = ForecastExportService().export(forecast)
    payload = exported.model_dump(mode="python")
    payload["forecast_type"] = ForecastKind.FCFF

    with pytest.raises(ValueError, match="forecast_type fcff requires"):
        ForecastExport.model_validate(payload)


def test_red_flags_export_keeps_evidence_source_observations_and_composes_sections():
    source_observation = RedFlagSourceObservation(
        concept=FinancialConcept.REVENUE,
        value=Decimal("100"),
        unit="USD",
        granularity=Granularity.ANNUAL,
        fiscal_year=2024,
        fiscal_period=FiscalPeriod.FY,
        period_end=PERIOD_END,
        provider="sec",
        source_concept="Revenues",
    )
    report = RedFlagsReport(
        provider="normalized-sec",
        company_id="1",
        company_name="Example Co",
        ticker="EX",
        granularity=Granularity.ANNUAL,
        configuration_name="default",
        flags=(
            RedFlag(
                code="test_flag",
                category=RedFlagCategory.DEBT,
                severity=RedFlagSeverity.HIGH,
                message="Debt is high",
                evidence=(
                    RedFlagEvidence(
                        metric="net_debt_to_ebitda",
                        value=Decimal("4"),
                        unit="x",
                        threshold=Decimal("2"),
                        threshold_unit="x",
                        comparison=">",
                        formula="net debt / EBITDA",
                        fiscal_year=2024,
                        fiscal_period=FiscalPeriod.FY,
                        period_end=PERIOD_END,
                        granularity=Granularity.ANNUAL,
                        input_concepts=(FinancialConcept.REVENUE,),
                        source_observations=(source_observation,),
                    ),
                ),
            ),
        ),
        warnings=(
            RedFlagWarning(
                code="warning",
                message="Some inputs were unavailable",
                required_concepts=(FinancialConcept.CASH_AND_EQUIVALENTS,),
            ),
        ),
    )
    red_flags = RedFlagsExportService().export(report)
    composed = CompanyAnalysisReportService().compose(
        financials=_financials(), red_flags=report
    )

    assert red_flags.flags[0].evidence[0].source_observations[0].source_concept == (
        "Revenues"
    )
    assert red_flags.warnings[0].required_concepts == (
        FinancialConcept.CASH_AND_EQUIVALENTS,
    )
    assert red_flags.as_of == PERIOD_END
    assert composed.section == ExportSection.COMPANY_ANALYSIS
    assert composed.financial_data is not None
    assert composed.red_flags is not None
    assert composed.model_dump_json()


def test_company_analysis_composition_rejects_mismatched_company_ids():
    metrics = CompanyMetrics(
        provider="normalized-sec",
        company_id="2",
        company_name="Another Co",
        observations=(),
    )

    with pytest.raises(ValueError, match="share company_id"):
        CompanyAnalysisReportService().compose(
            financials=_financials(), metrics=metrics
        )


def test_company_analysis_rejects_conflicting_names_and_tickers():
    with pytest.raises(ValueError, match="company_name"):
        CompanyAnalysisReportService().compose(
            financials=_financials(),
            metrics=CompanyMetrics(
                provider="yahoo",
                company_id="1",
                company_name="Another Co",
                ticker="EX",
                observations=(),
            ),
        )

    with pytest.raises(ValueError, match="ticker"):
        CompanyAnalysisReportService().compose(
            financials=_financials(),
            metrics=CompanyMetrics(
                provider="yahoo",
                company_id="1",
                company_name="Example Co",
                ticker="OTHER",
                observations=(),
            ),
        )


def test_company_analysis_allows_mixed_providers_with_shared_ticker():
    composed = CompanyAnalysisReportService().compose(
        financials=_financials().model_copy(update={"provider": "sec"}),
        metrics=CompanyMetrics(
            provider="yahoo",
            company_id="1",
            company_name="Example Co",
            ticker="EX",
            observations=(),
        ),
    )

    assert composed.provider == "mixed"
    assert composed.ticker == "EX"


def test_company_analysis_rejects_ambiguous_mixed_providers():
    with pytest.raises(ValueError, match="shared reliable identity"):
        CompanyAnalysisReportService().compose(
            financials=_financials().model_copy(
                update={"provider": "sec", "ticker": None}
            ),
            metrics=CompanyMetrics(
                provider="yahoo",
                company_id="1",
                company_name="Example Co",
                observations=(),
            ),
        )
