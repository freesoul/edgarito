"""Focused end-to-end coverage for explicit driver-based FCFF activation."""

import asyncio
import datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from edgarito.cli.use_cases import forecast as forecast_cli
from edgarito.cli.use_cases.financial_retrieval import merge_normalized_financials
from edgarito.config.valuation import ForecastMethod
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.forecasting import (
    DriverBasedFcffForecastResult,
    FcffForecastParameters,
)
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.operating import (
    OperatingArchetype,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
)
from edgarito.services.financials.availability import ObservationAvailabilityMode
from edgarito.services.forecasting import (
    DriverBasedFcffForecastService,
    DriverBasedForecastReadiness,
    FcffForecastOrchestrationService,
    build_forecast_context,
)
from edgarito.services.valuation import (
    FcffDcfCapitalBridge,
    FcffDcfParameters,
    FcffDcfService,
)

D = Decimal


def _financials() -> NormalizedCompanyFinancials:
    values = {
        2023: {
            "revenue": "205",
            "operating_income": "45",
            "pretax_income": "40",
            "income_tax_expense": "8",
            "depreciation_and_amortization": "10",
            "capital_expenditures": "12",
            "accounts_receivable": "30",
            "inventory": "20",
            "prepaid_and_other_current_assets": "10",
            "accounts_payable": "15",
            "accrued_liabilities": "10",
            "deferred_revenue_current": "5",
        },
        2024: {
            "revenue": "240",
            "operating_income": "60",
            "pretax_income": "50",
            "income_tax_expense": "10",
            "depreciation_and_amortization": "12",
            "capital_expenditures": "15",
            "accounts_receivable": "36",
            "inventory": "24",
            "prepaid_and_other_current_assets": "12",
            "accounts_payable": "18",
            "accrued_liabilities": "12",
            "deferred_revenue_current": "6",
        },
    }
    concepts = {item.value: item for item in FinancialConcept}
    return NormalizedCompanyFinancials(
        provider="fixture",
        company_id="fixture-company",
        company_name="Fixture Company",
        ticker="FIX",
        observations=[
            FinancialObservation(
                concept=concepts[concept],
                statement=concepts[concept].statement,
                value=D(value),
                unit="USD",
                granularity=Granularity.ANNUAL,
                fiscal_year=year,
                fiscal_period=FiscalPeriod.FY,
                period_end=datetime.date(year, 12, 31),
                provider="fixture",
                taxonomy="fixture",
                source_concept=concept,
            )
            for year, year_values in values.items()
            for concept, value in year_values.items()
        ],
    )


def _fixture():
    segments = (
        OperatingSegment(segment_id="cloud", name="Cloud", currency="USD"),
        OperatingSegment(segment_id="hardware", name="Hardware", currency="USD"),
    )
    definitions = tuple(
        OperatingDriverDefinition(
            driver_id=f"{segment_id}-revenue",
            archetype=OperatingArchetype.VOLUME_PRICE,
            segment_id=segment_id,
            output_metric="revenue",
            input_metrics=("volume", "price"),
            units={"volume": "units", "price": "USD/unit"},
            formula_id="volume_price",
            required_inputs=("volume", "price"),
        )
        for segment_id in ("cloud", "hardware")
    )
    observations = []
    paths = {
        2024: {"cloud": ("60", "2"), "hardware": ("40", "3")},
        2025: {"cloud": ("70", "2.2"), "hardware": ("40", "3.2")},
        2026: {"cloud": ("80", "2.4"), "hardware": ("42", "3.3")},
    }
    for year, segment_paths in paths.items():
        for segment_id, (volume, price) in segment_paths.items():
            observations.extend(
                (
                    OperatingDriverObservation(
                        segment_id=segment_id,
                        driver_id="volume",
                        fiscal_year=year,
                        value=D(volume),
                        unit="units",
                        origin="first_party_observation",
                        confidence="high",
                    ),
                    OperatingDriverObservation(
                        segment_id=segment_id,
                        driver_id="price",
                        fiscal_year=year,
                        value=D(price),
                        unit="USD/unit",
                        origin="first_party_observation",
                        confidence="high",
                    ),
                    OperatingDriverObservation(
                        segment_id=segment_id,
                        driver_id="gross_margin",
                        fiscal_year=year,
                        value=D("70" if segment_id == "cloud" else "50"),
                        unit="percent",
                        scope="segment",
                        origin="first_party_observation",
                        confidence="high",
                    ),
                )
            )
    company_paths = {
        2025: {
            "r_and_d": "20",
            "sg_and_a": "30",
            "other_operating_items": "0",
            "tax_rate": "20",
            "depreciation_and_amortization": "14",
            "capital_expenditures": "18",
            "operating_working_capital": "42",
        },
        2026: {
            "r_and_d": "21",
            "sg_and_a": "31",
            "other_operating_items": "0",
            "tax_rate": "20",
            "depreciation_and_amortization": "16",
            "capital_expenditures": "20",
            "operating_working_capital": "45",
        },
    }
    for year, metrics in company_paths.items():
        for metric, value in metrics.items():
            observations.append(
                OperatingDriverObservation(
                    segment_id="company",
                    driver_id=metric,
                    fiscal_year=year,
                    value=D(value),
                    unit="percent" if metric == "tax_rate" else "USD",
                    scope="company",
                    is_total=True,
                    origin="first_party_observation",
                    confidence="high",
                )
            )
    return segments, definitions, tuple(observations)


def _financials_with_normalized_operating_history():
    financials = _financials()
    concepts = {
        item.value: item for item in FinancialConcept
    }
    extras = [
        FinancialObservation(
            concept=concepts[concept],
            statement=concepts[concept].statement,
            value=D(value),
            unit="USD",
            granularity=Granularity.ANNUAL,
            fiscal_year=year,
            fiscal_period=FiscalPeriod.FY,
            period_end=datetime.date(year, 12, 31),
            provider="fixture",
            taxonomy="fixture",
            source_concept=concept,
        )
        for year, values in {
            2023: {
                "gross_profit": "125",
                "research_and_development_expense": "15",
                "selling_general_and_administrative_expense": "25",
            },
            2024: {
                "gross_profit": "145",
                "research_and_development_expense": "18",
                "selling_general_and_administrative_expense": "27",
            },
        }.items()
        for concept, value in values.items()
    ]
    return financials.model_copy(
        update={"observations": [*financials.observations, *extras]}
    )


def _driver_result():
    segments, definitions, observations = _fixture()
    return DriverBasedFcffForecastService().forecast(
        _financials(),
        FcffForecastParameters(forecast_years=2),
        segments=segments,
        definitions=definitions,
        observations=observations,
    )


def test_frozen_multisegment_golden_fixture_maps_every_year_and_audit():
    result = _driver_result()
    forecast = result.forecast

    assert result.readiness.ready
    assert [item.revenue for item in forecast.observations] == [D("282"), D("330.6")]
    assert [item.operating_income for item in forecast.observations] == [
        D("121.8"),
        D("151.7"),
    ]
    assert [item.fcff for item in forecast.observations] == [D("87.44"), D("114.36")]
    assert forecast.base_operating_working_capital == D("36")
    assert forecast.fiscal_year_end == datetime.date(2025, 12, 31)
    assert forecast.operating_source_by_year == {
        2025: "independent_operating",
        2026: "independent_operating",
    }
    assert forecast.observations[0].cell_audits["fcff"].source == "derived"
    assert "nopat_plus_da_minus_capex_minus_delta_owc" in forecast.observations[0].cell_audits["fcff"].method


def test_driver_result_is_immutable_and_round_trips():
    result = _driver_result()
    with pytest.raises((TypeError, ValueError)):
        result.readiness = DriverBasedForecastReadiness()
    assert DriverBasedFcffForecastResult.model_validate_json(result.model_dump_json()) == result


def test_shared_context_does_not_require_future_driver_inference():
    context = build_forecast_context(_financials(), FcffForecastParameters(forecast_years=2))
    assert context.context.seed_type.value == "FY"
    assert context.context.base.operating_working_capital == D("36")


def test_elapsed_incomplete_fy_uses_one_as_of_for_ytd_then_ttm_context():
    from test_generic_valuation_core import _forecast_financials

    service = DriverBasedFcffForecastService()
    parameters = FcffForecastParameters(forecast_years=2)
    ytd = service.build_context(
        _forecast_financials(),
        parameters,
        as_of=datetime.date(2026, 8, 30),
        availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
    ).context
    elapsed = service.build_context(
        _forecast_financials(),
        parameters,
        as_of=datetime.date(2027, 1, 1),
        availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
    ).context
    assert ytd.seed_type.value == "YTD+forecast"
    assert ytd.current_fiscal_year == 2026
    assert elapsed.seed_type.value == "TTM"
    assert elapsed.current_fiscal_year == 2027


@pytest.mark.parametrize("metric", ["gross_margin", "tax_rate", "capital_expenditures", "operating_working_capital"])
def test_missing_driver_layer_can_use_audited_normalized_automatic_assumption(metric):
    segments, definitions, observations = _fixture()
    observations = tuple(
        item
        for item in observations
        if not (item.driver_id == metric and item.fiscal_year == 2025)
    )
    result = DriverBasedFcffForecastService().forecast(
        _financials(),
        FcffForecastParameters(forecast_years=2),
        segments=segments,
        definitions=definitions,
        observations=observations,
    )
    assert result.readiness.ready
    assert any(metric in item for item in result.readiness.warnings)


def test_readiness_missing_map_is_blocking_and_round_trips_canonically():
    readiness = DriverBasedForecastReadiness(
        target_years=(2025,),
        missing_metrics_by_year={2025: ("gross_profit",)},
        missing_metric_years=(),
    )
    assert not readiness.ready
    assert readiness.missing_metric_years == ("FY2025 gross_profit",)
    restored = DriverBasedForecastReadiness.model_validate_json(
        readiness.model_dump_json()
    )
    assert restored == readiness


def test_forecast_cli_driver_fetches_quarters_and_never_calls_normalized_forecast():
    segments, definitions, observations = _fixture()
    evidence = {
        "segments": segments,
        "definitions": definitions,
        "observations": observations,
    }
    configured = SimpleNamespace(
        forecast_years=2,
        revenue_growth=None,
        operating_margin=None,
        tax_rate=None,
        depreciation_to_revenue=None,
        capex_to_revenue=None,
        operating_working_capital_to_revenue=None,
        revenue_anchors={},
        assumption_source_overrides={},
        historical_window=3,
    )
    profile = SimpleNamespace(
        forecast=SimpleNamespace(
            default_method=ForecastMethod.FCFF,
            fcff=configured,
            simplified=SimpleNamespace(
                forecast_years=2,
                revenue_growth=None,
                free_cash_flow_margin=None,
                historical_window=3,
            ),
        )
    )
    calls = []
    driver_calls = []
    merge_calls = []

    class NormalizedService:
        def required_concepts(self):
            return DriverBasedFcffForecastService.required_concepts()

        def forecast(self, *_args, **_kwargs):
            raise AssertionError("normalized FCFF must not run for explicit driver CLI")

    class RecordingDriverService:
        @classmethod
        def required_concepts(cls):
            return DriverBasedFcffForecastService.required_concepts()

        def __init__(self):
            self.delegate = DriverBasedFcffForecastService()

        def build_context(self, *args, **kwargs):
            driver_calls.append(("context", kwargs["as_of"], kwargs["availability_mode"]))
            return self.delegate.build_context(*args, **kwargs)

        def forecast(self, *args, **kwargs):
            driver_calls.append(("forecast", kwargs["as_of"], kwargs["availability_mode"]))
            return self.delegate.forecast(*args, **kwargs)

    def merge(annual, quarterly, *, as_of, availability_mode):
        merge_calls.append((as_of, availability_mode))
        return merge_normalized_financials(
            annual,
            quarterly,
            as_of=as_of,
            availability_mode=availability_mode,
        )

    async def retrieve(_args, granularity, _concepts):
        calls.append(granularity)
        return _financials()

    class Presenter:
        def render(self, value):
            assert value.method == "driver_based_fcff"
            return "driver forecast"

    args = SimpleNamespace(
        ticker="FIX",
        profile=None,
        forecast_method=None,
        fcff_forecast_method="driver_based",
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
    context = {
        "_load_selected_valuation_profile": lambda _args: profile,
        "_fcff_parameters": lambda _args, _configured: FcffForecastParameters(
            forecast_years=2
        ),
        "FcffForecastService": NormalizedService,
        "_retrieve_financials": retrieve,
        "DriverBasedFcffForecastService": RecordingDriverService,
        "_merge_normalized_financials": merge,
        "OPERATING_EVIDENCE": evidence,
        "ForecastConsolePresenter": Presenter,
    }
    assert asyncio.run(forecast_cli.run_forecast(args, context=context)) == 0
    assert calls == [Granularity.ANNUAL, Granularity.QUARTERLY]
    expected_as_of = datetime.date.today()
    assert merge_calls == [(expected_as_of, ObservationAvailabilityMode.CURRENT_SNAPSHOT)]
    assert driver_calls == [
        ("context", expected_as_of, ObservationAvailabilityMode.CURRENT_SNAPSHOT),
        ("forecast", expected_as_of, ObservationAvailabilityMode.CURRENT_SNAPSHOT),
    ]


def test_orchestration_routes_explicit_driver_and_existing_dcf_accepts_it():
    segments, definitions, observations = _fixture()
    result = FcffForecastOrchestrationService().forecast(
        _financials(),
        FcffForecastParameters(forecast_years=2),
        method="driver_based",
        evidence={
            "segments": segments,
            "definitions": definitions,
            "observations": observations,
        },
    )
    assert result.requested_method.value == "driver_based"
    assert result.resolved_method.value == "driver_based"
    dcf = FcffDcfService().value(
        result.forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        FcffDcfCapitalBridge(
            fiscal_year=2025,
            period_end=datetime.date(2025, 12, 31),
            unit="USD",
            net_debt=D("100"),
            diluted_shares=D("10"),
            net_debt_source="fixture",
            shares_source="fixture",
        ),
    )
    assert dcf.enterprise_value == D("1379.036363636363636363636364")


def test_driver_allows_normalized_component_assumptions_without_legacy_fcff():
    class LegacyNormalizedService:
        def forecast(self, *_args, **_kwargs):
            raise AssertionError("legacy consolidated FCFF must not be invoked")

    result = FcffForecastOrchestrationService(
        fcff_service=LegacyNormalizedService()
    ).forecast(
        _financials_with_normalized_operating_history(),
        FcffForecastParameters(forecast_years=2, revenue_growth=(D("10"), D("5"))),
        method="driver_based",
    )
    assert result.requested_method.value == "driver_based"
    assert result.resolved_method.value == "driver_based"
    assert result.driver_readiness.ready
    assert any("normalized historical" in item for item in result.warnings)
