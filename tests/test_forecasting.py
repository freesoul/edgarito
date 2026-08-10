import datetime
import json
from decimal import Decimal
from pathlib import Path

import pytest

from edgarito.cli import main
from edgarito.config.valuation import MultistageValuationConfiguration
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.forecasting import (
    AdaptiveMultistageFcffForecastService,
    FcffForecast,
    FcffForecastDriver,
    FcffForecastParameters,
    FcffForecastService,
    ForecastAssumptionSource,
    ForecastSeedType,
    ForecastValue,
    ForwardGrowthEvidence,
    ForwardGrowthOutlook,
    FreeCashFlowForecastService,
    MonetaryForecastConstraint,
    SimplifiedFcfForecastParameters,
    SimplifiedFcfForecastService,
)
from edgarito.services.metrics import FinancialMetric, FinancialMetricsService
from edgarito.services.valuation import (
    FcffDcfCapitalBridge,
    FcffDcfParameters,
    FcffDcfService,
)

FIXTURE = Path(__file__).parent / "fixtures" / "aapl_facts.json"


def _observation(
    concept: FinancialConcept,
    value: str,
    fiscal_year: int,
    *,
    granularity: Granularity = Granularity.ANNUAL,
    fiscal_period: FiscalPeriod = FiscalPeriod.FY,
    period_end: datetime.date | None = None,
) -> FinancialObservation:
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal(value),
        unit="USD",
        granularity=granularity,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        period_end=period_end or datetime.date(fiscal_year, 12, 31),
        provider="test",
        taxonomy="test",
        source_concept=concept.value,
    )


def _financials() -> NormalizedCompanyFinancials:
    values = {
        2023: {
            FinancialConcept.REVENUE: "100",
            FinancialConcept.OPERATING_CASH_FLOW: "15",
            FinancialConcept.CAPITAL_EXPENDITURES: "5",
        },
        2024: {
            FinancialConcept.REVENUE: "120",
            FinancialConcept.OPERATING_CASH_FLOW: "18",
            FinancialConcept.CAPITAL_EXPENDITURES: "6",
        },
    }
    return NormalizedCompanyFinancials(
        provider="test",
        company_id="0000000001",
        company_name="Test Company",
        ticker="TEST",
        observations=[
            _observation(concept, value, fiscal_year)
            for fiscal_year, period_values in values.items()
            for concept, value in period_values.items()
        ],
    )


def test_forecasts_fcf_with_constant_explicit_assumptions():
    parameters = SimplifiedFcfForecastParameters(
        forecast_years=2,
        revenue_growth=Decimal("10"),
        free_cash_flow_margin=Decimal("12.5"),
    )

    forecast = SimplifiedFcfForecastService().forecast(_financials(), parameters)

    assert forecast.base_fiscal_year == 2024
    assert forecast.base_revenue == Decimal("120")
    assert forecast.base_free_cash_flow == Decimal("12")
    assert forecast.historical_fiscal_years == (2023, 2024)
    assert forecast.revenue_growth_source == ForecastAssumptionSource.EXPLICIT
    assert forecast.free_cash_flow_margin_source == ForecastAssumptionSource.EXPLICIT
    assert [observation.revenue for observation in forecast.observations] == [
        Decimal("132.0"),
        Decimal("145.20"),
    ]
    assert [observation.free_cash_flow for observation in forecast.observations] == [
        Decimal("16.500"),
        Decimal("18.1500"),
    ]


def test_forecasts_with_year_specific_paths():
    parameters = SimplifiedFcfForecastParameters(
        forecast_years=2,
        revenue_growth=(Decimal("10"), Decimal("5")),
        free_cash_flow_margin=(Decimal("10"), Decimal("12")),
    )

    forecast = SimplifiedFcfForecastService().forecast(_financials(), parameters)

    assert [observation.revenue_growth for observation in forecast.observations] == [
        Decimal("10"),
        Decimal("5"),
    ]
    assert [
        observation.free_cash_flow_margin for observation in forecast.observations
    ] == [Decimal("10"), Decimal("12")]
    assert forecast.observations[-1].free_cash_flow == Decimal("16.6320")


def test_infers_trailing_average_growth_and_fcf_margin():
    parameters = SimplifiedFcfForecastParameters(forecast_years=1)

    forecast = SimplifiedFcfForecastService().forecast(_financials(), parameters)

    assert forecast.revenue_growth_source == ForecastAssumptionSource.TRAILING_AVERAGE
    assert (
        forecast.free_cash_flow_margin_source
        == ForecastAssumptionSource.TRAILING_AVERAGE
    )
    observation = forecast.observations[0]
    assert observation.revenue_growth == Decimal("20.0")
    assert observation.revenue == Decimal("144.00")
    assert observation.free_cash_flow_margin == Decimal("10.0")
    assert observation.free_cash_flow == Decimal("14.400")


def test_forecast_value_is_validated_and_frozen():
    value = ForecastValue(
        value="12.5",
        source="explicit",
        method="explicit forecast driver",
        confidence="HIGH",
    )

    assert value.value == Decimal("12.5")
    assert value.confidence == "high"
    with pytest.raises(ValueError, match="finite"):
        ForecastValue(
            value="NaN",
            source="explicit",
            method="test",
            confidence="high",
        )
    with pytest.raises(ValueError, match="high, medium, or low"):
        ForecastValue(
            value="1",
            source="explicit",
            method="test",
            confidence="certain",
        )
    with pytest.raises(ValueError):
        value.value = Decimal("13")


def test_fcff_absolute_revenue_anchor_replaces_historical_growth_for_that_year():
    parameters = FcffForecastParameters(
        forecast_years=1,
        revenue_growth=Decimal("5"),
        operating_margin=Decimal("25"),
        tax_rate=Decimal("20"),
        depreciation_to_revenue=Decimal("4"),
        capex_to_revenue=Decimal("6"),
        operating_working_capital_to_revenue=Decimal("15"),
        revenue_anchors={2025: Decimal("125")},
        assumption_source_overrides={
            FcffForecastDriver.REVENUE_GROWTH: (
                ForecastAssumptionSource.MANAGEMENT_GUIDANCE
            )
        },
    )

    forecast = FcffForecastService().forecast(_fcff_financials(), parameters)

    assert forecast.observations[0].revenue == Decimal("125")
    assert forecast.observations[0].revenue_growth == Decimal(
        "4.166666666666666666666666700"
    )
    assert (
        forecast.assumption_sources[FcffForecastDriver.REVENUE_GROWTH]
        == ForecastAssumptionSource.MANAGEMENT_GUIDANCE
    )


def test_revenue_anchor_audit_preserves_management_guidance_source():
    parameters = FcffForecastParameters(
        forecast_years=1,
        revenue_growth=Decimal("5"),
        operating_margin=Decimal("25"),
        tax_rate=Decimal("20"),
        depreciation_to_revenue=Decimal("4"),
        capex_to_revenue=Decimal("6"),
        operating_working_capital_to_revenue=Decimal("15"),
        revenue_anchors={2025: Decimal("125")},
        revenue_anchor_sources={2025: ForecastAssumptionSource.MANAGEMENT_GUIDANCE},
    )

    forecast = FcffForecastService().forecast(_fcff_financials(), parameters)
    revenue_audit = forecast.observations[0].cell_audits["revenue"]

    assert "revenue_anchor=management_guidance" in revenue_audit.source
    assert "management guidance revenue anchor" in revenue_audit.method
    assert revenue_audit.confidence == "high"


def test_parameters_reject_an_incomplete_year_specific_path():
    with pytest.raises(ValueError, match="must contain one value or 3 values"):
        SimplifiedFcfForecastParameters(
            forecast_years=3,
            revenue_growth=(Decimal("5"), Decimal("4")),
        )


def test_forecast_reports_missing_annual_inputs():
    financials = _financials().model_copy(
        update={
            "observations": [
                observation
                for observation in _financials().observations
                if observation.concept != FinancialConcept.CAPITAL_EXPENDITURES
            ]
        }
    )

    with pytest.raises(ValueError, match="requires annual revenue"):
        SimplifiedFcfForecastService().forecast(financials)


def _fcff_financials() -> NormalizedCompanyFinancials:
    financials = _financials()
    values = {
        2023: {
            FinancialConcept.OPERATING_INCOME: "20",
            FinancialConcept.PRETAX_INCOME: "18",
            FinancialConcept.INCOME_TAX_EXPENSE: "3.6",
            FinancialConcept.DEPRECIATION_AND_AMORTIZATION: "4",
            FinancialConcept.ACCOUNTS_RECEIVABLE: "15",
            FinancialConcept.INVENTORY: "10",
            FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: "5",
            FinancialConcept.ACCOUNTS_PAYABLE: "8",
            FinancialConcept.ACCRUED_LIABILITIES: "4",
            FinancialConcept.DEFERRED_REVENUE_CURRENT: "2",
        },
        2024: {
            FinancialConcept.OPERATING_INCOME: "30",
            FinancialConcept.PRETAX_INCOME: "24",
            FinancialConcept.INCOME_TAX_EXPENSE: "4.8",
            FinancialConcept.DEPRECIATION_AND_AMORTIZATION: "5",
            FinancialConcept.ACCOUNTS_RECEIVABLE: "18",
            FinancialConcept.INVENTORY: "12",
            FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: "6",
            FinancialConcept.ACCOUNTS_PAYABLE: "9",
            FinancialConcept.ACCRUED_LIABILITIES: "5",
            FinancialConcept.DEFERRED_REVENUE_CURRENT: "2",
        },
    }
    return financials.model_copy(
        update={
            "observations": [
                *financials.observations,
                *[
                    _observation(concept, value, fiscal_year)
                    for fiscal_year, period_values in values.items()
                    for concept, value in period_values.items()
                ],
            ]
        }
    )


def test_driver_based_fcff_forecasts_the_full_operating_bridge():
    parameters = FcffForecastParameters(
        forecast_years=2,
        revenue_growth=(Decimal("10"), Decimal("5")),
        operating_margin=Decimal("25"),
        tax_rate=Decimal("20"),
        depreciation_to_revenue=Decimal("4"),
        capex_to_revenue=Decimal("6"),
        operating_working_capital_to_revenue=Decimal("15"),
    )

    forecast = FcffForecastService().forecast(_fcff_financials(), parameters)

    assert forecast.method == "driver_based_fcff"
    assert forecast.base_operating_working_capital == Decimal("20")
    assert forecast.base_tax_rate == Decimal("20.0")
    assert forecast.base_nopat == Decimal("24.0")
    assert forecast.base_fcff == Decimal("19.0")
    first, second = forecast.observations
    assert first.revenue == Decimal("132.0")
    assert first.operating_income == Decimal("33.000")
    assert first.nopat == Decimal("26.4000")
    assert first.depreciation_and_amortization == Decimal("5.280")
    assert first.capital_expenditures == Decimal("7.920")
    assert first.change_in_operating_working_capital == Decimal("-0.20")
    assert first.fcff == Decimal("23.9600")
    assert second.fcff == Decimal("23.95800")
    assert first.formula.startswith("NOPAT + depreciation")
    expected_cells = {
        "revenue_growth",
        "revenue",
        "operating_margin",
        "operating_income",
        "tax_rate",
        "nopat",
        "depreciation_and_amortization",
        "capital_expenditures",
        "change_in_operating_working_capital",
        "fcff",
    }
    assert set(first.cell_audits) == expected_cells
    direct_driver_cells = {
        "revenue_growth",
        "operating_margin",
        "tax_rate",
    }
    for key in expected_cells:
        audit = first.cell_audits[key]
        assert audit.value == getattr(first, key)
        assert audit.confidence == "high"
        if key in direct_driver_cells:
            assert audit.source == "explicit"
        else:
            assert audit.source.startswith("derived[")
    assert "operating_margin=explicit" in first.cell_audits["operating_income"].source
    assert "revenue × operating margin" in first.cell_audits["operating_income"].method
    assert "NOPAT + depreciation" in first.cell_audits["fcff"].method
    assert FcffForecast.model_validate_json(forecast.model_dump_json()) == forecast


@pytest.mark.parametrize(
    ("constraint", "expected_capex"),
    [
        (MonetaryForecastConstraint(point=Decimal("15")), Decimal("15")),
        (MonetaryForecastConstraint(minimum=Decimal("10")), Decimal("10")),
        (MonetaryForecastConstraint(maximum=Decimal("6")), Decimal("6")),
        (
            MonetaryForecastConstraint(
                point=Decimal("9"), minimum=Decimal("8"), maximum=Decimal("10")
            ),
            Decimal("9"),
        ),
        (
            MonetaryForecastConstraint(
                point=Decimal("0"), minimum=Decimal("0"), maximum=Decimal("1")
            ),
            Decimal("0"),
        ),
    ],
)
def test_absolute_capex_constraints_recalculate_ratio_and_preserve_fcff_identity(
    constraint, expected_capex
):
    parameters = FcffForecastParameters(
        forecast_years=1,
        revenue_growth=Decimal("10"),
        operating_margin=Decimal("25"),
        tax_rate=Decimal("20"),
        depreciation_to_revenue=Decimal("4"),
        operating_working_capital_to_revenue=Decimal("15"),
        capex_constraints={2025: constraint},
    )

    forecast = FcffForecastService().forecast(_fcff_financials(), parameters)
    observation = forecast.observations[0]

    assert observation.capital_expenditures == expected_capex
    assert observation.capex_to_revenue == expected_capex / observation.revenue * 100
    assert forecast.capex_constraints_applied == (2025,)
    assert (
        forecast.assumption_sources[FcffForecastDriver.CAPEX_TO_REVENUE]
        == ForecastAssumptionSource.MANAGEMENT_GUIDANCE
    )
    assert (
        "management_guidance" in observation.cell_audits["capital_expenditures"].source
    )
    assert "capex constraint" in observation.cell_audits["capital_expenditures"].method
    assert "management_guidance" in observation.cell_audits["fcff"].source
    assert not FcffForecastService().economic_identity_issues(forecast)


def test_ytd_absolute_capex_constraint_targets_annual_total_and_recalculates_remainder_ratio():
    annual_values = {
        FinancialConcept.REVENUE: "100",
        FinancialConcept.OPERATING_INCOME: "20",
        FinancialConcept.PRETAX_INCOME: "18",
        FinancialConcept.INCOME_TAX_EXPENSE: "4.5",
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION: "4",
        FinancialConcept.CAPITAL_EXPENDITURES: "8",
        FinancialConcept.ACCOUNTS_RECEIVABLE: "15",
        FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: "5",
        FinancialConcept.ACCOUNTS_PAYABLE: "8",
        FinancialConcept.ACCRUED_LIABILITIES: "4",
        FinancialConcept.DEFERRED_REVENUE_CURRENT: "3",
    }
    quarterly_values = {
        FinancialConcept.REVENUE: "25",
        FinancialConcept.OPERATING_INCOME: "5",
        FinancialConcept.PRETAX_INCOME: "4.5",
        FinancialConcept.INCOME_TAX_EXPENSE: "1.125",
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION: "1",
        FinancialConcept.CAPITAL_EXPENDITURES: "2",
        FinancialConcept.ACCOUNTS_RECEIVABLE: "8",
        FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: "3",
        FinancialConcept.ACCOUNTS_PAYABLE: "4",
        FinancialConcept.ACCRUED_LIABILITIES: "2",
        FinancialConcept.DEFERRED_REVENUE_CURRENT: "1",
    }
    observations = [
        _observation(concept, value, 2024) for concept, value in annual_values.items()
    ]
    for quarter, period_end in (
        (FiscalPeriod.Q1, datetime.date(2025, 3, 31)),
        (FiscalPeriod.Q2, datetime.date(2025, 6, 30)),
    ):
        observations.extend(
            _observation(
                concept,
                value,
                2025,
                granularity=Granularity.QUARTERLY,
                fiscal_period=quarter,
                period_end=period_end,
            )
            for concept, value in quarterly_values.items()
        )
    financials = NormalizedCompanyFinancials(
        provider="test",
        company_id="ytd",
        company_name="YTD Test",
        observations=observations,
    )
    parameters = FcffForecastParameters(
        forecast_years=1,
        revenue_growth=Decimal("10"),
        operating_margin=Decimal("20"),
        tax_rate=Decimal("25"),
        depreciation_to_revenue=Decimal("4"),
        operating_working_capital_to_revenue=Decimal("5"),
        capex_constraints={
            2025: MonetaryForecastConstraint(point=Decimal("30")),
        },
    )

    forecast = FcffForecastService().forecast(financials, parameters)
    observation = forecast.observations[0]

    assert forecast.seed_type.value == "YTD+forecast"
    assert observation.revenue == Decimal("110")
    assert observation.capital_expenditures == Decimal("30")
    assert observation.capex_to_revenue == Decimal("27.27272727272727272727272727")
    assert forecast.ytd_anchor is not None
    assert forecast.ytd_anchor.capex_to_revenue == Decimal(
        "43.33333333333333333333333333"
    )
    assert (
        "management_guidance" in observation.cell_audits["capital_expenditures"].source
    )
    assert not FcffForecastService().economic_identity_issues(forecast)


def test_driver_based_fcff_extends_a_short_driver_path_with_its_final_value():
    parameters = FcffForecastParameters(
        forecast_years=4,
        revenue_growth=Decimal("5"),
        operating_margin=(Decimal("25"), Decimal("27")),
        tax_rate=Decimal("20"),
        depreciation_to_revenue=Decimal("4"),
        capex_to_revenue=Decimal("6"),
        operating_working_capital_to_revenue=Decimal("15"),
    )

    forecast = FcffForecastService().forecast(_fcff_financials(), parameters)

    assert [item.operating_margin for item in forecast.observations] == [
        Decimal("25"),
        Decimal("27"),
        Decimal("27"),
        Decimal("27"),
    ]


def test_fcff_infers_each_omitted_driver_from_trailing_history():
    forecast = FcffForecastService().forecast(
        _fcff_financials(), FcffForecastParameters(forecast_years=1)
    )

    assert set(forecast.assumption_sources) == set(FcffForecastDriver)
    assert set(forecast.assumption_sources.values()) == {
        ForecastAssumptionSource.TRAILING_AVERAGE
    }
    observation = forecast.observations[0]
    assert observation.revenue_growth == Decimal("20.0")
    assert observation.operating_margin == Decimal("22.500")
    assert observation.tax_rate == Decimal("20.0")
    assert observation.fcff == (
        observation.nopat
        + observation.depreciation_and_amortization
        - observation.capital_expenditures
        - observation.change_in_operating_working_capital
    )
    assert observation.cell_audits["revenue_growth"].source == "trailing_average"
    assert observation.cell_audits["revenue_growth"].confidence == "medium"
    assert observation.cell_audits["revenue"].source.startswith("derived[")
    assert observation.cell_audits["fcff"].confidence == "medium"


def test_adaptive_multistage_projection_is_invariant_after_stable_stage():
    financials = _fcff_financials()
    base_service = FcffForecastService()
    adaptive = AdaptiveMultistageFcffForecastService(base_service)
    configuration = MultistageValuationConfiguration(
        terminal_return_on_invested_capital=Decimal("15")
    )

    forecasts = []
    plans = []
    for requested_years in (5, 10):
        parameters = FcffForecastParameters(forecast_years=requested_years)
        seed = base_service.forecast(financials, parameters)
        forecast, plan = adaptive.forecast(
            financials,
            seed,
            parameters,
            Decimal("3"),
            configuration,
            normalized_tax_rate=Decimal("25"),
        )
        forecasts.append(forecast)
        plans.append(plan)

    assert plans[0].requested_years == 5
    assert plans[0].effective_years == 9
    assert plans[0].extended_to_stable
    assert plans[0].high_growth_years == 2
    assert plans[0].transition_years == 6
    assert plans[0].stable_years == 1
    assert plans[1].effective_years == 10
    assert plans[1].stable_years == 2
    assert [item.revenue_growth for item in forecasts[0].observations] == [
        item.revenue_growth for item in forecasts[1].observations[:9]
    ]
    terminal = forecasts[0].observations[-1]
    terminal_net_reinvestment = (
        terminal.capital_expenditures
        - terminal.depreciation_and_amortization
        + terminal.change_in_operating_working_capital
    )
    assert abs(terminal_net_reinvestment / terminal.nopat - Decimal("0.2")) < Decimal(
        "1e-20"
    )
    assert plans[0].terminal_return_on_invested_capital == Decimal("15")
    assert plans[0].terminal_reinvestment_rate == Decimal("20")
    for forecast in forecasts:
        assert len(forecast.observations) == len(forecast.parameters.revenue_growth)
        assert all(
            item.cell_audits["revenue_growth"].source == "adaptive_multistage"
            for item in forecast.observations
        )
        assert all(
            item.cell_audits["revenue_growth"].confidence == "medium"
            for item in forecast.observations
        )
        assert all(
            "revenue_growth=adaptive_multistage" in item.cell_audits["fcff"].source
            for item in forecast.observations
        )

    bridge = FcffDcfCapitalBridge(
        fiscal_year=2024,
        period_end=datetime.date(2024, 12, 31),
        unit="USD",
        net_debt=Decimal(0),
        diluted_shares=Decimal(1),
        net_debt_source="test",
        shares_source="test",
    )
    dcf_parameters = FcffDcfParameters(wacc="8", perpetual_growth_rate="3")
    values = [
        FcffDcfService().value(forecast, dcf_parameters, bridge)
        for forecast in forecasts
    ]
    assert abs(values[0].enterprise_value - values[1].enterprise_value) < Decimal(
        "1e-20"
    )
    assert not any(
        "terminal transition is abrupt" in item for item in values[0].warnings
    )


def test_forward_evidence_delays_growth_fade_without_exceeding_stage_caps():
    financials = _fcff_financials()
    parameters = FcffForecastParameters(forecast_years=5)
    service = FcffForecastService()
    seed = service.forecast(financials, parameters)
    adaptive = AdaptiveMultistageFcffForecastService(service)
    configuration = MultistageValuationConfiguration(
        terminal_return_on_invested_capital=Decimal("15")
    )
    _, historical_plan = adaptive.forecast(
        financials, seed, parameters, Decimal("3"), configuration
    )
    _, evidence_plan = adaptive.forecast(
        financials,
        seed,
        parameters,
        Decimal("3"),
        configuration,
        forward_evidence=ForwardGrowthEvidence(
            backlog=True,
            guidance=True,
            capacity=True,
            growth_visibility=Decimal("1"),
            lifecycle="growth",
        ),
    )

    assert evidence_plan.high_growth_years >= historical_plan.high_growth_years
    assert evidence_plan.transition_years >= historical_plan.transition_years
    assert evidence_plan.effective_years <= 30
    assert evidence_plan.forward_evidence_score > 0
    assert evidence_plan.forward_evidence_summary


def test_ytd_slowdown_does_not_replace_normalized_forward_growth_anchor():
    observations = list(_fcff_financials().observations)
    quarterly_values = {
        FinancialConcept.REVENUE: "25",
        FinancialConcept.OPERATING_INCOME: "5",
        FinancialConcept.PRETAX_INCOME: "4",
        FinancialConcept.INCOME_TAX_EXPENSE: "1",
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION: "1",
        FinancialConcept.CAPITAL_EXPENDITURES: "2",
        FinancialConcept.ACCOUNTS_RECEIVABLE: "5",
        FinancialConcept.INVENTORY: "2",
        FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: "1",
        FinancialConcept.ACCOUNTS_PAYABLE: "2",
        FinancialConcept.ACCRUED_LIABILITIES: "1",
        FinancialConcept.DEFERRED_REVENUE_CURRENT: "1",
    }
    for period, period_end in (
        (FiscalPeriod.Q1, datetime.date(2025, 3, 31)),
        (FiscalPeriod.Q2, datetime.date(2025, 6, 30)),
    ):
        observations.extend(
            _observation(
                concept,
                value,
                2025,
                granularity=Granularity.QUARTERLY,
                fiscal_period=period,
                period_end=period_end,
            )
            for concept, value in quarterly_values.items()
        )
    financials = _fcff_financials().model_copy(update={"observations": observations})
    service = FcffForecastService()
    seed_parameters = FcffForecastParameters(
        forecast_years=5,
        revenue_growth=Decimal("2.4"),
    )
    seed = service.forecast(financials, seed_parameters)
    requested = seed_parameters.model_copy(update={"revenue_growth": None})
    forecast, plan = AdaptiveMultistageFcffForecastService(service).forecast(
        financials,
        seed,
        requested,
        Decimal("3"),
        MultistageValuationConfiguration(
            terminal_return_on_invested_capital=Decimal("15")
        ),
    )

    assert seed.seed_type == ForecastSeedType.YTD_PLUS_FORECAST
    assert seed.observations[0].revenue_growth == Decimal("2.4")
    assert forecast.observations[0].revenue_growth == Decimal("2.4")
    assert plan.current_growth_rate == Decimal("2.4")
    assert plan.forward_growth_rate == Decimal("20")
    assert plan.forward_growth_rate > plan.terminal_growth_rate
    assert not plan.stable_state_supported
    assert forecast.observations[1].revenue_growth == Decimal("20")
    assert plan.current_growth_years == 1


def test_fixed_ytd_scenario_preserves_current_stage_boundaries():
    observations = list(_fcff_financials().observations)
    quarterly_values = {
        FinancialConcept.REVENUE: "25",
        FinancialConcept.OPERATING_INCOME: "5",
        FinancialConcept.PRETAX_INCOME: "4",
        FinancialConcept.INCOME_TAX_EXPENSE: "1",
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION: "1",
        FinancialConcept.CAPITAL_EXPENDITURES: "2",
        FinancialConcept.ACCOUNTS_RECEIVABLE: "5",
        FinancialConcept.INVENTORY: "2",
        FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: "1",
        FinancialConcept.ACCOUNTS_PAYABLE: "2",
        FinancialConcept.ACCRUED_LIABILITIES: "1",
        FinancialConcept.DEFERRED_REVENUE_CURRENT: "1",
    }
    for period, period_end in (
        (FiscalPeriod.Q1, datetime.date(2025, 3, 31)),
        (FiscalPeriod.Q2, datetime.date(2025, 6, 30)),
    ):
        observations.extend(
            _observation(
                concept,
                value,
                2025,
                granularity=Granularity.QUARTERLY,
                fiscal_period=period,
                period_end=period_end,
            )
            for concept, value in quarterly_values.items()
        )
    financials = _fcff_financials().model_copy(update={"observations": observations})
    service = FcffForecastService()
    adaptive = AdaptiveMultistageFcffForecastService(service)
    parameters = FcffForecastParameters(forecast_years=5)
    configuration = MultistageValuationConfiguration(
        terminal_return_on_invested_capital=Decimal("15")
    )
    seed = service.forecast(financials, parameters)
    base_forecast, base_plan = adaptive.forecast(
        financials,
        seed,
        parameters,
        Decimal("3"),
        configuration,
    )
    scenario_forecast, scenario_plan = adaptive.forecast(
        financials,
        base_forecast,
        parameters.model_copy(
            update={"revenue_growth": (Decimal("4.4"),) * parameters.forecast_years}
        ),
        Decimal("3"),
        configuration,
        fixed_plan=base_plan,
    )

    assert base_plan.current_growth_years == 1
    assert scenario_plan.current_growth_years == base_plan.current_growth_years
    assert (
        scenario_plan.explicit_growth_prefix_years,
        scenario_plan.high_growth_years,
        scenario_plan.transition_years,
        scenario_plan.stable_years,
    ) == (
        base_plan.explicit_growth_prefix_years,
        base_plan.high_growth_years,
        base_plan.transition_years,
        base_plan.stable_years,
    )
    expected_stages = (
        ("current",) * base_plan.current_growth_years
        + ("explicit",) * base_plan.explicit_growth_prefix_years
        + ("near_term",) * base_plan.high_growth_years
        + ("transition",) * base_plan.transition_years
        + ("stable",) * base_plan.stable_years
    )
    assert scenario_forecast.adaptive_stages == expected_stages
    assert len(scenario_forecast.observations) == len(expected_stages)
    assert scenario_forecast.observations[0].revenue_growth == Decimal("4.4")

    transition_start = (
        base_plan.current_growth_years
        + base_plan.explicit_growth_prefix_years
        + base_plan.high_growth_years
    )
    assert base_plan.transition_years > 0
    assert scenario_forecast.observations[transition_start].revenue_growth == (
        scenario_plan.initial_growth_rate
        + (Decimal("3") - scenario_plan.initial_growth_rate)
        / Decimal(base_plan.transition_years)
    )
    stable_start = transition_start + base_plan.transition_years
    assert all(
        observation.revenue_growth == Decimal("3")
        for observation in scenario_forecast.observations[stable_start:]
    )


def test_mature_low_growth_history_can_support_stable_state_without_current_proximity():
    financials = _fcff_financials().model_copy(
        update={
            "observations": [
                item.model_copy(update={"value": Decimal("102.5")})
                if item.concept == FinancialConcept.REVENUE
                and item.fiscal_year == 2024
                else item
                for item in _fcff_financials().observations
            ]
        }
    )
    service = FcffForecastService()
    parameters = FcffForecastParameters(forecast_years=5)
    seed = service.forecast(financials, parameters)
    forecast, plan = AdaptiveMultistageFcffForecastService(service).forecast(
        financials,
        seed,
        parameters,
        Decimal("3"),
        MultistageValuationConfiguration(
            terminal_return_on_invested_capital=Decimal("15")
        ),
        forward_evidence=ForwardGrowthEvidence(lifecycle="mature"),
    )

    assert plan.forward_growth_rate == Decimal("2.5")
    assert plan.stable_state_supported
    assert plan.high_growth_years == 0
    assert plan.transition_years == 0
    assert plan.stable_years == 5
    assert all(item.revenue_growth == Decimal("3") for item in forecast.observations)
    assert "near_term" not in forecast.adaptive_stages


def test_explicit_forward_growth_path_is_preserved_before_terminal_fade():
    financials = _fcff_financials()
    service = FcffForecastService()
    parameters = FcffForecastParameters(forecast_years=5)
    seed = service.forecast(financials, parameters)
    forward_growth = ForwardGrowthOutlook(
        growth_path=(Decimal("12"), Decimal("10"), Decimal("8")),
        source=ForecastAssumptionSource.EXPLICIT.value,
        confidence="high",
    )
    forecast, plan = AdaptiveMultistageFcffForecastService(service).forecast(
        financials,
        seed,
        parameters,
        Decimal("3"),
        MultistageValuationConfiguration(
            terminal_return_on_invested_capital=Decimal("15")
        ),
        forward_growth=forward_growth,
    )

    assert plan.forward_growth_source == ForecastAssumptionSource.EXPLICIT.value
    assert plan.forward_growth_confidence == "high"
    assert plan.forward_growth_path == (Decimal("12"), Decimal("10"), Decimal("8"))
    assert [
        item.revenue_growth for item in forecast.observations[:3]
    ] == [Decimal("12"), Decimal("10"), Decimal("8")]
    assert forecast.observations[3].revenue_growth > Decimal("3")


def test_management_revenue_anchor_resolves_as_forward_growth_evidence():
    financials = _fcff_financials()
    service = FcffForecastService()
    seed_parameters = FcffForecastParameters(forecast_years=5)
    seed = service.forecast(financials, seed_parameters)
    requested = seed_parameters.model_copy(
        update={
            "revenue_anchors": {2025: Decimal("132")},
            "revenue_anchor_sources": {
                2025: ForecastAssumptionSource.MANAGEMENT_GUIDANCE
            },
        }
    )
    forecast, plan = AdaptiveMultistageFcffForecastService(service).forecast(
        financials,
        seed,
        requested,
        Decimal("3"),
        MultistageValuationConfiguration(
            terminal_return_on_invested_capital=Decimal("15")
        ),
    )

    assert plan.forward_growth_source == ForecastAssumptionSource.MANAGEMENT_GUIDANCE.value
    assert plan.forward_growth_rate == Decimal("10")
    assert plan.forward_growth_confidence == "high"
    assert forecast.observations[0].revenue == Decimal("132")
    assert forecast.observations[0].cell_audits["revenue"].source.startswith(
        "derived["
    )


def test_run_rate_only_growth_is_low_confidence_and_does_not_claim_stability():
    financials = _fcff_financials().model_copy(
        update={
            "observations": [
                item
                for item in _fcff_financials().observations
                if item.fiscal_year == 2024
            ]
        }
    )
    parameters = FcffForecastParameters(
        forecast_years=5,
        revenue_growth=Decimal("2"),
        operating_margin=Decimal("25"),
        tax_rate=Decimal("20"),
        depreciation_to_revenue=Decimal("4"),
        capex_to_revenue=Decimal("6"),
        operating_working_capital_to_revenue=Decimal("15"),
    )
    service = FcffForecastService()
    seed = service.forecast(financials, parameters)
    requested = parameters.model_copy(update={"revenue_growth": None})
    forecast, plan = AdaptiveMultistageFcffForecastService(service).forecast(
        financials,
        seed,
        requested,
        Decimal("3"),
        MultistageValuationConfiguration(
            terminal_return_on_invested_capital=Decimal("15")
        ),
    )

    assert plan.forward_growth_source == ForecastAssumptionSource.CURRENT_RUN_RATE.value
    assert plan.forward_growth_confidence == "low"
    assert not plan.stable_state_supported
    assert any(
        warning.startswith("LOW confidence") and "run-rate" in warning
        for warning in forecast.warnings
    )


def test_adaptive_multistage_preserves_guidance_prefix_sources_by_year():
    financials = _fcff_financials()
    base_service = FcffForecastService()
    parameters = FcffForecastParameters(
        forecast_years=5,
        revenue_growth=(Decimal("10"), Decimal("8")),
        assumption_source_overrides={
            FcffForecastDriver.REVENUE_GROWTH: ForecastAssumptionSource.MANAGEMENT_GUIDANCE
        },
    )
    seed = base_service.forecast(financials, parameters)
    configuration = MultistageValuationConfiguration(
        terminal_return_on_invested_capital=Decimal("15")
    )
    forecast, plan = AdaptiveMultistageFcffForecastService(base_service).forecast(
        financials,
        seed,
        parameters,
        Decimal("3"),
        configuration,
    )

    assert plan.explicit_growth_prefix_years == 2
    assert [
        item.cell_audits["revenue_growth"].source for item in forecast.observations[:2]
    ] == ["management_guidance", "management_guidance"]
    assert forecast.observations[2].cell_audits["revenue_growth"].source == (
        "adaptive_multistage"
    )
    assert (
        "transition" in forecast.observations[2].cell_audits["revenue_growth"].method
        or "stable" in forecast.observations[2].cell_audits["revenue_growth"].method
    )


def _absolute_capex_shock_parameters() -> FcffForecastParameters:
    return FcffForecastParameters(
        forecast_years=1,
        revenue_growth=Decimal("0"),
        operating_margin=Decimal("25"),
        tax_rate=Decimal("20"),
        operating_working_capital_to_revenue=Decimal("15"),
        capex_constraints={
            2025: MonetaryForecastConstraint(point=Decimal("30")),
        },
    )


def test_adaptive_capex_shock_fades_without_revenue_transition_or_a_capex_cliff():
    parameters = _absolute_capex_shock_parameters()
    base_service = FcffForecastService()
    seed = base_service.forecast(_fcff_financials(), parameters)
    configuration = MultistageValuationConfiguration(
        terminal_return_on_invested_capital=Decimal("15"),
        capex_transition_years=3,
        depreciable_asset_life_years=4,
    )

    forecast, plan = AdaptiveMultistageFcffForecastService(base_service).forecast(
        _fcff_financials(),
        seed,
        parameters,
        Decimal("0"),
        configuration,
    )

    assert plan.transition_years == 0
    assert plan.capex_transition_years == 3
    assert plan.effective_years == 4
    assert [item.revenue_growth for item in forecast.observations] == [Decimal("0")] * 4
    assert forecast.observations[0].capital_expenditures == Decimal("30")
    assert forecast.observations[0].capex_to_revenue == Decimal("25")
    assert [item.capital_expenditures for item in forecast.observations] == sorted(
        (item.capital_expenditures for item in forecast.observations), reverse=True
    )
    assert forecast.observations[-1].capex_to_revenue == (
        plan.terminal_capex_to_revenue
    )
    assert forecast.capex_constraints_applied == (2025,)
    assert not FcffForecastService().economic_identity_issues(forecast)
    assert not plan.capex_benefits_modeled
    assert "not modeled" in plan.capex_benefits_disclosure
    assert forecast.observations[-1].capex_to_revenue == plan.terminal_capex_to_revenue


def test_adaptive_rejects_a_material_absolute_capex_shock_without_asset_life():
    parameters = _absolute_capex_shock_parameters()
    base_service = FcffForecastService()
    seed = base_service.forecast(_fcff_financials(), parameters)
    configuration = MultistageValuationConfiguration(
        terminal_return_on_invested_capital=Decimal("15"),
    )

    with pytest.raises(ValueError, match="Material absolute CAPEX shock.*asset"):
        AdaptiveMultistageFcffForecastService(base_service).forecast(
            _fcff_financials(),
            seed,
            parameters,
            Decimal("0"),
            configuration,
        )


def test_adaptive_capex_shock_rolls_d_and_a_forward_from_post_shock_capex():
    parameters = _absolute_capex_shock_parameters()
    base_service = FcffForecastService()
    seed = base_service.forecast(_fcff_financials(), parameters)
    configuration = MultistageValuationConfiguration(
        terminal_return_on_invested_capital=Decimal("15"),
        capex_transition_years=3,
        depreciable_asset_life_years=4,
    )

    forecast, _ = AdaptiveMultistageFcffForecastService(base_service).forecast(
        _fcff_financials(),
        seed,
        parameters,
        Decimal("0"),
        configuration,
    )

    first, second = forecast.observations[:2]
    assert first.depreciation_and_amortization == Decimal("5")
    assert second.depreciation_and_amortization > first.depreciation_and_amortization
    assert second.depreciation_and_amortization == Decimal("11.25")
    assert (
        "depreciation_to_revenue=adaptive_multistage"
        in second.cell_audits["depreciation_and_amortization"].source
    )


def test_generic_forecast_service_is_the_fcff_default():
    assert FreeCashFlowForecastService is FcffForecastService
    assert FcffForecastService.required_concepts() == (
        FinancialMetricsService.required_concepts({FinancialMetric.FCFF})
        | {FinancialConcept.REVENUE}
    )


def test_fcff_reports_periods_missing_both_liability_representations():
    financials = _fcff_financials().model_copy(
        update={
            "observations": [
                item
                for item in _fcff_financials().observations
                if item.concept != FinancialConcept.ACCRUED_LIABILITIES
            ]
        }
    )

    with pytest.raises(ValueError, match="working-capital liabilities"):
        FcffForecastService().forecast(financials)


def test_fcff_falls_back_to_aggregate_current_liabilities_for_working_capital():
    financials = _fcff_financials()
    current_liabilities = {2023: "14", 2024: "16"}
    financials = financials.model_copy(
        update={
            "observations": [
                item
                for item in financials.observations
                if item.concept != FinancialConcept.ACCRUED_LIABILITIES
            ]
            + [
                _observation(
                    FinancialConcept.CURRENT_LIABILITIES,
                    value,
                    fiscal_year,
                )
                for fiscal_year, value in current_liabilities.items()
            ]
        }
    )

    forecast = FcffForecastService().forecast(
        financials,
        FcffForecastParameters(
            forecast_years=1,
            revenue_growth=Decimal("5"),
            operating_margin=Decimal("25"),
            tax_rate=Decimal("20"),
            depreciation_to_revenue=Decimal("4"),
            capex_to_revenue=Decimal("5"),
            operating_working_capital_to_revenue=Decimal("10"),
        ),
    )

    assert forecast.base_operating_working_capital == Decimal("20")
    assert forecast.base_fcff == Decimal("19.0")


def test_fcff_falls_back_to_aggregate_current_assets_and_liabilities():
    financials = _fcff_financials()
    aggregate_balances = {
        2023: {
            FinancialConcept.CURRENT_ASSETS: "50",
            FinancialConcept.CASH_AND_EQUIVALENTS: "8",
            FinancialConcept.SHORT_TERM_INVESTMENTS: "2",
            FinancialConcept.CURRENT_LIABILITIES: "30",
            FinancialConcept.SHORT_TERM_DEBT: "5",
        },
        2024: {
            FinancialConcept.CURRENT_ASSETS: "60",
            FinancialConcept.CASH_AND_EQUIVALENTS: "10",
            FinancialConcept.SHORT_TERM_INVESTMENTS: "3",
            FinancialConcept.CURRENT_LIABILITIES: "35",
            FinancialConcept.SHORT_TERM_DEBT: "8",
        },
    }
    financials = financials.model_copy(
        update={
            "observations": [
                item
                for item in financials.observations
                if item.concept
                not in {
                    FinancialConcept.ACCOUNTS_RECEIVABLE,
                    FinancialConcept.INVENTORY,
                    FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS,
                    FinancialConcept.ACCOUNTS_PAYABLE,
                    FinancialConcept.ACCRUED_LIABILITIES,
                    FinancialConcept.DEFERRED_REVENUE_CURRENT,
                }
            ]
            + [
                _observation(concept, value, fiscal_year)
                for fiscal_year, period_values in aggregate_balances.items()
                for concept, value in period_values.items()
            ]
        }
    )

    forecast = FcffForecastService().forecast(
        financials,
        FcffForecastParameters(forecast_years=1),
    )

    assert forecast.historical_fiscal_years == (2023, 2024)
    assert forecast.base_operating_working_capital == Decimal("20")
    assert forecast.base_fcff == Decimal("18.0")


def test_fcff_accepts_inventory_folded_into_other_current_assets():
    financials = _fcff_financials()
    combined_other_assets = {2023: "15", 2024: "18"}
    financials = financials.model_copy(
        update={
            "observations": [
                item
                for item in financials.observations
                if item.concept
                not in {
                    FinancialConcept.INVENTORY,
                    FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS,
                }
            ]
            + [
                _observation(
                    FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS,
                    value,
                    fiscal_year,
                )
                for fiscal_year, value in combined_other_assets.items()
            ]
        }
    )

    forecast = FcffForecastService().forecast(
        financials,
        FcffForecastParameters(forecast_years=1),
    )

    assert forecast.historical_fiscal_years == (2023, 2024)
    assert forecast.base_operating_working_capital == Decimal("20")


def test_cli_defaults_to_driver_based_fcff_from_cached_sec_data(tmp_path, capsys):
    ticker_path = (
        tmp_path
        / "providers"
        / "edgar"
        / "www.sec.gov"
        / "files"
        / "company_tickers.json"
    )
    facts_path = (
        tmp_path
        / "providers"
        / "edgar"
        / "data.sec.gov"
        / "api"
        / "xbrl"
        / "companyfacts"
        / "CIK0000320193.json"
    )
    ticker_path.parent.mkdir(parents=True)
    facts_path.parent.mkdir(parents=True)
    ticker_path.write_text(
        json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}),
        encoding="utf-8",
    )
    facts_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    exit_code = main(
        [
            "forecast",
            "--ticker",
            "AAPL",
            "--years",
            "2",
            "--revenue-growth",
            "5",
            "--operating-margin",
            "25",
            "--tax-rate",
            "21",
            "--depreciation-to-revenue",
            "4",
            "--capex-to-revenue",
            "3",
            "--operating-working-capital-to-revenue",
            "10",
            "--cache-dir",
            str(tmp_path),
            "--user-agent",
            "Edgarito Tests (tests@example.com)",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "AAPL - Apple Inc." in output
    assert "Method: driver-based FCFF" in output
    assert "Revenue Growth: explicit" in output
    assert "FY2026E" in output
    assert "FY2027E" in output
    assert "FCFF (USD B)" in output

    simplified_exit_code = main(
        [
            "forecast",
            "--ticker",
            "AAPL",
            "--method",
            "simplified",
            "--years",
            "1",
            "--revenue-growth",
            "5",
            "--fcf-margin",
            "20",
            "--cache-dir",
            str(tmp_path),
            "--user-agent",
            "Edgarito Tests (tests@example.com)",
        ]
    )
    simplified_output = capsys.readouterr().out
    assert simplified_exit_code == 0
    assert "Method: simplified projected revenue × free cash flow margin" in (
        simplified_output
    )
