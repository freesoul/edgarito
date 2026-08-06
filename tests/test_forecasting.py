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
    FreeCashFlowForecastService,
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
    concept: FinancialConcept, value: str, fiscal_year: int
) -> FinancialObservation:
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal(value),
        unit="USD",
        granularity=Granularity.ANNUAL,
        fiscal_year=fiscal_year,
        fiscal_period=FiscalPeriod.FY,
        period_end=datetime.date(fiscal_year, 12, 31),
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
    assert FcffForecast.model_validate_json(forecast.model_dump_json()) == forecast


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


def test_adaptive_multistage_projection_is_invariant_after_stable_stage():
    financials = _fcff_financials()
    base_service = FcffForecastService()
    adaptive = AdaptiveMultistageFcffForecastService(base_service)
    configuration = MultistageValuationConfiguration()

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
