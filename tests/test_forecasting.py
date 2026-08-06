import datetime
import json
from decimal import Decimal
from pathlib import Path

import pytest

from edgarito.cli import main
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.forecasting import (
    ForecastAssumptionSource,
    FreeCashFlowForecastParameters,
    FreeCashFlowForecastService,
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
    parameters = FreeCashFlowForecastParameters(
        forecast_years=2,
        revenue_growth=Decimal("10"),
        free_cash_flow_margin=Decimal("12.5"),
    )

    forecast = FreeCashFlowForecastService().forecast(_financials(), parameters)

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
    parameters = FreeCashFlowForecastParameters(
        forecast_years=2,
        revenue_growth=(Decimal("10"), Decimal("5")),
        free_cash_flow_margin=(Decimal("10"), Decimal("12")),
    )

    forecast = FreeCashFlowForecastService().forecast(_financials(), parameters)

    assert [observation.revenue_growth for observation in forecast.observations] == [
        Decimal("10"),
        Decimal("5"),
    ]
    assert [
        observation.free_cash_flow_margin for observation in forecast.observations
    ] == [Decimal("10"), Decimal("12")]
    assert forecast.observations[-1].free_cash_flow == Decimal("16.6320")


def test_infers_trailing_average_growth_and_fcf_margin():
    parameters = FreeCashFlowForecastParameters(forecast_years=1)

    forecast = FreeCashFlowForecastService().forecast(_financials(), parameters)

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
        FreeCashFlowForecastParameters(
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
        FreeCashFlowForecastService().forecast(financials)


def test_cli_forecasts_from_cached_sec_data(tmp_path, capsys):
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
            "--fcf-margin",
            "20",
            "--cache-dir",
            str(tmp_path),
            "--user-agent",
            "Edgarito Tests (tests@example.com)",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "AAPL - Apple Inc." in output
    assert "Method: projected revenue × free cash flow margin" in output
    assert "Revenue growth assumptions: explicit" in output
    assert "FY2026E" in output
    assert "FY2027E" in output
    assert "Free Cash Flow (USD B)" in output
