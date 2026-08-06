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
from edgarito.services.metrics import FinancialMetric, FinancialMetricsService

FIXTURE = Path(__file__).parent / "fixtures" / "aapl_facts.json"


def _observation(
    concept: FinancialConcept,
    value: str,
    fiscal_year: int,
    fiscal_period: FiscalPeriod = FiscalPeriod.FY,
    granularity: Granularity = Granularity.ANNUAL,
) -> FinancialObservation:
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal(value),
        unit="USD",
        granularity=granularity,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        period_end=datetime.date(fiscal_year, 12, 31),
        provider="test",
        taxonomy="test",
        source_concept=concept.value,
    )


def _financials() -> NormalizedCompanyFinancials:
    values = {
        2023: {
            FinancialConcept.REVENUE: "100",
            FinancialConcept.OPERATING_INCOME: "20",
            FinancialConcept.PRETAX_INCOME: "18",
            FinancialConcept.INCOME_TAX_EXPENSE: "3.6",
            FinancialConcept.NET_INCOME: "10",
            FinancialConcept.TOTAL_ASSETS: "200",
            FinancialConcept.TOTAL_LIABILITIES: "100",
            FinancialConcept.STOCKHOLDERS_EQUITY: "100",
            FinancialConcept.CASH_AND_EQUIVALENTS: "20",
            FinancialConcept.OPERATING_CASH_FLOW: "15",
            FinancialConcept.DEPRECIATION_AND_AMORTIZATION: "4",
            FinancialConcept.CAPITAL_EXPENDITURES: "5",
            FinancialConcept.ACCOUNTS_RECEIVABLE: "15",
            FinancialConcept.INVENTORY: "10",
            FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: "5",
            FinancialConcept.ACCOUNTS_PAYABLE: "8",
            FinancialConcept.ACCRUED_LIABILITIES: "4",
            FinancialConcept.DEFERRED_REVENUE_CURRENT: "2",
            FinancialConcept.SHORT_TERM_DEBT: "5",
            FinancialConcept.LONG_TERM_DEBT_CURRENT: "3",
            FinancialConcept.LONG_TERM_DEBT_NONCURRENT: "40",
            FinancialConcept.GOODWILL: "10",
            FinancialConcept.INTANGIBLE_ASSETS_NET: "5",
        },
        2024: {
            FinancialConcept.REVENUE: "120",
            FinancialConcept.OPERATING_INCOME: "30",
            FinancialConcept.PRETAX_INCOME: "24",
            FinancialConcept.INCOME_TAX_EXPENSE: "4.8",
            FinancialConcept.NET_INCOME: "12",
            FinancialConcept.TOTAL_ASSETS: "220",
            FinancialConcept.TOTAL_LIABILITIES: "110",
            FinancialConcept.STOCKHOLDERS_EQUITY: "110",
            FinancialConcept.CASH_AND_EQUIVALENTS: "22",
            FinancialConcept.OPERATING_CASH_FLOW: "18",
            FinancialConcept.DEPRECIATION_AND_AMORTIZATION: "5",
            FinancialConcept.CAPITAL_EXPENDITURES: "6",
            FinancialConcept.ACCOUNTS_RECEIVABLE: "18",
            FinancialConcept.INVENTORY: "12",
            FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: "6",
            FinancialConcept.ACCOUNTS_PAYABLE: "9",
            FinancialConcept.ACCRUED_LIABILITIES: "5",
            FinancialConcept.DEFERRED_REVENUE_CURRENT: "2",
            FinancialConcept.SHORT_TERM_DEBT: "6",
            FinancialConcept.LONG_TERM_DEBT_CURRENT: "4",
            FinancialConcept.LONG_TERM_DEBT_NONCURRENT: "42",
            FinancialConcept.GOODWILL: "11",
            FinancialConcept.INTANGIBLE_ASSETS_NET: "6",
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


def _metric_value(company_metrics, metric: FinancialMetric, fiscal_year: int):
    return next(
        observation.value
        for observation in company_metrics.observations
        if observation.metric == metric and observation.fiscal_year == fiscal_year
    )


def test_financial_observation_rejects_a_concept_statement_mismatch():
    with pytest.raises(ValueError, match="revenue belongs to income_statement"):
        FinancialObservation(
            concept=FinancialConcept.REVENUE,
            statement=FinancialConcept.TOTAL_ASSETS.statement,
            value=Decimal("100"),
            unit="USD",
            granularity=Granularity.ANNUAL,
            fiscal_year=2024,
            fiscal_period=FiscalPeriod.FY,
            period_end=datetime.date(2024, 12, 31),
            provider="test",
            taxonomy="test",
            source_concept="revenue",
        )


def test_calculates_supported_metrics_without_mixing_periods():
    metrics = FinancialMetricsService().calculate(_financials())

    assert metrics.provider == "test"
    assert metrics.ticker == "TEST"
    assert _metric_value(metrics, FinancialMetric.REVENUE_GROWTH, 2024) == Decimal(
        "20.0"
    )
    assert _metric_value(metrics, FinancialMetric.OPERATING_MARGIN, 2024) == Decimal(
        "25.00"
    )
    assert _metric_value(metrics, FinancialMetric.NET_MARGIN, 2024) == Decimal("10.0")
    assert _metric_value(metrics, FinancialMetric.EFFECTIVE_TAX_RATE, 2024) == Decimal(
        "20.0"
    )
    assert _metric_value(metrics, FinancialMetric.NOPAT, 2024) == Decimal("24.0")
    assert _metric_value(metrics, FinancialMetric.EBITDA, 2024) == Decimal("35")
    assert _metric_value(metrics, FinancialMetric.FREE_CASH_FLOW, 2024) == Decimal("12")
    assert _metric_value(
        metrics, FinancialMetric.FREE_CASH_FLOW_MARGIN, 2024
    ) == Decimal("10.0")
    assert _metric_value(metrics, FinancialMetric.RETURN_ON_ASSETS, 2024) == (
        Decimal("12") / Decimal("210") * Decimal("100")
    )
    assert _metric_value(metrics, FinancialMetric.RETURN_ON_EQUITY, 2024) == (
        Decimal("12") / Decimal("105") * Decimal("100")
    )
    assert _metric_value(
        metrics, FinancialMetric.LIABILITIES_TO_ASSETS, 2024
    ) == Decimal("50.0")
    assert _metric_value(metrics, FinancialMetric.CASH_TO_LIABILITIES, 2024) == Decimal(
        "20.0"
    )
    assert _metric_value(
        metrics, FinancialMetric.OPERATING_CASH_FLOW_TO_NET_INCOME, 2024
    ) == Decimal("150.0")
    assert _metric_value(
        metrics, FinancialMetric.OPERATING_WORKING_CAPITAL, 2024
    ) == Decimal("20")
    assert _metric_value(
        metrics, FinancialMetric.CHANGE_IN_OPERATING_WORKING_CAPITAL, 2024
    ) == Decimal("4")
    assert _metric_value(metrics, FinancialMetric.GROSS_DEBT, 2024) == Decimal("52")
    assert _metric_value(metrics, FinancialMetric.NET_DEBT, 2024) == Decimal("30")
    assert _metric_value(
        metrics, FinancialMetric.TANGIBLE_BOOK_EQUITY, 2024
    ) == Decimal("93")
    assert _metric_value(metrics, FinancialMetric.FCFF, 2024) == Decimal("19.0")

    first_year_metrics = {
        observation.metric
        for observation in metrics.observations
        if observation.fiscal_year == 2023
    }
    assert FinancialMetric.REVENUE_GROWTH not in first_year_metrics
    assert FinancialMetric.RETURN_ON_ASSETS not in first_year_metrics
    assert FinancialMetric.RETURN_ON_EQUITY not in first_year_metrics
    assert FinancialMetric.CHANGE_IN_OPERATING_WORKING_CAPITAL not in first_year_metrics
    assert FinancialMetric.FCFF not in first_year_metrics


def test_valuation_metrics_require_complete_inputs_and_consistent_units():
    financials = _financials()
    financials.observations = [
        observation
        for observation in financials.observations
        if not (
            observation.fiscal_year == 2024
            and observation.concept == FinancialConcept.ACCRUED_LIABILITIES
        )
    ]
    debt = next(
        observation
        for observation in financials.observations
        if observation.fiscal_year == 2024
        and observation.concept == FinancialConcept.LONG_TERM_DEBT_NONCURRENT
    )
    debt.unit = "EUR"

    metrics = FinancialMetricsService().calculate(
        financials,
        metrics={
            FinancialMetric.OPERATING_WORKING_CAPITAL,
            FinancialMetric.CHANGE_IN_OPERATING_WORKING_CAPITAL,
            FinancialMetric.GROSS_DEBT,
            FinancialMetric.NET_DEBT,
            FinancialMetric.FCFF,
        },
    )
    second_year_metrics = {
        observation.metric
        for observation in metrics.observations
        if observation.fiscal_year == 2024
    }

    assert second_year_metrics == set()


def test_debt_metrics_sum_reported_components_when_one_line_is_absent():
    financials = _financials()
    financials.observations = [
        observation
        for observation in financials.observations
        if not (
            observation.fiscal_year == 2024
            and observation.concept == FinancialConcept.SHORT_TERM_DEBT
        )
    ]

    metrics = FinancialMetricsService().calculate(
        financials,
        metrics={FinancialMetric.GROSS_DEBT, FinancialMetric.NET_DEBT},
    )

    assert _metric_value(metrics, FinancialMetric.GROSS_DEBT, 2024) == Decimal("46")
    assert _metric_value(metrics, FinancialMetric.NET_DEBT, 2024) == Decimal("24")
    gross_debt = next(
        observation
        for observation in metrics.observations
        if observation.metric == FinancialMetric.GROSS_DEBT
        and observation.fiscal_year == 2024
    )
    assert FinancialConcept.SHORT_TERM_DEBT not in gross_debt.input_concepts


def test_fcff_retains_formula_and_atomic_input_concepts():
    metrics = FinancialMetricsService().calculate(
        _financials(), metrics={FinancialMetric.FCFF}
    )

    fcff = next(observation for observation in metrics.observations)
    assert fcff.fiscal_year == 2024
    assert fcff.formula == (
        "NOPAT + depreciation and amortization - capital expenditures - "
        "change in operating working capital"
    )
    assert set(fcff.input_concepts) == FinancialMetricsService.required_concepts(
        {FinancialMetric.FCFF}
    )


def test_filters_metrics_and_reports_their_required_concepts():
    selected = {FinancialMetric.NET_MARGIN}
    metrics = FinancialMetricsService().calculate(_financials(), metrics=selected)

    assert {observation.metric for observation in metrics.observations} == selected
    assert FinancialMetricsService.required_concepts(selected) == {
        FinancialConcept.REVENUE,
        FinancialConcept.NET_INCOME,
    }


def test_growth_keeps_annual_and_quarterly_series_independent():
    financials = NormalizedCompanyFinancials(
        provider="test",
        company_id="1",
        company_name="Test Company",
        observations=[
            _observation(FinancialConcept.REVENUE, "100", 2024),
            _observation(FinancialConcept.REVENUE, "110", 2025),
            _observation(
                FinancialConcept.REVENUE,
                "40",
                2024,
                FiscalPeriod.Q4,
                Granularity.QUARTERLY,
            ),
            _observation(
                FinancialConcept.REVENUE,
                "50",
                2025,
                FiscalPeriod.Q1,
                Granularity.QUARTERLY,
            ),
        ],
    )

    metrics = FinancialMetricsService().calculate(
        financials, metrics={FinancialMetric.REVENUE_GROWTH}
    )

    assert [
        (observation.granularity, observation.value)
        for observation in metrics.observations
    ] == [
        (Granularity.ANNUAL, Decimal("10.0")),
        (Granularity.QUARTERLY, Decimal("25.00")),
    ]


def test_growth_requires_a_consecutive_period_and_nonzero_denominator():
    financials = NormalizedCompanyFinancials(
        provider="test",
        company_id="1",
        company_name="Test Company",
        observations=[
            _observation(FinancialConcept.REVENUE, "0", 2023),
            _observation(FinancialConcept.REVENUE, "120", 2024),
            _observation(FinancialConcept.REVENUE, "150", 2026),
        ],
    )

    metrics = FinancialMetricsService().calculate(
        financials, metrics={FinancialMetric.REVENUE_GROWTH}
    )

    assert metrics.observations == []


def test_cli_displays_selected_metrics_from_cached_sec_data(tmp_path, capsys):
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
            "metrics",
            "--ticker",
            "AAPL",
            "--metric",
            "net_margin",
            "--metric",
            "free_cash_flow",
            "--cache-dir",
            str(tmp_path),
            "--user-agent",
            "Edgarito Tests (tests@example.com)",
            "--limit",
            "2",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "AAPL - Apple Inc." in output
    assert "Net Margin (%)" in output
    assert "Free Cash Flow (USD B)" in output
    assert "Operating Margin" not in output
    assert "FY2024" in output
    assert "FY2025" in output
