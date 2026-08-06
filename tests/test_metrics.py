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
            FinancialConcept.NET_INCOME: "10",
            FinancialConcept.TOTAL_ASSETS: "200",
            FinancialConcept.TOTAL_LIABILITIES: "100",
            FinancialConcept.STOCKHOLDERS_EQUITY: "100",
            FinancialConcept.CASH_AND_EQUIVALENTS: "20",
            FinancialConcept.OPERATING_CASH_FLOW: "15",
            FinancialConcept.CAPITAL_EXPENDITURES: "5",
        },
        2024: {
            FinancialConcept.REVENUE: "120",
            FinancialConcept.OPERATING_INCOME: "30",
            FinancialConcept.NET_INCOME: "12",
            FinancialConcept.TOTAL_ASSETS: "220",
            FinancialConcept.TOTAL_LIABILITIES: "110",
            FinancialConcept.STOCKHOLDERS_EQUITY: "110",
            FinancialConcept.CASH_AND_EQUIVALENTS: "22",
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

    first_year_metrics = {
        observation.metric
        for observation in metrics.observations
        if observation.fiscal_year == 2023
    }
    assert FinancialMetric.REVENUE_GROWTH not in first_year_metrics
    assert FinancialMetric.RETURN_ON_ASSETS not in first_year_metrics
    assert FinancialMetric.RETURN_ON_EQUITY not in first_year_metrics


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
