import datetime
import json
from decimal import Decimal
from pathlib import Path

from edgarito.cli import main
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import FinancialConcept
from edgarito.schemas.providers.edgar.company_facts import CompanyFacts
from edgarito.services.normalization.sec_us_gaap import SecUsGaapNormalizer

FIXTURE = Path(__file__).parent / "fixtures" / "aapl_facts.json"
JPM_FIXTURE = Path(__file__).parent / "fixtures" / "jpm_facts.json"


def load_aapl_facts() -> CompanyFacts:
    return CompanyFacts.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def find_observation(financials, concept, year, period):
    return next(
        observation
        for observation in financials.observations
        if observation.concept == concept
        and observation.fiscal_year == year
        and observation.fiscal_period == period
    )


def test_normalizes_recent_annual_revenue():
    financials = SecUsGaapNormalizer().normalize(
        load_aapl_facts(),
        ticker="AAPL",
        granularity=Granularity.ANNUAL,
    )

    revenue = find_observation(
        financials, FinancialConcept.REVENUE, 2025, FiscalPeriod.FY
    )
    assert revenue.value == Decimal("416161000000")
    assert revenue.granularity == Granularity.ANNUAL
    assert (
        revenue.source_concept == "RevenueFromContractWithCustomerExcludingAssessedTax"
    )
    assert not revenue.is_derived


def test_preserves_non_calendar_fiscal_year_and_derives_q4():
    financials = SecUsGaapNormalizer().normalize(
        load_aapl_facts(),
        ticker="AAPL",
        granularity=Granularity.QUARTERLY,
        concepts={FinancialConcept.REVENUE},
    )

    q1 = find_observation(financials, FinancialConcept.REVENUE, 2024, FiscalPeriod.Q1)
    q4 = find_observation(financials, FinancialConcept.REVENUE, 2024, FiscalPeriod.Q4)

    assert q1.period_end == datetime.date(2023, 12, 30)
    assert q1.value == Decimal("119575000000")
    assert q4.value == Decimal("94930000000")
    assert q4.is_derived
    assert q4.derivation == "Q4 = FY - Q1 - Q2 - Q3"


def test_year_end_balance_sheet_fact_becomes_quarter_four_once():
    financials = SecUsGaapNormalizer().normalize(
        load_aapl_facts(),
        granularity=Granularity.QUARTERLY,
        concepts={FinancialConcept.TOTAL_ASSETS},
    )

    fiscal_2024 = [
        observation
        for observation in financials.observations
        if observation.fiscal_year == 2024
    ]
    assert [observation.fiscal_period for observation in fiscal_2024] == [
        FiscalPeriod.Q1,
        FiscalPeriod.Q2,
        FiscalPeriod.Q3,
        FiscalPeriod.Q4,
    ]
    assert fiscal_2024[-1].value == Decimal("364980000000")


def test_bank_revenue_uses_net_interest_revenue_fallback():
    facts = CompanyFacts.model_validate_json(JPM_FIXTURE.read_text(encoding="utf-8"))
    financials = SecUsGaapNormalizer().normalize(
        facts,
        granularity=Granularity.QUARTERLY,
        concepts={FinancialConcept.REVENUE},
    )

    latest = financials.observations[-1]
    assert latest.fiscal_year == 2025
    assert latest.fiscal_period == FiscalPeriod.Q3
    assert latest.source_concept == "RevenuesNetOfInterestExpense"


def test_cli_runs_from_cached_provider_snapshots(tmp_path, capsys):
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
            "financials",
            "--ticker",
            "AAPL",
            "--cache-dir",
            str(tmp_path),
            "--user-agent",
            "Edgarito Tests tests@example.com",
            "--limit",
            "2",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "AAPL - Apple Inc." in output
    assert "FY2024" in output
    assert "FY2025" in output
    assert "Revenue (USD B)" in output
    assert "QUARTERLY" not in output
