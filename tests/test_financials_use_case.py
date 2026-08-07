import datetime
import json
from decimal import Decimal
from pathlib import Path

from edgarito.cli import main
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    ObservationDerivationKind,
)
from edgarito.schemas.providers.edgar.company_facts import CompanyFacts
from edgarito.services.normalization.sec_us_gaap import SecUsGaapNormalizer

FIXTURE = Path(__file__).parent / "fixtures" / "aapl_facts.json"
JNJ_FIXTURE = Path(__file__).parent / "fixtures" / "jnj_facts.json"
JPM_FIXTURE = Path(__file__).parent / "fixtures" / "jpm_facts.json"
TSLA_FIXTURE = Path(__file__).parent / "fixtures" / "tsla_facts.json"


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
    assert q4.derivation_kind == ObservationDerivationKind.PERIOD_RECONSTRUCTION
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


def test_normalizes_sec_valuation_inputs_with_correct_units_and_taxonomies():
    concepts = {
        FinancialConcept.PRETAX_INCOME,
        FinancialConcept.INCOME_TAX_EXPENSE,
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
        FinancialConcept.CURRENT_ASSETS,
        FinancialConcept.ACCOUNTS_RECEIVABLE,
        FinancialConcept.INVENTORY,
        FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS,
        FinancialConcept.CURRENT_LIABILITIES,
        FinancialConcept.ACCOUNTS_PAYABLE,
        FinancialConcept.ACCRUED_LIABILITIES,
        FinancialConcept.DEFERRED_REVENUE_CURRENT,
        FinancialConcept.SHORT_TERM_DEBT,
        FinancialConcept.LONG_TERM_DEBT_CURRENT,
        FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
        FinancialConcept.DIVIDENDS_PAID,
        FinancialConcept.DIVIDENDS_PER_SHARE,
        FinancialConcept.SHARES_OUTSTANDING,
        FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES,
        FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES,
    }
    financials = SecUsGaapNormalizer().normalize(
        load_aapl_facts(),
        ticker="AAPL",
        granularity=Granularity.ANNUAL,
        concepts=concepts,
    )

    expected = {
        FinancialConcept.PRETAX_INCOME: Decimal("132729000000"),
        FinancialConcept.INCOME_TAX_EXPENSE: Decimal("20719000000"),
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION: Decimal("11698000000"),
        FinancialConcept.CURRENT_ASSETS: Decimal("147957000000"),
        FinancialConcept.ACCOUNTS_RECEIVABLE: Decimal("39777000000"),
        FinancialConcept.INVENTORY: Decimal("5718000000"),
        FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: Decimal("14585000000"),
        FinancialConcept.CURRENT_LIABILITIES: Decimal("165631000000"),
        FinancialConcept.ACCOUNTS_PAYABLE: Decimal("69860000000"),
        FinancialConcept.ACCRUED_LIABILITIES: Decimal("44452000000"),
        FinancialConcept.DEFERRED_REVENUE_CURRENT: Decimal("9055000000"),
        FinancialConcept.SHORT_TERM_DEBT: Decimal("7979000000"),
        FinancialConcept.LONG_TERM_DEBT_CURRENT: Decimal("12350000000"),
        FinancialConcept.LONG_TERM_DEBT_NONCURRENT: Decimal("78328000000"),
        FinancialConcept.DIVIDENDS_PAID: Decimal("15421000000"),
        FinancialConcept.DIVIDENDS_PER_SHARE: Decimal("1.02"),
        FinancialConcept.SHARES_OUTSTANDING: Decimal("14776353000"),
        FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES: Decimal("14948500000"),
        FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES: Decimal("15004697000"),
    }
    observations = {
        concept: find_observation(financials, concept, 2025, FiscalPeriod.FY)
        for concept in concepts
    }

    assert {concept: item.value for concept, item in observations.items()} == expected
    assert observations[FinancialConcept.DIVIDENDS_PER_SHARE].unit == "USD/shares"
    assert observations[FinancialConcept.SHARES_OUTSTANDING].unit == "shares"
    assert observations[FinancialConcept.SHARES_OUTSTANDING].taxonomy == "dei"
    assert (
        observations[FinancialConcept.SHARES_OUTSTANDING].source_concept
        == "EntityCommonStockSharesOutstanding"
    )
    assert observations[FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES].unit == (
        "shares"
    )
    assert (
        observations[FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES].taxonomy
        == "us-gaap"
    )


def test_normalizes_tesla_current_and_total_debt_fallback_tags():
    facts = CompanyFacts.model_validate_json(TSLA_FIXTURE.read_text(encoding="utf-8"))
    financials = SecUsGaapNormalizer().normalize(
        facts,
        ticker="TSLA",
        granularity=Granularity.QUARTERLY,
        concepts={
            FinancialConcept.SHORT_TERM_DEBT,
            FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
        },
    )

    current = find_observation(
        financials, FinancialConcept.SHORT_TERM_DEBT, 2025, FiscalPeriod.Q3
    )
    long_term = find_observation(
        financials,
        FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
        2025,
        FiscalPeriod.Q3,
    )

    assert current.value == Decimal("1852000000")
    assert current.source_concept == "DebtCurrent"
    assert long_term.value == Decimal("5609000000")
    assert long_term.source_concept == "LongTermDebt"


def test_normalizes_interest_goodwill_and_intangibles_with_fallback_tags():
    facts = CompanyFacts.model_validate_json(JNJ_FIXTURE.read_text(encoding="utf-8"))
    financials = SecUsGaapNormalizer().normalize(
        facts,
        granularity=Granularity.ANNUAL,
        concepts={
            FinancialConcept.INTEREST_EXPENSE,
            FinancialConcept.GOODWILL,
            FinancialConcept.INTANGIBLE_ASSETS_NET,
        },
    )

    interest = find_observation(
        financials, FinancialConcept.INTEREST_EXPENSE, 2024, FiscalPeriod.FY
    )
    goodwill = find_observation(
        financials, FinancialConcept.GOODWILL, 2024, FiscalPeriod.FY
    )
    intangibles = find_observation(
        financials, FinancialConcept.INTANGIBLE_ASSETS_NET, 2024, FiscalPeriod.FY
    )

    assert interest.value == Decimal("755000000")
    assert interest.source_concept == "InterestExpenseNonoperating"
    assert goodwill.value == Decimal("44200000000")
    assert goodwill.source_concept == "Goodwill"
    assert intangibles.value == Decimal("37618000000")
    assert intangibles.source_concept == "IntangibleAssetsNetExcludingGoodwill"


def test_combines_separately_reported_depreciation_and_amortization():
    facts = CompanyFacts.model_validate(
        {
            "cik": 789019,
            "entityName": "Microsoft Corporation",
            "facts": {
                "dei": {},
                "us-gaap": {
                    source_concept: {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-07-01",
                                    "end": "2026-06-30",
                                    "val": value,
                                    "accn": "0001193125-26-000001",
                                    "fy": 2026,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-07-29",
                                }
                            ]
                        }
                    }
                    for source_concept, value in {
                        "Depreciation": 34_300_000_000,
                        "AmortizationOfIntangibleAssets": 4_700_000_000,
                    }.items()
                },
            },
        }
    )

    financials = SecUsGaapNormalizer().normalize(
        facts,
        granularity=Granularity.ANNUAL,
        concepts={FinancialConcept.DEPRECIATION_AND_AMORTIZATION},
    )
    depreciation = find_observation(
        financials,
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
        2026,
        FiscalPeriod.FY,
    )

    assert depreciation.value == Decimal("39000000000")
    assert depreciation.source_concept == (
        "Depreciation + AmortizationOfIntangibleAssets"
    )
    assert depreciation.derivation_kind == (
        ObservationDerivationKind.COMPONENT_AGGREGATION
    )


def test_uses_depreciation_when_no_amortization_fact_is_reported():
    facts = CompanyFacts.model_validate(
        {
            "cik": 1652044,
            "entityName": "Alphabet Inc.",
            "facts": {
                "dei": {},
                "us-gaap": {
                    "Depreciation": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-01-01",
                                    "end": "2025-12-31",
                                    "val": 21_136_000_000,
                                    "accn": "0001652044-26-000018",
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-02-05",
                                }
                            ]
                        }
                    }
                },
            },
        }
    )

    financials = SecUsGaapNormalizer().normalize(
        facts,
        granularity=Granularity.ANNUAL,
        concepts={FinancialConcept.DEPRECIATION_AND_AMORTIZATION},
    )
    depreciation = financials.observations[0]

    assert depreciation.value == Decimal("21136000000")
    assert depreciation.source_concept == "Depreciation"
    assert depreciation.derivation_kind == ObservationDerivationKind.CONCEPT_FALLBACK


def test_recovers_prior_fy_balance_first_disclosed_as_quarterly_comparative():
    facts = CompanyFacts.model_validate(
        {
            "cik": 1652044,
            "entityName": "Alphabet Inc.",
            "facts": {
                "dei": {},
                "us-gaap": {
                    "InventoryNet": {
                        "units": {
                            "USD": [
                                {
                                    "end": end,
                                    "val": value,
                                    "accn": "0001652044-26-000071",
                                    "fy": 2026,
                                    "fp": "Q2",
                                    "form": "10-Q",
                                    "filed": "2026-07-23",
                                }
                                for end, value in (
                                    ("2025-12-31", 2_439_000_000),
                                    ("2026-06-30", 9_991_000_000),
                                )
                            ]
                        }
                    }
                },
            },
        }
    )

    financials = SecUsGaapNormalizer().normalize(
        facts,
        granularity=Granularity.ANNUAL,
        concepts={FinancialConcept.INVENTORY},
    )
    inventory = financials.observations[0]

    assert inventory.fiscal_year == 2025
    assert inventory.fiscal_period == FiscalPeriod.FY
    assert inventory.value == Decimal("2439000000")


def test_does_not_derive_non_additive_weighted_average_shares():
    financials = SecUsGaapNormalizer().normalize(
        load_aapl_facts(),
        granularity=Granularity.QUARTERLY,
        concepts={
            FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES,
            FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES,
        },
    )

    assert financials.observations
    assert all(not observation.is_derived for observation in financials.observations)
    assert all(
        observation.fiscal_period != FiscalPeriod.Q4
        for observation in financials.observations
    )


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
