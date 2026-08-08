import datetime
from decimal import Decimal

from edgarito.config.red_flags import (
    AccountingQualityConfiguration,
    AcquisitionsConfiguration,
    CashConversionConfiguration,
    ConcentrationConfiguration,
    DebtConfiguration,
    DilutionSbcConfiguration,
    FcfVsEarningsConfiguration,
    MarginsGrowthConfiguration,
    RedFlagsConfiguration,
    RedFlagsProfileLoader,
    RoicConfiguration,
)
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.red_flags import RedFlagCategory
from edgarito.services.red_flags import InvestmentRedFlagsService


def _observation(
    concept: FinancialConcept, value: str, fiscal_year: int
) -> FinancialObservation:
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal(value),
        unit="shares" if "shares" in concept.value else "USD",
        granularity=Granularity.ANNUAL,
        fiscal_year=fiscal_year,
        fiscal_period=FiscalPeriod.FY,
        period_end=datetime.date(fiscal_year, 12, 31),
        provider="test",
        taxonomy="test",
        source_concept=concept.value,
    )


def _financials(values_by_year: dict[int, dict[FinancialConcept, str]]):
    return NormalizedCompanyFinancials(
        provider="test",
        company_id="1",
        company_name="Test Company",
        ticker="TEST",
        observations=[
            _observation(concept, value, year)
            for year, values in values_by_year.items()
            for concept, value in values.items()
        ],
    )


def _disabled_configuration(**overrides) -> RedFlagsConfiguration:
    categories = {
        "fcf_vs_earnings": FcfVsEarningsConfiguration(enabled=False),
        "debt": DebtConfiguration(enabled=False),
        "dilution_sbc": DilutionSbcConfiguration(enabled=False),
        "acquisitions": AcquisitionsConfiguration(enabled=False),
        "margins_growth": MarginsGrowthConfiguration(enabled=False),
        "roic": RoicConfiguration(enabled=False),
        "cash_conversion": CashConversionConfiguration(enabled=False),
        "concentration": ConcentrationConfiguration(enabled=False),
        "accounting_quality": AccountingQualityConfiguration(enabled=False),
    }
    categories.update(overrides)
    return RedFlagsConfiguration(**categories)


def test_red_flags_loader_reads_packaged_typed_default():
    configuration = RedFlagsProfileLoader.load()

    assert configuration.name == "default"
    assert configuration.fcf_vs_earnings.minimum_fcf_to_net_income_pct == Decimal("80")
    assert RedFlagCategory.CONCENTRATION in configuration.enabled_categories


def test_fcf_debt_and_cash_conversion_flags_retain_deterministic_evidence():
    financials = _financials(
        {
            2023: {
                FinancialConcept.NET_INCOME: "100",
                FinancialConcept.OPERATING_CASH_FLOW: "90",
                FinancialConcept.CAPITAL_EXPENDITURES: "10",
                FinancialConcept.SHORT_TERM_DEBT: "20",
                FinancialConcept.LONG_TERM_DEBT_NONCURRENT: "180",
                FinancialConcept.CASH_AND_EQUIVALENTS: "10",
                FinancialConcept.OPERATING_INCOME: "40",
                FinancialConcept.DEPRECIATION_AND_AMORTIZATION: "10",
                FinancialConcept.INTEREST_EXPENSE: "20",
            },
            2024: {
                FinancialConcept.NET_INCOME: "100",
                FinancialConcept.OPERATING_CASH_FLOW: "50",
                FinancialConcept.CAPITAL_EXPENDITURES: "30",
                FinancialConcept.SHORT_TERM_DEBT: "20",
                FinancialConcept.LONG_TERM_DEBT_NONCURRENT: "180",
                FinancialConcept.CASH_AND_EQUIVALENTS: "10",
                FinancialConcept.OPERATING_INCOME: "40",
                FinancialConcept.DEPRECIATION_AND_AMORTIZATION: "10",
                FinancialConcept.INTEREST_EXPENSE: "20",
            },
        }
    )
    configuration = _disabled_configuration(
        fcf_vs_earnings=FcfVsEarningsConfiguration(
            minimum_fcf_to_net_income_pct=Decimal("80")
        ),
        debt=DebtConfiguration(
            maximum_net_debt_to_ebitda=Decimal("2"),
            minimum_interest_coverage=Decimal("3"),
        ),
        cash_conversion=CashConversionConfiguration(
            minimum_operating_cash_flow_to_net_income_pct=Decimal("80")
        ),
    )

    report = InvestmentRedFlagsService(configuration).analyze(financials)

    assert [flag.code for flag in report.flags] == [
        "interest_coverage_low",
        "net_debt_to_ebitda_high",
        "fcf_below_earnings",
        "interest_coverage_low",
        "net_debt_to_ebitda_high",
        "cash_conversion_low",
    ]
    evidence = next(
        flag.evidence[0] for flag in report.flags if flag.code == "fcf_below_earnings"
    )
    assert evidence.value == Decimal("20")
    assert evidence.threshold == Decimal("80")
    assert evidence.input_concepts == (
        FinancialConcept.CAPITAL_EXPENDITURES,
        FinancialConcept.NET_INCOME,
        FinancialConcept.OPERATING_CASH_FLOW,
    )


def test_red_flag_messages_round_numeric_values_without_changing_evidence_precision():
    financials = _financials(
        {
            2024: {
                FinancialConcept.NET_INCOME: "100",
                FinancialConcept.OPERATING_CASH_FLOW: "50.123456",
                FinancialConcept.CAPITAL_EXPENDITURES: "0",
            }
        }
    )
    configuration = _disabled_configuration(
        fcf_vs_earnings=FcfVsEarningsConfiguration(
            minimum_fcf_to_net_income_pct=Decimal("80.123456")
        )
    )

    report = InvestmentRedFlagsService(configuration).analyze(financials)

    flag = report.flags[0]
    assert flag.message == (
        "Free cash flow converted 50.12% of net income, "
        "below the configured 80.12% floor."
    )
    assert flag.evidence[0].value == Decimal("50.123456")
    assert flag.evidence[0].threshold == Decimal("80.123456")


def test_dilution_sbc_and_acquisition_rules_are_configurable():
    financials = _financials(
        {
            2023: {
                FinancialConcept.REVENUE: "100",
                FinancialConcept.SHARES_OUTSTANDING: "100",
                FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES: "100",
                FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES: "105",
                FinancialConcept.STOCK_BASED_COMPENSATION: "4",
                FinancialConcept.ACQUISITION_CASH_PAID: "5",
                FinancialConcept.GOODWILL: "10",
                FinancialConcept.OPERATING_CASH_FLOW: "30",
                FinancialConcept.CAPITAL_EXPENDITURES: "10",
            },
            2024: {
                FinancialConcept.REVENUE: "100",
                FinancialConcept.SHARES_OUTSTANDING: "110",
                FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES: "100",
                FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES: "115",
                FinancialConcept.STOCK_BASED_COMPENSATION: "8",
                FinancialConcept.ACQUISITION_CASH_PAID: "20",
                FinancialConcept.GOODWILL: "20",
                FinancialConcept.OPERATING_CASH_FLOW: "30",
                FinancialConcept.CAPITAL_EXPENDITURES: "10",
            },
        }
    )
    configuration = _disabled_configuration(
        dilution_sbc=DilutionSbcConfiguration(
            maximum_share_count_growth_pct=Decimal("3"),
            maximum_diluted_share_premium_pct=Decimal("5"),
            maximum_sbc_to_revenue_pct=Decimal("5"),
        ),
        acquisitions=AcquisitionsConfiguration(
            maximum_acquisition_to_revenue_pct=Decimal("10"),
            maximum_acquisition_to_fcf_pct=Decimal("50"),
            maximum_goodwill_growth_pct=Decimal("15"),
        ),
    )

    report = InvestmentRedFlagsService(configuration).analyze(financials)

    assert {flag.code for flag in report.flags} == {
        "share_count_growth_high",
        "diluted_share_premium_high",
        "stock_based_compensation_high",
        "acquisition_spend_high",
        "acquisition_spend_exceeds_fcf",
        "goodwill_growth_high",
    }


def test_concentration_and_missing_inputs_are_warnings_not_clean_results():
    financials = _financials(
        {2024: {FinancialConcept.REVENUE: "100"}}
    )
    report = InvestmentRedFlagsService(
        RedFlagsConfiguration(history_periods=2)
    ).analyze(financials)

    warning_codes = {warning.code for warning in report.warnings}
    assert "concentration_data_unavailable" in warning_codes
    assert "fcf_vs_earnings_unavailable" in warning_codes
    assert report.flags == ()
    assert report.data_complete is False
    assert report.is_clean is False


def test_margins_roic_and_accounting_rules_cover_trends():
    financials = _financials(
        {
            2023: {
                FinancialConcept.REVENUE: "100",
                FinancialConcept.OPERATING_INCOME: "30",
                FinancialConcept.PRETAX_INCOME: "25",
                FinancialConcept.INCOME_TAX_EXPENSE: "5",
                FinancialConcept.STOCKHOLDERS_EQUITY: "100",
                FinancialConcept.LONG_TERM_DEBT_NONCURRENT: "20",
                FinancialConcept.CASH_AND_EQUIVALENTS: "10",
                FinancialConcept.ACCOUNTS_RECEIVABLE: "10",
                FinancialConcept.INVENTORY: "10",
                FinancialConcept.GOODWILL: "10",
                FinancialConcept.TOTAL_ASSETS: "150",
            },
            2024: {
                FinancialConcept.REVENUE: "110",
                FinancialConcept.OPERATING_INCOME: "20",
                FinancialConcept.PRETAX_INCOME: "15",
                FinancialConcept.INCOME_TAX_EXPENSE: "3",
                FinancialConcept.STOCKHOLDERS_EQUITY: "100",
                FinancialConcept.LONG_TERM_DEBT_NONCURRENT: "20",
                FinancialConcept.CASH_AND_EQUIVALENTS: "10",
                FinancialConcept.ACCOUNTS_RECEIVABLE: "30",
                FinancialConcept.INVENTORY: "30",
                FinancialConcept.GOODWILL: "90",
                FinancialConcept.TOTAL_ASSETS: "150",
            },
        }
    )
    configuration = _disabled_configuration(
        margins_growth=MarginsGrowthConfiguration(
            minimum_revenue_growth_pct=Decimal("15"),
            maximum_operating_margin_decline_pp=Decimal("3"),
        ),
        roic=RoicConfiguration(
            minimum_roic_pct=Decimal("20"), maximum_roic_decline_pp=Decimal("3")
        ),
        accounting_quality=AccountingQualityConfiguration(
            maximum_receivables_growth_premium_pp=Decimal("10"),
            maximum_inventory_growth_premium_pp=Decimal("10"),
            maximum_goodwill_to_assets_pct=Decimal("50"),
        ),
    )

    report = InvestmentRedFlagsService(configuration).analyze(financials)

    assert {flag.code for flag in report.flags} == {
        "revenue_growth_low",
        "operating_margin_decline",
        "roic_low",
        "roic_decline",
        "receivables_growth_ahead_of_revenue",
        "inventory_growth_ahead_of_revenue",
        "goodwill_to_assets_high",
    }
