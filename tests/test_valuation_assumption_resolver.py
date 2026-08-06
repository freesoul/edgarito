import datetime
from decimal import Decimal

from edgarito.config.valuation import (
    DiscountRateConfiguration,
    TerminalValueConfiguration,
)
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.market import (
    MarketDataFrequency,
    PriceBar,
    ReferenceMarketSeries,
    ReferenceObservation,
    ReferenceSeriesKind,
    ReferenceValueUnit,
    SecurityMarketData,
)
from edgarito.schemas.normalization.classification import (
    NormalizedCompanyClassification,
    Sector,
)
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.valuation.assumptions import ValuationAssumptionKind
from edgarito.schemas.valuation.reference import (
    CountryRiskPremium,
    CountryRiskPremiumSnapshot,
    IndustryBeta,
    IndustryBetaSnapshot,
    ReferenceDatasetMetadata,
)
from edgarito.services.valuation import (
    FcffDcfCapitalBridge,
    ValuationAssumptionResolver,
)

TODAY = datetime.date(2026, 8, 6)
RETRIEVED = datetime.datetime(2026, 8, 6, tzinfo=datetime.timezone.utc)


def test_explicit_dcf_assumptions_do_not_require_external_data():
    result = ValuationAssumptionResolver().resolve(
        financials=_financials(),
        capital_bridge=_bridge(),
        discount_configuration=DiscountRateConfiguration(wacc="8"),
        terminal_configuration=TerminalValueConfiguration(perpetual_growth_rate="2"),
        terminal_is_perpetuity=True,
        valuation_date=TODAY,
    )

    assert result.wacc == Decimal("8")
    assert result.perpetual_growth_rate == Decimal("2")
    assert [item.kind for item in result.assumption_set.assumptions] == [
        ValuationAssumptionKind.WACC,
        ValuationAssumptionKind.TERMINAL_GROWTH,
    ]


def test_resolver_derives_wacc_and_terminal_growth_from_provider_inputs():
    result = ValuationAssumptionResolver().resolve(
        financials=_financials(),
        capital_bridge=_bridge(),
        discount_configuration=DiscountRateConfiguration(),
        terminal_configuration=TerminalValueConfiguration(),
        terminal_is_perpetuity=True,
        valuation_date=TODAY,
        classification=_classification(),
        market_data=_market_data(),
        risk_free_series=_reference_series(
            "ECB 10-year AAA yield", ReferenceSeriesKind.GOVERNMENT_YIELD, ["3"]
        ),
        inflation_series=_reference_series(
            "ECB HICP", ReferenceSeriesKind.INFLATION_RATE, ["2.0", "2.2", "2.4"]
        ),
        country_snapshot=_country_snapshot(),
        industry_snapshot=_industry_snapshot(),
    )

    assert Decimal("6") < result.wacc < Decimal("7")
    assert result.perpetual_growth_rate == Decimal("2.2")
    assert "ecb" in result.wacc_source
    assert "damodaran" in result.wacc_source
    assert "yahoo" in result.wacc_source
    assumptions = {item.kind: item for item in result.assumption_set.assumptions}
    assert assumptions[ValuationAssumptionKind.NORMALIZED_TAX_RATE].value == Decimal(
        "25"
    )
    assert assumptions[ValuationAssumptionKind.PRETAX_COST_OF_DEBT].value == Decimal(
        "5"
    )
    assert assumptions[ValuationAssumptionKind.UNLEVERED_BETA].value == Decimal("0.85")


def test_luxury_goods_uses_damodaran_apparel_beta_proxy():
    classification = _classification().model_copy(update={"industry": "Luxury Goods"})
    snapshot = _industry_snapshot().model_copy(
        update={
            "industries": (
                _industry_snapshot().industries[0].model_copy(
                    update={"industry": "Apparel"}
                ),
            )
        }
    )

    result = ValuationAssumptionResolver().resolve(
        financials=_financials(),
        capital_bridge=_bridge(),
        discount_configuration=DiscountRateConfiguration(),
        terminal_configuration=TerminalValueConfiguration(perpetual_growth_rate="2"),
        terminal_is_perpetuity=True,
        valuation_date=TODAY,
        classification=classification,
        market_data=_market_data(),
        risk_free_series=_reference_series(
            "ECB 10-year AAA yield", ReferenceSeriesKind.GOVERNMENT_YIELD, ["3"]
        ),
        country_snapshot=_country_snapshot(),
        industry_snapshot=snapshot,
    )

    beta = result.assumption_set.find(ValuationAssumptionKind.UNLEVERED_BETA)
    assert beta is not None
    assert beta.value == Decimal("0.85")
    assert beta.industry == "Apparel"


def test_country_premium_prefers_sovereign_cds_total_erp():
    country_snapshot = _country_snapshot().model_copy(
        update={
            "countries": (
                _country_snapshot().countries[0].model_copy(
                    update={"cds_equity_risk_premium": Decimal("4.94")}
                ),
            )
        }
    )
    result = ValuationAssumptionResolver().resolve(
        financials=_financials(),
        capital_bridge=_bridge(),
        discount_configuration=DiscountRateConfiguration(),
        terminal_configuration=TerminalValueConfiguration(perpetual_growth_rate="2"),
        terminal_is_perpetuity=True,
        valuation_date=TODAY,
        classification=_classification(),
        market_data=_market_data(),
        risk_free_series=_reference_series(
            "ECB 10-year AAA yield", ReferenceSeriesKind.GOVERNMENT_YIELD, ["3"]
        ),
        country_snapshot=country_snapshot,
        industry_snapshot=_industry_snapshot(),
    )

    premium = result.assumption_set.find(
        ValuationAssumptionKind.COUNTRY_RISK_PREMIUM
    )
    assert premium is not None
    assert premium.value == Decimal("0.71")
    assert premium.provenance.methodology == (
        "Sovereign-CDS total ERP minus mature-market ERP"
    )


def _financials():
    return NormalizedCompanyFinancials(
        provider="test",
        company_id="EX",
        company_name="Example",
        ticker="EX.DE",
        observations=[
            _observation(FinancialConcept.PRETAX_INCOME, "100"),
            _observation(FinancialConcept.INCOME_TAX_EXPENSE, "25"),
            _observation(FinancialConcept.INTEREST_EXPENSE, "5"),
        ],
    )


def _observation(concept, value):
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal(value),
        unit="EUR",
        granularity=Granularity.ANNUAL,
        fiscal_year=2025,
        fiscal_period=FiscalPeriod.FY,
        period_start=datetime.date(2025, 1, 1),
        period_end=datetime.date(2025, 12, 31),
        provider="test",
        taxonomy="test",
        source_concept=concept.value,
    )


def _bridge():
    return FcffDcfCapitalBridge(
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="EUR",
        net_debt=Decimal("80"),
        diluted_shares=Decimal("10"),
        net_debt_source="test",
        shares_source="test",
        gross_debt=Decimal("100"),
        cash_and_equivalents=Decimal("20"),
    )


def _classification():
    return NormalizedCompanyClassification(
        provider="yahoo",
        company_id="EX",
        company_name="Example",
        ticker="EX.DE",
        sector=Sector.INDUSTRIALS,
        industry="Aerospace & Defense",
        industry_taxonomy="yahoo-profile",
        country="Germany",
    )


def _market_data():
    return SecurityMarketData(
        provider="yahoo",
        provider_symbol="EX.DE",
        identifiers=SecurityIdentifiers(ticker="EX.DE"),
        currency="EUR",
        frequency=MarketDataFrequency.DAILY,
        retrieved_at=RETRIEVED,
        prices=(PriceBar(observed_on=TODAY, close=Decimal("100")),),
    )


def _reference_series(name, kind, values):
    observations = tuple(
        ReferenceObservation(
            period_end=datetime.date(2026, month, 1), value=Decimal(value)
        )
        for month, value in enumerate(values, start=1)
    )
    return ReferenceMarketSeries(
        provider="ecb",
        series_id=name.lower().replace(" ", "-"),
        name=name,
        kind=kind,
        unit=ReferenceValueUnit.PERCENTAGE_POINTS,
        frequency=MarketDataFrequency.MONTHLY,
        retrieved_at=RETRIEVED,
        observations=observations,
        currency="EUR",
    )


def _metadata(dataset):
    return ReferenceDatasetMetadata(
        provider="damodaran",
        dataset=dataset,
        version="2026",
        published_on=datetime.date(2026, 1, 5),
        retrieved_at=RETRIEVED,
        source_url="https://example.com/reference",
        sha256="a" * 64,
    )


def _country_snapshot():
    return CountryRiskPremiumSnapshot(
        metadata=_metadata("country-risk-premiums"),
        countries=(
            CountryRiskPremium(
                country="Germany",
                adjusted_default_spread=Decimal("0"),
                country_risk_premium=Decimal("0"),
                equity_risk_premium=Decimal("4.23"),
                corporate_tax_rate=Decimal("25"),
            ),
        ),
    )


def _industry_snapshot():
    return IndustryBetaSnapshot(
        metadata=_metadata("industry-betas"),
        industries=(
            IndustryBeta(
                industry="Aerospace/Defense",
                number_of_firms=20,
                levered_beta=Decimal("0.95"),
                debt_to_equity=Decimal("10"),
                effective_tax_rate=Decimal("20"),
                unlevered_beta=Decimal("0.85"),
                cash_to_firm_value=Decimal("5"),
                cash_adjusted_unlevered_beta=Decimal("0.9"),
            ),
        ),
    )
