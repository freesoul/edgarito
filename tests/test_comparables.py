import datetime
from decimal import Decimal
from statistics import median

import pytest

from edgarito.config.valuation import MultipleResolutionConfiguration
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.market import PriceBar, SecurityMarketData
from edgarito.schemas.normalization.classification import Sector
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.forecasting import (
    FcffForecast,
    FcffForecastObservation,
    FcffForecastParameters,
)
from edgarito.services.valuation import (
    BusinessArchetype,
    CompanyLifecycle,
    ComparableImpliedValuationService,
    ComparableMultiplesService,
    Cyclicality,
    FcffDcfCapitalBridge,
    FcffDcfParameters,
    FcffDcfService,
    HistoricalMultipleObservation,
    HistoricalMultipleSummary,
    LtmMultiplesService,
    MultipleResolver,
    MultipleStatus,
    PeerSelectionParameters,
    PeerUniverseSelector,
    RelativeValuationBasis,
    ValuationProfile,
)


def _profile(
    ticker: str,
    *,
    sector: Sector = Sector.TECHNOLOGY,
    industry: str = "Software Infrastructure",
    archetype: BusinessArchetype = BusinessArchetype.GENERAL_OPERATING,
    revenue: str = "1000",
) -> ValuationProfile:
    return ValuationProfile(
        provider="test",
        company_id=ticker,
        company_name=f"{ticker} Inc.",
        ticker=ticker,
        sector=sector,
        industry=industry,
        country="US",
        exchange="NASDAQ",
        reporting_currency="USD",
        latest_revenue=Decimal(revenue),
        business_archetype=archetype,
        lifecycle=CompanyLifecycle.MATURE,
        cyclicality=Cyclicality.LOW,
    )


def test_peer_selector_ranks_economic_fit_and_excludes_sector_mismatch():
    target = _profile("TARGET")
    exact = _profile("EXACT", revenue="1200")
    adjacent = _profile("ADJ", industry="Application Software", revenue="2500")
    bank = _profile(
        "BANK",
        sector=Sector.FINANCIALS,
        industry="Banks Regional",
        archetype=BusinessArchetype.FINANCIAL_INTERMEDIARY,
    )

    universe = PeerUniverseSelector().select(
        target,
        [adjacent, bank, exact],
        PeerSelectionParameters(
            max_peers=2,
            preferred_minimum=2,
            minimum_score=40,
        ),
    )

    assert universe.selected_tickers == ("EXACT", "ADJ")
    assert universe.candidates[0].score > universe.candidates[1].score
    bank_assessment = next(
        item for item in universe.candidates if item.ticker == "BANK"
    )
    assert "Sector differs from the target" in bank_assessment.exclusions


def _observation(
    concept: FinancialConcept,
    value: str,
    fiscal_year: int,
    fiscal_period: FiscalPeriod,
    period_end: datetime.date,
) -> FinancialObservation:
    quarter_start_month = {
        FiscalPeriod.Q1: 1,
        FiscalPeriod.Q2: 4,
        FiscalPeriod.Q3: 7,
    }.get(
        fiscal_period,
        10,
    )
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal(value),
        unit="shares"
        if concept
        in {
            FinancialConcept.SHARES_OUTSTANDING,
            FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES,
        }
        else "USD",
        granularity=Granularity.QUARTERLY,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        period_start=(
            None
            if concept.statement.value == "balance_sheet"
            else datetime.date(period_end.year, quarter_start_month, 1)
        ),
        period_end=period_end,
        provider="test",
        taxonomy="test",
        source_concept=concept.value,
    )


def _financials(*, net_income: str = "12") -> NormalizedCompanyFinancials:
    periods = (
        (2025, FiscalPeriod.Q2, datetime.date(2025, 6, 30)),
        (2025, FiscalPeriod.Q3, datetime.date(2025, 9, 30)),
        (2025, FiscalPeriod.Q4, datetime.date(2025, 12, 31)),
        (2026, FiscalPeriod.Q1, datetime.date(2026, 3, 31)),
    )
    flows = {
        FinancialConcept.REVENUE: "100",
        FinancialConcept.OPERATING_INCOME: "20",
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION: "5",
        FinancialConcept.NET_INCOME: net_income,
        FinancialConcept.OPERATING_CASH_FLOW: "18",
        FinancialConcept.CAPITAL_EXPENDITURES: "6",
        FinancialConcept.DIVIDENDS_PAID: "1",
    }
    observations = [
        _observation(concept, value, fiscal_year, fiscal_period, period_end)
        for fiscal_year, fiscal_period, period_end in periods
        for concept, value in flows.items()
    ]
    balances = {
        FinancialConcept.STOCKHOLDERS_EQUITY: "200",
        FinancialConcept.GOODWILL: "20",
        FinancialConcept.INTANGIBLE_ASSETS_NET: "10",
        FinancialConcept.CASH_AND_EQUIVALENTS: "30",
        FinancialConcept.SHORT_TERM_DEBT: "10",
        FinancialConcept.LONG_TERM_DEBT_CURRENT: "5",
        FinancialConcept.LONG_TERM_DEBT_NONCURRENT: "40",
        FinancialConcept.SHARES_OUTSTANDING: "50",
    }
    observations.extend(
        _observation(
            concept,
            value,
            2026,
            FiscalPeriod.Q1,
            datetime.date(2026, 3, 31),
        )
        for concept, value in balances.items()
    )
    return NormalizedCompanyFinancials(
        provider="test",
        company_id="TARGET",
        company_name="Target Inc.",
        ticker="TARGET",
        observations=observations,
    )


def _market_data(currency: str = "USD") -> SecurityMarketData:
    return SecurityMarketData(
        provider="test-market",
        provider_symbol="TARGET",
        identifiers=SecurityIdentifiers(ticker="TARGET"),
        currency=currency,
        retrieved_at=datetime.datetime(2026, 4, 2, tzinfo=datetime.timezone.utc),
        prices=(PriceBar(observed_on=datetime.date(2026, 4, 1), close=Decimal("10")),),
    )


def test_computes_ltm_fundamentals_enterprise_value_and_multiples():
    result = LtmMultiplesService().compute(_financials(), _market_data())
    multiples = {item.basis: item for item in result.multiples}

    assert result.fundamentals.period_start == datetime.date(2025, 4, 1)
    assert result.fundamentals.period_end == datetime.date(2026, 3, 31)
    assert result.fundamentals.revenue == Decimal("400")
    assert result.fundamentals.ebitda == Decimal("100")
    assert result.fundamentals.free_cash_flow == Decimal("48")
    assert result.market_capitalization == Decimal("500")
    assert result.enterprise_value == Decimal("520")
    assert multiples[RelativeValuationBasis.PE].value == Decimal("500") / Decimal("48")
    assert multiples[RelativeValuationBasis.EV_TO_EBITDA].value == Decimal("5.2")
    assert multiples[RelativeValuationBasis.DIVIDEND_YIELD].value == Decimal("0.8")


def test_marks_negative_denominators_and_currency_mismatch_explicitly():
    loss_result = LtmMultiplesService().compute(
        _financials(net_income="-2"), _market_data()
    )
    pe = next(
        item
        for item in loss_result.multiples
        if item.basis == RelativeValuationBasis.PE
    )
    assert pe.status == MultipleStatus.NOT_MEANINGFUL

    currency_result = LtmMultiplesService().compute(_financials(), _market_data("EUR"))
    assert currency_result.enterprise_value is None
    assert all(
        item.status != MultipleStatus.COMPUTED
        for item in currency_result.multiples
        if item.basis.value.startswith("ev_to")
    )
    assert "FX alignment" in currency_result.warnings[0]


def test_ltm_requires_four_consecutive_revenue_quarters():
    financials = _financials().model_copy(
        update={
            "observations": [
                item
                for item in _financials().observations
                if not (
                    item.concept == FinancialConcept.REVENUE
                    and item.fiscal_period == FiscalPeriod.Q3
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="four consecutive"):
        LtmMultiplesService().compute(financials, _market_data())


def test_latest_annual_fallback_is_explicit_when_quarters_are_unavailable():
    annual_observations = [
        item.model_copy(
            update={
                "granularity": Granularity.ANNUAL,
                "fiscal_period": FiscalPeriod.FY,
                "period_start": (
                    None
                    if item.concept.statement.value == "balance_sheet"
                    else datetime.date(2025, 4, 1)
                ),
            }
        )
        for item in _financials().observations
        if item.fiscal_period == FiscalPeriod.Q1
    ]
    financials = _financials().model_copy(update={"observations": annual_observations})

    result = LtmMultiplesService().compute(financials, _market_data())
    multiple = next(
        item
        for item in result.multiples
        if item.basis == RelativeValuationBasis.EV_TO_EBITDA
    )

    assert multiple.status == MultipleStatus.COMPUTED
    assert any("latest annual" in warning for warning in result.warnings)


def test_comparable_report_aggregates_only_selected_peer_values():
    target_profile = _profile("TARGET")
    peer_profiles = [_profile("PEER1"), _profile("PEER2", revenue="1800")]
    universe = PeerUniverseSelector().select(
        target_profile,
        peer_profiles,
        PeerSelectionParameters(
            max_peers=2,
            preferred_minimum=2,
            minimum_score=0,
        ),
    )
    target = LtmMultiplesService().compute(_financials(), _market_data())
    first = target.model_copy(
        update={
            "ticker": "PEER1",
            "multiples": [
                item.model_copy(update={"value": Decimal("8")})
                if item.status == MultipleStatus.COMPUTED
                else item
                for item in target.multiples
            ],
        }
    )
    second = target.model_copy(
        update={
            "ticker": "PEER2",
            "multiples": [
                item.model_copy(update={"value": Decimal("12")})
                if item.status == MultipleStatus.COMPUTED
                else item
                for item in target.multiples
            ],
        }
    )

    report = ComparableMultiplesService().build(
        universe,
        target,
        [first, second],
    )
    pe = next(
        item for item in report.summaries if item.basis == RelativeValuationBasis.PE
    )

    assert pe.median == Decimal("10")
    assert pe.minimum == Decimal("8")
    assert pe.maximum == Decimal("12")
    assert pe.sample_size == 2


def test_multiple_resolver_keeps_fundamental_anchor_and_premium_separate():
    target_profile = _profile("TARGET")
    peer_profiles = [_profile("PEER1"), _profile("PEER2")]
    target = LtmMultiplesService().compute(_financials(), _market_data())
    peers = [
        target.model_copy(
            update={
                "ticker": f"PEER{index}",
                "multiples": [
                    item.model_copy(update={"value": value})
                    if item.basis == RelativeValuationBasis.EV_TO_EBITDA
                    else item
                    for item in target.multiples
                ],
            }
        )
        for index, value in enumerate((Decimal("8"), Decimal("12")), start=1)
    ]
    universe = PeerUniverseSelector().select(
        target_profile,
        peer_profiles,
        PeerSelectionParameters(
            max_peers=2,
            preferred_minimum=2,
            minimum_score=0,
        ),
    )
    report = ComparableMultiplesService().build(universe, target, peers)
    forecast = _relative_forecast()
    bridge = FcffDcfCapitalBridge(
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
        net_debt=Decimal("20"),
        diluted_shares=Decimal("10"),
        net_debt_source="test",
        shares_source="test",
    )
    intrinsic = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc="10", perpetual_growth_rate="2"),
        bridge,
    )
    history = HistoricalMultipleSummary(
        basis=RelativeValuationBasis.EV_TO_EBITDA,
        observations=tuple(
            HistoricalMultipleObservation(
                observed_on=datetime.date(2022 + index, 12, 31), value=value
            )
            for index, value in enumerate(
                (Decimal("11"), Decimal("12"), Decimal("13"), Decimal("14"))
            )
        ),
        median=Decimal("12.5"),
        volatility=Decimal("0.1"),
    )
    peer_histories = tuple(
        HistoricalMultipleSummary(
            basis=RelativeValuationBasis.EV_TO_EBITDA,
            observations=tuple(
                HistoricalMultipleObservation(
                    observed_on=datetime.date(2022 + index, 12, 31), value=value
                )
                for index, value in enumerate(values)
            ),
            median=median(values),
        )
        for values in (
            (Decimal("8"), Decimal("8.5"), Decimal("9"), Decimal("9.5")),
            (Decimal("9"), Decimal("9"), Decimal("9.5"), Decimal("10")),
        )
    )

    resolved = MultipleResolver().resolve(
        basis=RelativeValuationBasis.EV_TO_EBITDA,
        target=target,
        target_history=history,
        peer_histories=peer_histories,
        peer_report=report,
        target_forecast=forecast,
        intrinsic_valuation=intrinsic,
        horizon_years=Decimal(1),
        policy=MultipleResolutionConfiguration(minimum_peer_sample=2),
    )
    implied = ComparableImpliedValuationService().value(
        target_forecast=forecast,
        capital_bridge=bridge,
        projected_shares=bridge.diluted_shares,
        resolved_multiple=resolved,
        valuation_date=datetime.date(2025, 12, 31),
        horizon_years=Decimal(1),
        discount_rate=Decimal("10"),
        current_price=Decimal("18"),
        analyst_target_price=Decimal("20"),
        intrinsic_value_per_share=intrinsic.value_per_share,
    )

    assert resolved.fundamental_anchor == (
        intrinsic.terminal_value.terminal_value / Decimal("125")
    )
    assert resolved.peer_anchor == Decimal("10")
    assert resolved.historical_anchor == Decimal("12.5")
    assert resolved.lower_bound <= resolved.point_estimate <= resolved.upper_bound
    assert resolved.persistence_factor < Decimal(1)
    assert resolved.premium_history_sample_size == 4
    assert resolved.premium_mean_reversion_beta is not None
    assert implied.point_case.implied_enterprise_value == (
        implied.forecast_metric * resolved.point_estimate
    )
    assert implied.point_case.present_value_per_share < (
        implied.point_case.implied_value_per_share
    )
    assert implied.analyst_target_implied_multiple == Decimal("1.76")
    assert implied.current_price_implied_multiple == Decimal("1.6")


def _relative_forecast() -> FcffForecast:
    observation = FcffForecastObservation(
        forecast_year=1,
        fiscal_year=2026,
        period_end=datetime.date(2026, 12, 31),
        revenue_growth=Decimal("5"),
        revenue=Decimal("1000"),
        operating_margin=Decimal("10"),
        operating_income=Decimal("100"),
        tax_rate=Decimal("20"),
        nopat=Decimal("80"),
        depreciation_to_revenue=Decimal("2.5"),
        depreciation_and_amortization=Decimal("25"),
        capex_to_revenue=Decimal("3"),
        capital_expenditures=Decimal("30"),
        operating_working_capital_to_revenue=Decimal("10"),
        operating_working_capital=Decimal("100"),
        change_in_operating_working_capital=Decimal("5"),
        fcff=Decimal("70"),
        unit="USD",
    )
    return FcffForecast(
        provider="test",
        company_id="TARGET",
        company_name="Target Inc.",
        ticker="TARGET",
        base_fiscal_year=2025,
        base_period_end=datetime.date(2025, 12, 31),
        base_revenue=Decimal("950"),
        base_operating_income=Decimal("95"),
        base_tax_rate=Decimal("20"),
        base_nopat=Decimal("76"),
        base_depreciation_and_amortization=Decimal("24"),
        base_capital_expenditures=Decimal("29"),
        base_operating_working_capital=Decimal("95"),
        base_fcff=Decimal("66"),
        unit="USD",
        parameters=FcffForecastParameters(forecast_years=1),
        historical_fiscal_years=(2024, 2025),
        assumption_sources={},
        observations=[observation],
    )
