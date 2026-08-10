import asyncio
import datetime
from decimal import Decimal

import pytest

from edgarito.config.valuation import MultistageValuationConfiguration
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.classification import Sector
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.providers.yahoo.fundamentals import YahooCompanyFinancials
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.forecasting import (
    AdaptiveMultistageFcffForecastService,
    FcffForecastParameters,
    FcffForecastService,
    ForecastSeedType,
)
from edgarito.services.valuation import (
    BusinessArchetype,
    CompanyLifecycle,
    Cyclicality,
    FcffDcfCapitalBridge,
    FcffDcfCapitalBridgeResolver,
    FcffDcfParameters,
    FcffDcfService,
    PeerDiscoveryResult,
    PeerSelectionParameters,
    PeerUniverseSelector,
    TerminalRoicResolver,
    ValuationProfile,
    YahooScreenerPeerDiscoveryProvider,
)

VALUATION_DATE = datetime.date(2026, 8, 7)


def _observation(
    concept,
    value,
    fiscal_year,
    period_end,
    *,
    granularity=Granularity.ANNUAL,
    fiscal_period=FiscalPeriod.FY,
    filed=None,
    provider="test",
):
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal(str(value)),
        unit=(
            "shares"
            if concept
            in {
                FinancialConcept.SHARES_OUTSTANDING,
                FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES,
            }
            else "USD"
        ),
        granularity=granularity,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        period_end=period_end,
        filed=filed,
        provider=provider,
        taxonomy="test",
        source_concept=concept.value,
    )


def _roic_financials(ticker, roics):
    observations = []
    for year, roic in zip(range(2021, 2021 + len(roics)), roics, strict=True):
        end = datetime.date(year, 12, 31)
        # Invested capital is 90 equity + 20 debt - 10 cash = 100.
        values = {
            FinancialConcept.OPERATING_INCOME: Decimal(str(roic)) / Decimal("0.75"),
            FinancialConcept.PRETAX_INCOME: "20",
            FinancialConcept.INCOME_TAX_EXPENSE: "5",
            FinancialConcept.STOCKHOLDERS_EQUITY: "90",
            FinancialConcept.SHORT_TERM_DEBT: "5",
            FinancialConcept.LONG_TERM_DEBT_NONCURRENT: "15",
            FinancialConcept.CASH_AND_EQUIVALENTS: "10",
        }
        observations.extend(
            _observation(concept, value, year, end) for concept, value in values.items()
        )
    return NormalizedCompanyFinancials(
        provider="test",
        company_id=ticker,
        company_name=ticker,
        ticker=ticker,
        observations=observations,
    )


def test_terminal_roic_rewards_durable_returns_but_mean_reverts_a_peak():
    resolver = TerminalRoicResolver()
    durable = resolver.resolve(
        _roic_financials("RACE", [28, 30, 31, 30, 32]),
        wacc=Decimal("8"),
        terminal_growth=Decimal("2.5"),
        valuation_date=VALUATION_DATE,
        currency="USD",
        lifecycle=CompanyLifecycle.MATURE,
        cyclicality=Cyclicality.LOW,
    )
    average = resolver.resolve(
        _roic_financials("STABLE", [9, 10, 9, 10, 9]),
        wacc=Decimal("8"),
        terminal_growth=Decimal("2.5"),
        valuation_date=VALUATION_DATE,
        currency="USD",
        lifecycle=CompanyLifecycle.MATURE,
        cyclicality=Cyclicality.LOW,
    )
    temporary = resolver.resolve(
        _roic_financials("PEAK", [9, 10, 10, 11, 50]),
        wacc=Decimal("8"),
        terminal_growth=Decimal("2.5"),
        valuation_date=VALUATION_DATE,
        currency="USD",
        lifecycle=CompanyLifecycle.MATURE,
        cyclicality=Cyclicality.HIGH,
    )

    assert durable.value > average.value
    assert temporary.value < durable.value
    assert temporary.normalized_roic is not None
    assert abs(temporary.normalized_roic - Decimal("10")) < Decimal("1e-20")
    assert any("peak" in warning for warning in temporary.warnings)
    assert durable.confidence == "high"
    assert all(item.value > Decimal("2.5") for item in (durable, average, temporary))


def test_explicit_terminal_roic_overrides_automatic_evidence():
    result = TerminalRoicResolver().resolve(
        _roic_financials("MSFT", [35, 38, 40, 42]),
        wacc=Decimal("8"),
        terminal_growth=Decimal("2.5"),
        valuation_date=VALUATION_DATE,
        currency="USD",
        explicit_roic=Decimal("24"),
        explicit_source="explicit CLI override",
    )

    assert result.value == Decimal("24")
    assert result.source == "explicit CLI override"
    assert result.confidence == "high"


def test_reliable_peer_roic_evidence_is_blended_without_replacing_company_history():
    financials = _roic_financials("DURABLE", [18, 19, 20, 19, 20])
    resolver = TerminalRoicResolver()
    standalone = resolver.resolve(
        financials,
        wacc=Decimal("8"),
        terminal_growth=Decimal("2.5"),
        valuation_date=VALUATION_DATE,
        currency="USD",
    )
    peer_aware = resolver.resolve(
        financials,
        wacc=Decimal("8"),
        terminal_growth=Decimal("2.5"),
        valuation_date=VALUATION_DATE,
        currency="USD",
        peer_roics=(Decimal("10"), Decimal("11"), Decimal("12")),
    )

    assert peer_aware.value < standalone.value
    assert peer_aware.normalized_roic == standalone.normalized_roic
    assert "blended with peer ROIC" in peer_aware.source


def test_higher_sustainable_roic_requires_less_reinvestment_and_raises_fcff():
    financials = _forecast_financials()
    parameters = _explicit_forecast_parameters().model_copy(
        update={"forecast_years": 5}
    )
    base_service = FcffForecastService()
    seed = base_service.forecast(financials, parameters)
    adaptive = AdaptiveMultistageFcffForecastService(base_service)
    results = []
    for terminal_roic in (Decimal("10"), Decimal("20")):
        forecast, plan = adaptive.forecast(
            financials,
            seed,
            parameters,
            Decimal("2"),
            MultistageValuationConfiguration(
                terminal_return_on_invested_capital=terminal_roic
            ),
        )
        results.append((forecast, plan))

    low, high = results
    assert high[1].terminal_reinvestment_rate < low[1].terminal_reinvestment_rate
    assert high[0].observations[-1].fcff > low[0].observations[-1].fcff
    assert high[1].terminal_reinvestment_rate == Decimal("10")


@pytest.mark.parametrize(
    ("ticker", "roics", "cyclicality"),
    [
        ("RACE", [28, 30, 31, 30, 32], Cyclicality.LOW),
        ("MSFT", [32, 35, 38, 40, 39], Cyclicality.LOW),
        ("MATURE", [10, 10, 11, 10, 10], Cyclicality.LOW),
        ("CYCLICAL", [4, 18, 6, 20, 8], Cyclicality.HIGH),
        ("CAPITAL", [7, 8, 8, 9, 8], Cyclicality.MODERATE),
        ("LEVERAGED", [8, 9, 7, 8, 9], Cyclicality.MODERATE),
    ],
)
def test_cross_company_terminal_roic_regressions_are_bounded_and_auditable(
    ticker, roics, cyclicality
):
    result = TerminalRoicResolver().resolve(
        _roic_financials(ticker, roics),
        wacc=Decimal("8"),
        terminal_growth=Decimal("2.5"),
        valuation_date=VALUATION_DATE,
        currency="USD",
        lifecycle=CompanyLifecycle.MATURE,
        cyclicality=cyclicality,
    )

    assert Decimal("3") <= result.value <= Decimal("60")
    assert result.assumption.rationale
    assert "persistence" in result.methodology
    assert result.value.is_finite()


_FLOW_VALUES = {
    FinancialConcept.REVENUE: "25",
    FinancialConcept.OPERATING_INCOME: "5",
    FinancialConcept.PRETAX_INCOME: "4",
    FinancialConcept.INCOME_TAX_EXPENSE: "1",
    FinancialConcept.DEPRECIATION_AND_AMORTIZATION: "1",
    FinancialConcept.CAPITAL_EXPENDITURES: "2",
    FinancialConcept.ACCOUNTS_RECEIVABLE: "6",
    FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: "2",
    FinancialConcept.ACCOUNTS_PAYABLE: "3",
    FinancialConcept.ACCRUED_LIABILITIES: "1",
    FinancialConcept.DEFERRED_REVENUE_CURRENT: "1",
}


def _forecast_financials(*, fiscal_end_month=12):
    annual_ends = (
        datetime.date(2024, fiscal_end_month, 30 if fiscal_end_month == 6 else 31),
        datetime.date(2025, fiscal_end_month, 30 if fiscal_end_month == 6 else 31),
    )
    observations = []
    for year, end, revenue in (
        (2024, annual_ends[0], "80"),
        (2025, annual_ends[1], "100"),
    ):
        values = {**_FLOW_VALUES, FinancialConcept.REVENUE: revenue}
        values[FinancialConcept.OPERATING_INCOME] = Decimal(revenue) * Decimal("0.2")
        values[FinancialConcept.PRETAX_INCOME] = Decimal(revenue) * Decimal("0.18")
        values[FinancialConcept.INCOME_TAX_EXPENSE] = Decimal(revenue) * Decimal(
            "0.045"
        )
        values[FinancialConcept.DEPRECIATION_AND_AMORTIZATION] = Decimal(
            revenue
        ) * Decimal("0.04")
        values[FinancialConcept.CAPITAL_EXPENDITURES] = Decimal(revenue) * Decimal(
            "0.08"
        )
        observations.extend(_observation(c, v, year, end) for c, v in values.items())

    if fiscal_end_month == 12:
        quarters = (
            (2025, FiscalPeriod.Q3, datetime.date(2025, 9, 30)),
            (2025, FiscalPeriod.Q4, datetime.date(2025, 12, 31)),
            (2026, FiscalPeriod.Q1, datetime.date(2026, 3, 31)),
            (2026, FiscalPeriod.Q2, datetime.date(2026, 6, 30)),
        )
    else:
        quarters = (
            (2025, FiscalPeriod.Q3, datetime.date(2025, 3, 31)),
            (2025, FiscalPeriod.Q4, datetime.date(2025, 6, 30)),
            (2026, FiscalPeriod.Q1, datetime.date(2025, 9, 30)),
            (2026, FiscalPeriod.Q2, datetime.date(2025, 12, 31)),
        )
    observations.extend(
        _observation(
            concept,
            value,
            fiscal_year,
            end,
            granularity=Granularity.QUARTERLY,
            fiscal_period=period,
        )
        for fiscal_year, period, end in quarters
        for concept, value in _FLOW_VALUES.items()
    )
    return NormalizedCompanyFinancials(
        provider="test",
        company_id="TTM",
        company_name="TTM Test",
        ticker="TTM",
        observations=observations,
    )


def _explicit_forecast_parameters():
    return FcffForecastParameters(
        forecast_years=2,
        revenue_growth=Decimal("10"),
        operating_margin=Decimal("20"),
        tax_rate=Decimal("25"),
        depreciation_to_revenue=Decimal("4"),
        capex_to_revenue=Decimal("8"),
        operating_working_capital_to_revenue=Decimal("5"),
    )


def test_current_ytd_and_ttm_context_seed_the_current_fiscal_year_without_overlap():
    financials = _forecast_financials()
    forecast = FcffForecastService().forecast(
        financials, _explicit_forecast_parameters()
    )

    assert forecast.seed_type == ForecastSeedType.YTD_PLUS_FORECAST
    assert forecast.actual_quarters == 2
    assert forecast.seed_period_end == datetime.date(2026, 6, 30)
    assert forecast.base_revenue == Decimal("100")  # latest four quarters
    assert forecast.historical_fiscal_years == (2024, 2025)
    assert forecast.observations[0].fiscal_year == 2026
    assert forecast.observations[0].period_end == datetime.date(2026, 12, 31)
    assert "latest-four-quarter" in forecast.seed_methodology
    assert forecast.ytd_anchor is not None
    assert forecast.ytd_anchor.fiscal_year == 2026
    assert forecast.ytd_anchor.ytd_period_end == datetime.date(2026, 6, 30)
    assert forecast.ytd_anchor.fiscal_year_end == datetime.date(2026, 12, 31)
    assert forecast.ytd_anchor.actual_revenue == Decimal("50")
    assert forecast.ytd_anchor.latest_annual_revenue == Decimal("100")
    assert forecast.ytd_anchor.actual_tax_rate == Decimal("25")
    assert forecast.ytd_anchor.revenue_growth == Decimal("10")
    first_audits = forecast.observations[0].cell_audits
    for field in ("revenue_growth", "operating_margin", "tax_rate"):
        assert first_audits[field].source.startswith("derived[")
        assert "ytd_actual" in first_audits[field].source
        assert "projected_remainder" in first_audits[field].source
        assert first_audits[field].confidence == "high"
    assert (
        "ytd_actual_depreciation=reported"
        in first_audits["depreciation_and_amortization"].source
    )
    assert "ytd_actual_capex=reported" in first_audits["capital_expenditures"].source
    assert (
        "projected_remainder=prior_forecast"
        in first_audits["change_in_operating_working_capital"].source
    )

    annual_only = financials.model_copy(
        update={
            "observations": [
                item
                for item in financials.observations
                if item.granularity == Granularity.ANNUAL
            ]
        }
    )
    fallback = FcffForecastService().forecast(
        annual_only, _explicit_forecast_parameters()
    )
    assert fallback.seed_type == ForecastSeedType.FISCAL_YEAR
    assert fallback.ytd_anchor is None
    assert fallback.observations[0].fiscal_year == 2026


def test_ytd_revenue_anchor_forecasts_only_the_guided_remaining_amount():
    parameters = _explicit_forecast_parameters().model_copy(
        update={"revenue_anchors": {2026: Decimal("125")}}
    )

    forecast = FcffForecastService().forecast(_forecast_financials(), parameters)

    # Reported H1 revenue is 50; the implied remaining-year revenue is 75.
    assert forecast.observations[0].revenue == Decimal("125")
    assert forecast.observations[0].operating_income == Decimal("25")

    invalid = parameters.model_copy(update={"revenue_anchors": {2026: Decimal("40")}})
    with pytest.raises(ValueError, match="below reported YTD revenue"):
        FcffForecastService().forecast(_forecast_financials(), invalid)


def test_non_calendar_fiscal_year_keeps_fiscal_alignment():
    forecast = FcffForecastService().forecast(
        _forecast_financials(fiscal_end_month=6), _explicit_forecast_parameters()
    )

    assert forecast.seed_type == ForecastSeedType.YTD_PLUS_FORECAST
    assert forecast.current_fiscal_year == 2026
    assert forecast.observations[0].period_end == datetime.date(2026, 6, 30)


def test_post_fiscal_year_reporting_gap_starts_with_next_unelapsed_fiscal_year():
    forecast = FcffForecastService().forecast(
        _forecast_financials(fiscal_end_month=6),
        _explicit_forecast_parameters(),
        as_of=VALUATION_DATE,
    )

    assert forecast.seed_type == ForecastSeedType.TTM
    assert forecast.ytd_anchor is None
    assert forecast.seed_period_end == datetime.date(2025, 12, 31)
    assert forecast.observations[0].fiscal_year == 2027
    assert forecast.observations[0].period_end == datetime.date(2027, 6, 30)
    assert forecast.observations[0].period_end > VALUATION_DATE
    assert "year ended" in forecast.seed_methodology

    result = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc=Decimal("8"), perpetual_growth_rate=Decimal("2")),
        FcffDcfCapitalBridge(
            fiscal_year=2026,
            period_end=datetime.date(2026, 6, 30),
            unit="USD",
            net_debt=Decimal("10"),
            diluted_shares=Decimal("10"),
            net_debt_source="controlled fixture",
            shares_source="controlled fixture",
        ),
        valuation_date=VALUATION_DATE,
    )
    assert result.value_per_share.is_finite()


def test_post_fiscal_year_gap_uses_labeled_ytd_run_rate_without_four_quarters():
    financials = _forecast_financials(fiscal_end_month=6)
    financials = financials.model_copy(
        update={
            "observations": [
                item
                for item in financials.observations
                if not (
                    item.granularity == Granularity.QUARTERLY
                    and item.fiscal_year == 2025
                    and item.fiscal_period == FiscalPeriod.Q3
                )
            ]
        }
    )

    forecast = FcffForecastService().forecast(
        financials,
        _explicit_forecast_parameters(),
        as_of=VALUATION_DATE,
    )

    assert forecast.seed_type == ForecastSeedType.YTD_RUN_RATE
    assert forecast.actual_quarters == 2
    assert forecast.observations[0].fiscal_year == 2027
    assert forecast.observations[0].period_end == datetime.date(2027, 6, 30)
    assert "annualized" in forecast.seed_methodology


def test_forecast_seed_excludes_quarterly_data_filed_after_as_of_date():
    financials = _forecast_financials()
    observations = []
    for item in financials.observations:
        filed = None
        if item.granularity == Granularity.QUARTERLY:
            filed = (
                datetime.date(2026, 8, 20)
                if item.fiscal_period == FiscalPeriod.Q2 and item.fiscal_year == 2026
                else min(item.period_end + datetime.timedelta(days=20), VALUATION_DATE)
            )
        observations.append(item.model_copy(update={"filed": filed}))
    financials = financials.model_copy(update={"observations": observations})

    forecast = FcffForecastService().forecast(
        financials,
        _explicit_forecast_parameters(),
        as_of=VALUATION_DATE,
    )

    assert forecast.actual_quarters == 1
    assert forecast.seed_period_end == datetime.date(2026, 3, 31)


def test_latest_quarterly_bridge_overrides_annual_and_excludes_future_filing():
    observations = []
    periods = (
        (
            2025,
            FiscalPeriod.FY,
            Granularity.ANNUAL,
            datetime.date(2025, 12, 31),
            None,
            {"cash": "20", "debt": "100", "shares": "10"},
        ),
        (
            2026,
            FiscalPeriod.Q1,
            Granularity.QUARTERLY,
            datetime.date(2026, 3, 31),
            datetime.date(2026, 5, 1),
            {"cash": "35", "debt": "90", "shares": "9"},
        ),
        (
            2026,
            FiscalPeriod.Q2,
            Granularity.QUARTERLY,
            datetime.date(2026, 6, 30),
            datetime.date(2026, 8, 20),
            {"cash": "50", "debt": "80", "shares": "8"},
        ),
    )
    for year, period, granularity, end, filed, values in periods:
        observations.extend(
            (
                _observation(
                    FinancialConcept.CASH_AND_EQUIVALENTS,
                    values["cash"],
                    year,
                    end,
                    granularity=granularity,
                    fiscal_period=period,
                    filed=filed,
                ),
                _observation(
                    FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
                    values["debt"],
                    year,
                    end,
                    granularity=granularity,
                    fiscal_period=period,
                    filed=filed,
                ),
                _observation(
                    FinancialConcept.SHARES_OUTSTANDING,
                    values["shares"],
                    year,
                    end,
                    granularity=granularity,
                    fiscal_period=period,
                    filed=filed,
                ),
            )
        )
    financials = NormalizedCompanyFinancials(
        provider="test",
        company_id="BRIDGE",
        company_name="Bridge",
        observations=observations,
    )
    bridge = FcffDcfCapitalBridgeResolver().resolve(
        financials,
        fiscal_year=2025,
        period_end=datetime.date(2025, 12, 31),
        unit="USD",
        valuation_date=VALUATION_DATE,
    )

    assert bridge.period_end == datetime.date(2026, 3, 31)
    assert bridge.gross_debt == Decimal("90")
    assert bridge.cash_and_equivalents == Decimal("35")
    assert bridge.diluted_shares == Decimal("9")
    assert bridge.debt_date == bridge.cash_date == bridge.shares_date
    assert not any("annual" in warning for warning in bridge.warnings)


def _valuation_profile(ticker, *, revenue="1000", industry="Software Infrastructure"):
    return ValuationProfile(
        provider="test",
        company_id=ticker,
        company_name=ticker,
        ticker=ticker,
        sector=Sector.TECHNOLOGY,
        industry=industry,
        country="US",
        reporting_currency="USD",
        latest_revenue=Decimal(revenue),
        business_archetype=BusinessArchetype.GENERAL_OPERATING,
        lifecycle=CompanyLifecycle.MATURE,
        cyclicality=Cyclicality.LOW,
    )


def test_provider_peer_discovery_is_deterministic_and_manual_override_remains_explicit(
    tmp_path,
):
    response = {
        "quotes": [
            {"symbol": "FAR", "quoteType": "EQUITY", "marketCap": 10000},
            {"symbol": "NEAR2", "quoteType": "EQUITY", "marketCap": 1100},
            {"symbol": "TARGET", "quoteType": "EQUITY", "marketCap": 1000},
            {"symbol": "NEAR1", "quoteType": "EQUITY", "marketCap": 900},
        ]
    }

    def screen(*_args, **_kwargs):
        return response

    target_source = YahooCompanyFinancials(
        symbol="TARGET",
        company_name="Target",
        currency="USD",
        sector="Technology",
        country="United States",
        market_capitalization=Decimal("1000"),
    )
    provider = YahooScreenerPeerDiscoveryProvider(
        FileSystemCache(tmp_path), screen=screen, use_cache=False, make_cache=False
    )

    async def discover_twice():
        return (
            await provider.discover(target_source, max_candidates=3),
            await provider.discover(target_source, max_candidates=3),
        )

    first, second = asyncio.run(discover_twice())

    assert (
        first.candidate_tickers
        == second.candidate_tickers
        == (
            "FAR",
            "NEAR2",
            "NEAR1",
        )
    )
    assert any("fell outside" in warning for warning in first.warnings)
    auto = PeerUniverseSelector().select(
        _valuation_profile("TARGET"),
        [_valuation_profile(symbol) for symbol in first.candidate_tickers],
        PeerSelectionParameters(
            max_peers=3,
            preferred_minimum=2,
            minimum_score=40,
            require_same_sector=False,
        ),
        discovery=first,
    )
    manual = PeerUniverseSelector().select(
        _valuation_profile("TARGET"),
        [_valuation_profile("HAND")],
        PeerSelectionParameters(max_peers=1, preferred_minimum=1, minimum_score=40),
    )
    weak = PeerUniverseSelector().select(
        _valuation_profile("TARGET"),
        [],
        discovery=PeerDiscoveryResult(
            provider="test",
            target_ticker="TARGET",
            methodology="No provider coverage",
            confidence="low",
            warnings=("No coverage",),
        ),
    )

    assert auto.selected_tickers == ("FAR", "NEAR1", "NEAR2")
    assert auto.discovery_source == "yahoo-screener"
    assert manual.selected_tickers == ("HAND",)
    assert manual.discovery_source == "manual override"
    assert weak.discovery_confidence == "low"
    assert not weak.selected_tickers
    assert "No coverage" in weak.warnings


def test_peer_selection_deduplicates_cross_listings_of_one_issuer():
    target = _valuation_profile("TARGET")
    primary = _valuation_profile("ISSUER")
    duplicate = _valuation_profile("ISSUER.L").model_copy(
        update={"company_name": primary.company_name + " CDR"}
    )
    distinct = _valuation_profile("DISTINCT")

    result = PeerUniverseSelector().select(
        target,
        [primary, duplicate, distinct],
        PeerSelectionParameters(max_peers=3, preferred_minimum=2, minimum_score=40),
    )

    assert result.selected_tickers == ("DISTINCT", "ISSUER")
    excluded = next(item for item in result.candidates if item.ticker == "ISSUER.L")
    assert any("Duplicate listing" in reason for reason in excluded.exclusions)


@pytest.mark.parametrize(
    ("ticker", "roics", "cyclicality", "growth", "capex", "net_debt"),
    [
        ("RACE", [28, 30, 31, 30, 32], Cyclicality.LOW, "10", "8", "5"),
        ("MSFT", [32, 35, 38, 40, 39], Cyclicality.LOW, "14", "24", "-10"),
        ("MATURE", [10, 10, 11, 10, 10], Cyclicality.LOW, "4", "5", "20"),
        ("CYCLICAL", [4, 18, 6, 20, 8], Cyclicality.HIGH, "7", "10", "35"),
        ("CAPITAL", [7, 8, 8, 9, 8], Cyclicality.MODERATE, "5", "18", "45"),
        ("LEVERAGED", [8, 9, 7, 8, 9], Cyclicality.MODERATE, "5", "7", "150"),
    ],
)
def test_cross_company_generic_workflow_produces_auditable_finite_results(
    ticker, roics, cyclicality, growth, capex, net_debt
):
    terminal = TerminalRoicResolver().resolve(
        _roic_financials(ticker, roics),
        wacc=Decimal("8"),
        terminal_growth=Decimal("2.5"),
        valuation_date=VALUATION_DATE,
        currency="USD",
        lifecycle=CompanyLifecycle.MATURE,
        cyclicality=cyclicality,
    )
    parameters = _explicit_forecast_parameters().model_copy(
        update={
            "forecast_years": 5,
            "revenue_growth": (Decimal(growth),),
            "capex_to_revenue": (Decimal(capex),),
        }
    )
    financials = _forecast_financials()
    seed = FcffForecastService().forecast(financials, parameters, as_of=VALUATION_DATE)
    forecast, plan = AdaptiveMultistageFcffForecastService().forecast(
        financials,
        seed,
        parameters,
        Decimal("2.5"),
        MultistageValuationConfiguration(
            terminal_return_on_invested_capital=terminal.value
        ),
        as_of=VALUATION_DATE,
    )
    result = FcffDcfService().value(
        forecast,
        FcffDcfParameters(wacc=Decimal("8"), perpetual_growth_rate=Decimal("2.5")),
        FcffDcfCapitalBridge(
            fiscal_year=2026,
            period_end=datetime.date(2026, 6, 30),
            unit="USD",
            net_debt=Decimal(net_debt),
            diluted_shares=Decimal("10"),
            net_debt_source="controlled archetype fixture",
            shares_source="controlled archetype fixture",
        ),
        multistage_plan=plan,
        valuation_date=VALUATION_DATE,
    )

    assert forecast.observations[0].period_end > VALUATION_DATE
    assert result.enterprise_value.is_finite()
    assert result.value_per_share.is_finite()
    assert plan.terminal_reinvestment_rate == Decimal("250") / terminal.value
    assert terminal.assumption.rationale
