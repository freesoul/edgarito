import datetime
from argparse import Namespace
from decimal import Decimal

from edgarito.cli.__main__ import _financial_snapshot_warnings
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.forecasting import FcffForecastParameters, ForecastSeedType
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.financials.availability import (
    FinancialObservationAvailabilityService,
    ObservationAvailabilityMode,
)
from edgarito.services.forecasting._fcff.service import FcffForecastService
from edgarito.services.valuation.fcff_dcf import FcffDcfCapitalBridgeResolver

UTC = datetime.timezone.utc
VALUATION_DATE = datetime.date(2026, 8, 7)
RETRIEVED_AT = datetime.datetime(2026, 8, 7, 12, tzinfo=UTC)


def _observation(
    concept: FinancialConcept,
    value: str,
    fiscal_year: int,
    fiscal_period: FiscalPeriod,
    period_end: datetime.date,
    *,
    filed: datetime.date | None = None,
) -> FinancialObservation:
    granularity = (
        Granularity.ANNUAL
        if fiscal_period == FiscalPeriod.FY
        else Granularity.QUARTERLY
    )
    return FinancialObservation(
        concept=concept,
        statement=concept.statement,
        value=Decimal(value),
        unit="EUR" if concept != FinancialConcept.SHARES_OUTSTANDING else "shares",
        granularity=granularity,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        period_end=period_end,
        provider="yahoo",
        taxonomy="yahoo-standardized",
        source_concept=concept.value,
        filed=filed,
    )


def _operating_period(
    fiscal_year: int,
    fiscal_period: FiscalPeriod,
    period_end: datetime.date,
    scale: Decimal,
) -> list[FinancialObservation]:
    values = {
        FinancialConcept.REVENUE: Decimal("100") * scale,
        FinancialConcept.OPERATING_INCOME: Decimal("20") * scale,
        FinancialConcept.PRETAX_INCOME: Decimal("18") * scale,
        FinancialConcept.INCOME_TAX_EXPENSE: Decimal("4") * scale,
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION: Decimal("5") * scale,
        FinancialConcept.CAPITAL_EXPENDITURES: Decimal("7") * scale,
        FinancialConcept.ACCOUNTS_RECEIVABLE: Decimal("20") * scale,
        FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS: Decimal("5") * scale,
        FinancialConcept.ACCOUNTS_PAYABLE: Decimal("8") * scale,
        FinancialConcept.ACCRUED_LIABILITIES: Decimal("4") * scale,
        FinancialConcept.DEFERRED_REVENUE_CURRENT: Decimal("3") * scale,
    }
    return [
        _observation(
            concept,
            str(value),
            fiscal_year,
            fiscal_period,
            period_end,
        )
        for concept, value in values.items()
    ]


def _financials() -> NormalizedCompanyFinancials:
    observations = [
        *_operating_period(
            2024,
            FiscalPeriod.FY,
            datetime.date(2024, 12, 31),
            Decimal("3.6"),
        ),
        *_operating_period(
            2025,
            FiscalPeriod.FY,
            datetime.date(2025, 12, 31),
            Decimal("4"),
        ),
        *_operating_period(
            2026,
            FiscalPeriod.Q1,
            datetime.date(2026, 3, 31),
            Decimal("1"),
        ),
        *_operating_period(
            2026,
            FiscalPeriod.Q2,
            datetime.date(2026, 6, 30),
            Decimal("1.1"),
        ),
    ]
    for fiscal_period, period_end, debt, cash, investments, shares in (
        (
            FiscalPeriod.Q1,
            datetime.date(2026, 3, 31),
            "50",
            "12",
            "3",
            "100",
        ),
        (
            FiscalPeriod.Q2,
            datetime.date(2026, 6, 30),
            "45",
            "15",
            "4",
            "99",
        ),
    ):
        observations.extend(
            (
                _observation(
                    FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
                    debt,
                    2026,
                    fiscal_period,
                    period_end,
                ),
                _observation(
                    FinancialConcept.CASH_AND_EQUIVALENTS,
                    cash,
                    2026,
                    fiscal_period,
                    period_end,
                ),
                _observation(
                    FinancialConcept.SHORT_TERM_INVESTMENTS,
                    investments,
                    2026,
                    fiscal_period,
                    period_end,
                ),
                _observation(
                    FinancialConcept.SHARES_OUTSTANDING,
                    shares,
                    2026,
                    fiscal_period,
                    period_end,
                ),
            )
        )
    return NormalizedCompanyFinancials(
        provider="yahoo",
        company_id="ASML.AS",
        company_name="ASML Holding N.V.",
        ticker="ASML.AS",
        retrieved_at=RETRIEVED_AT,
        observations=observations,
    )


def _parameters() -> FcffForecastParameters:
    return FcffForecastParameters(
        forecast_years=2,
        revenue_growth=Decimal("8"),
        operating_margin=Decimal("22"),
        tax_rate=Decimal("20"),
        depreciation_to_revenue=Decimal("5"),
        capex_to_revenue=Decimal("7"),
        operating_working_capital_to_revenue=Decimal("10"),
    )


def test_current_snapshot_includes_yahoo_q2_that_is_present_now():
    forecast = FcffForecastService().forecast(
        _financials(),
        _parameters(),
        as_of=VALUATION_DATE,
        availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
    )

    assert forecast.seed_type == ForecastSeedType.YTD_PLUS_FORECAST
    assert forecast.actual_quarters == 2
    assert forecast.seed_period_end == datetime.date(2026, 6, 30)
    assert forecast.financial_snapshot_retrieved_at == RETRIEVED_AT
    assert forecast.availability_mode == "current_snapshot"


def test_point_in_time_excludes_q2_until_conservative_yahoo_date():
    financials = _financials()
    q2_revenue = next(
        item
        for item in financials.observations
        if item.concept == FinancialConcept.REVENUE
        and item.fiscal_period == FiscalPeriod.Q2
    )
    assert FinancialObservationAvailabilityService().available_on(
        q2_revenue,
        mode=ObservationAvailabilityMode.POINT_IN_TIME,
    ) == datetime.date(2026, 8, 14)

    before = FcffForecastService().forecast(
        financials,
        _parameters(),
        as_of=VALUATION_DATE,
        availability_mode=ObservationAvailabilityMode.POINT_IN_TIME,
    )
    after = FcffForecastService().forecast(
        financials,
        _parameters(),
        as_of=datetime.date(2026, 8, 15),
        availability_mode=ObservationAvailabilityMode.POINT_IN_TIME,
    )

    assert before.actual_quarters == 1
    assert before.seed_period_end == datetime.date(2026, 3, 31)
    assert after.actual_quarters == 2
    assert after.seed_period_end == datetime.date(2026, 6, 30)


def test_actual_filing_date_overrides_the_yahoo_estimate():
    financials = _financials()
    financials.observations = [
        item.model_copy(update={"filed": datetime.date(2026, 7, 15)})
        if item.period_end == datetime.date(2026, 6, 30)
        else item
        for item in financials.observations
    ]

    forecast = FcffForecastService().forecast(
        financials,
        _parameters(),
        as_of=VALUATION_DATE,
        availability_mode=ObservationAvailabilityMode.POINT_IN_TIME,
    )

    assert forecast.actual_quarters == 2


def test_incomplete_current_quarter_falls_back_to_last_complete_quarter():
    financials = _financials()
    financials.observations = [
        item
        for item in financials.observations
        if not (
            item.period_end == datetime.date(2026, 6, 30)
            and item.concept == FinancialConcept.CAPITAL_EXPENDITURES
        )
    ]

    forecast = FcffForecastService().forecast(
        financials,
        _parameters(),
        as_of=VALUATION_DATE,
        availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
    )

    assert forecast.actual_quarters == 1
    assert forecast.seed_period_end == datetime.date(2026, 3, 31)
    assert "Capital Expenditures" in forecast.warnings[0]
    assert "falls back" in forecast.warnings[0]


def test_capital_bridge_uses_q2_only_for_current_snapshot_on_august_7():
    financials = _financials()
    resolver = FcffDcfCapitalBridgeResolver()
    current = resolver.resolve(
        financials,
        fiscal_year=2026,
        period_end=datetime.date(2026, 6, 30),
        unit="EUR",
        valuation_date=VALUATION_DATE,
        availability_mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
    )
    historical = resolver.resolve(
        financials,
        fiscal_year=2026,
        period_end=datetime.date(2026, 3, 31),
        unit="EUR",
        valuation_date=VALUATION_DATE,
        availability_mode=ObservationAvailabilityMode.POINT_IN_TIME,
    )

    assert current.period_end == datetime.date(2026, 6, 30)
    assert current.gross_debt == Decimal("45")
    assert current.cash_and_equivalents == Decimal("15")
    assert current.non_operating_assets == Decimal("4")
    assert current.diluted_shares == Decimal("99")
    assert historical.period_end == datetime.date(2026, 3, 31)
    assert historical.gross_debt == Decimal("50")
    assert historical.cash_and_equivalents == Decimal("12")


def test_sec_filing_date_behavior_is_unchanged():
    observation = _observation(
        FinancialConcept.REVENUE,
        "100",
        2026,
        FiscalPeriod.Q2,
        datetime.date(2026, 6, 30),
        filed=datetime.date(2026, 7, 25),
    ).model_copy(update={"provider": "sec"})
    service = FinancialObservationAvailabilityService()

    for mode in ObservationAvailabilityMode:
        assert not service.is_available(
            observation,
            as_of=datetime.date(2026, 7, 24),
            mode=mode,
        )
        assert service.is_available(
            observation,
            as_of=datetime.date(2026, 7, 25),
            mode=mode,
        )


def test_current_snapshot_cannot_prove_backdated_availability():
    observation = next(
        item
        for item in _financials().observations
        if item.concept == FinancialConcept.REVENUE
        and item.fiscal_period == FiscalPeriod.Q2
    )

    assert not FinancialObservationAvailabilityService().is_available(
        observation,
        as_of=datetime.date(2026, 8, 6),
        mode=ObservationAvailabilityMode.CURRENT_SNAPSHOT,
        snapshot_retrieved_at=RETRIEVED_AT,
    )


def test_current_valuation_surfaces_stale_yahoo_snapshot_provenance():
    stale = _financials().model_copy(
        update={
            "retrieved_at": datetime.datetime.now(UTC) - datetime.timedelta(hours=25)
        }
    )

    warnings = _financial_snapshot_warnings(
        stale, Namespace(financial_snapshot_max_age_hours=24)
    )

    assert len(warnings) == 1
    assert "stale" in warnings[0]
    assert "--refresh" in warnings[0]
