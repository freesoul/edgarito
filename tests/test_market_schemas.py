import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.market import (
    CashDividend,
    MarketDataFrequency,
    PriceBar,
    ReferenceMarketSeries,
    ReferenceObservation,
    ReferenceSeriesKind,
    ReferenceValueUnit,
    SecurityMarketData,
    StockSplit,
)

UTC = datetime.timezone.utc


def _price(observed_on: datetime.date, close: str) -> PriceBar:
    return PriceBar(
        observed_on=observed_on,
        open=Decimal(close) - Decimal("1"),
        high=Decimal(close) + Decimal("1"),
        low=Decimal(close) - Decimal("2"),
        close=Decimal(close),
        adjusted_close=Decimal(close) - Decimal("0.5"),
        volume=100,
    )


def test_security_market_data_preserves_prices_and_corporate_actions():
    market_data = SecurityMarketData(
        provider=" alphavantage ",
        provider_symbol=" AAPL ",
        identifiers=SecurityIdentifiers(ticker="aapl", isin="US0378331005"),
        currency="usd",
        exchange=" NASDAQ ",
        frequency=MarketDataFrequency.DAILY,
        retrieved_at=datetime.datetime(2026, 8, 6, 12, tzinfo=UTC),
        source_version=" daily-adjusted-v1 ",
        prices=(
            _price(datetime.date(2026, 8, 5), "210"),
            _price(datetime.date(2026, 8, 6), "212"),
        ),
        dividends=(
            CashDividend(
                ex_date=datetime.date(2026, 8, 10),
                amount=Decimal("0.26"),
                currency="usd",
                payment_date=datetime.date(2026, 8, 13),
            ),
        ),
        splits=(
            StockSplit(
                effective_date=datetime.date(2020, 8, 31),
                from_shares=Decimal("1"),
                to_shares=Decimal("4"),
            ),
        ),
    )

    assert market_data.provider == "alphavantage"
    assert market_data.provider_symbol == "AAPL"
    assert market_data.currency == "USD"
    assert market_data.latest_price.close == Decimal("212")
    assert market_data.dividends[0].currency == "USD"
    assert market_data.splits[0].factor == Decimal("4")
    assert SecurityMarketData.model_validate_json(market_data.model_dump_json()) == (
        market_data
    )


def test_market_data_rejects_invalid_prices_duplicates_and_naive_timestamps():
    with pytest.raises(ValidationError, match="High price cannot be below"):
        PriceBar(
            observed_on=datetime.date(2026, 8, 6),
            open=Decimal("100"),
            high=Decimal("99"),
            close=Decimal("101"),
        )

    repeated = _price(datetime.date(2026, 8, 6), "100")
    with pytest.raises(ValidationError, match="cannot repeat a date"):
        SecurityMarketData(
            provider="alphavantage",
            provider_symbol="AAPL",
            identifiers=SecurityIdentifiers(ticker="AAPL"),
            currency="USD",
            retrieved_at=datetime.datetime(2026, 8, 6, 12, tzinfo=UTC),
            prices=(repeated, repeated),
        )

    with pytest.raises(ValidationError, match="must include a timezone"):
        SecurityMarketData(
            provider="alphavantage",
            provider_symbol="AAPL",
            identifiers=SecurityIdentifiers(ticker="AAPL"),
            currency="USD",
            retrieved_at=datetime.datetime(2026, 8, 6, 12),
            prices=(repeated,),
        )


def test_reference_market_series_retains_release_dates_and_latest_value():
    series = ReferenceMarketSeries(
        provider=" treasury ",
        series_id=" DGS10 ",
        name=" 10-Year Treasury Yield ",
        kind=ReferenceSeriesKind.GOVERNMENT_YIELD,
        unit=ReferenceValueUnit.PERCENTAGE_POINTS,
        frequency=MarketDataFrequency.DAILY,
        currency="usd",
        country="us",
        tenor_months=120,
        source_version="2026-08-06",
        retrieved_at=datetime.datetime(2026, 8, 6, 18, tzinfo=UTC),
        observations=(
            ReferenceObservation(
                period_end=datetime.date(2026, 8, 5),
                available_on=datetime.date(2026, 8, 5),
                value=Decimal("4.15"),
            ),
            ReferenceObservation(
                period_end=datetime.date(2026, 8, 6),
                available_on=datetime.date(2026, 8, 6),
                value=Decimal("4.12"),
            ),
        ),
    )

    assert series.provider == "treasury"
    assert series.currency == "USD"
    assert series.country == "US"
    assert series.latest_observation.value == Decimal("4.12")
    assert ReferenceMarketSeries.model_validate_json(series.model_dump_json()) == series


def test_reference_market_series_requires_unique_nonempty_observations():
    observation = ReferenceObservation(
        period_end=datetime.date(2026, 8, 6), value=Decimal("4.12")
    )
    common = {
        "provider": "treasury",
        "series_id": "DGS10",
        "name": "10-Year Treasury Yield",
        "kind": ReferenceSeriesKind.GOVERNMENT_YIELD,
        "unit": ReferenceValueUnit.PERCENTAGE_POINTS,
        "frequency": MarketDataFrequency.DAILY,
        "retrieved_at": datetime.datetime(2026, 8, 6, 18, tzinfo=UTC),
    }

    with pytest.raises(ValidationError, match="at least one observation"):
        ReferenceMarketSeries(**common, observations=())
    with pytest.raises(ValidationError, match="cannot repeat a period"):
        ReferenceMarketSeries(**common, observations=(observation, observation))
