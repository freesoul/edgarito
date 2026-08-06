import asyncio
import datetime
from decimal import Decimal

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
from edgarito.services.valuation import EcbMarketDataCurrencyConverter

RETRIEVED = datetime.datetime(2026, 8, 6, tzinfo=datetime.timezone.utc)


class _FakeEcbClient:
    async def get_series(self, flow_ref, key, **kwargs):
        assert flow_ref == "EXR"
        currency = key.split(".")[1]
        rates = {"USD": Decimal("1.15"), "GBP": Decimal("0.85")}
        return ReferenceMarketSeries(
            provider="ecb",
            series_id=f"EXR/{key}",
            name=f"{currency} per EUR",
            kind=ReferenceSeriesKind.EXCHANGE_RATE,
            unit=ReferenceValueUnit.CURRENCY_PER_CURRENCY,
            frequency=MarketDataFrequency.DAILY,
            retrieved_at=RETRIEVED,
            observations=(
                ReferenceObservation(
                    period_end=datetime.date(2026, 8, 5),
                    value=rates[currency],
                ),
            ),
            currency=currency,
        )


def test_ecb_fx_converter_aligns_yahoo_quote_with_statement_currency():
    market_data = SecurityMarketData(
        provider="yahoo",
        provider_symbol="RACE",
        identifiers=SecurityIdentifiers(ticker="RACE"),
        currency="USD",
        frequency=MarketDataFrequency.DAILY,
        retrieved_at=RETRIEVED,
        source_version="test",
        prices=(
            PriceBar(
                observed_on=datetime.date(2026, 8, 5),
                open=Decimal("113.85"),
                high=Decimal("117.30"),
                low=Decimal("112.70"),
                close=Decimal("115"),
            ),
        ),
    )

    converted = asyncio.run(
        EcbMarketDataCurrencyConverter(_FakeEcbClient()).convert(market_data, "EUR")
    )

    assert converted.currency == "EUR"
    assert converted.provider == "yahoo+ecb-fx"
    assert converted.latest_price is not None
    assert converted.latest_price.close == Decimal("100")
    assert "EXR/D.USD.EUR.SP00.A" in converted.source_version


def test_ecb_fx_converter_uses_euro_cross_for_two_non_euro_currencies():
    market_data = SecurityMarketData(
        provider="yahoo",
        provider_symbol="EX",
        identifiers=SecurityIdentifiers(ticker="EX"),
        currency="USD",
        frequency=MarketDataFrequency.DAILY,
        retrieved_at=RETRIEVED,
        prices=(PriceBar(observed_on=datetime.date(2026, 8, 5), close=Decimal("115")),),
    )

    converted = asyncio.run(
        EcbMarketDataCurrencyConverter(_FakeEcbClient()).convert(market_data, "GBP")
    )

    assert converted.latest_price is not None
    assert converted.latest_price.close == Decimal("85")
