import asyncio
import datetime
from decimal import Decimal

from edgarito.schemas.market import (
    CashDividend,
    PriceBar,
    ReferenceMarketSeries,
    ReferenceSeriesKind,
    ReferenceValueUnit,
    SecurityMarketData,
)
from edgarito.services.providers.ecb import EcbClient


class EcbMarketDataCurrencyConverter:
    """Convert market history through date-aligned ECB currency-per-euro rates."""

    def __init__(self, client: EcbClient):
        self._client = client

    async def convert(
        self,
        market_data: SecurityMarketData,
        target_currency: str,
        *,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> SecurityMarketData:
        target = target_currency.strip().upper()
        source = market_data.currency
        if source == target:
            return market_data
        if not market_data.prices:
            raise ValueError("FX conversion requires market prices")
        first_date = min(item.observed_on for item in market_data.prices)
        last_date = max(item.observed_on for item in market_data.prices)

        currencies = [currency for currency in (source, target) if currency != "EUR"]
        series = await asyncio.gather(
            *(
                self._currency_per_euro_series(
                    currency,
                    first_date,
                    last_date,
                    use_cache=use_cache,
                    make_cache=make_cache,
                )
                for currency in currencies
            )
        )
        by_currency = dict(zip(currencies, series, strict=True))
        converted_prices = tuple(
            self._convert_price(item, source, target, by_currency)
            for item in market_data.prices
        )
        converted_dividends = tuple(
            self._convert_dividend(item, source, target, by_currency)
            for item in market_data.dividends
        )
        series_ids = ", ".join(item.series_id for item in series)
        source_version = (
            f"{market_data.source_version or 'unversioned'}; date-aligned ECB FX "
            f"{series_ids}; {len(converted_prices)} prices converted {target}/{source}"
        )
        retrieved_at = max(
            [market_data.retrieved_at, *(item.retrieved_at for item in series)]
        )
        return SecurityMarketData(
            provider=f"{market_data.provider}+ecb-fx",
            provider_symbol=market_data.provider_symbol,
            identifiers=market_data.identifiers,
            currency=target,
            exchange=market_data.exchange,
            frequency=market_data.frequency,
            retrieved_at=retrieved_at,
            source_version=source_version,
            prices=converted_prices,
            dividends=converted_dividends,
            splits=market_data.splits,
        )

    async def _currency_per_euro_series(
        self,
        currency: str,
        first_price_date: datetime.date,
        last_price_date: datetime.date,
        *,
        use_cache: bool,
        make_cache: bool,
    ) -> ReferenceMarketSeries:
        return await self._client.get_series(
            "EXR",
            f"D.{currency}.EUR.SP00.A",
            kind=ReferenceSeriesKind.EXCHANGE_RATE,
            unit=ReferenceValueUnit.CURRENCY_PER_CURRENCY,
            start_period=first_price_date - datetime.timedelta(days=14),
            end_period=last_price_date,
            use_cache=use_cache,
            make_cache=make_cache,
        )

    @classmethod
    def _factor_on(cls, observed_on, source, target, series):
        if source == "EUR":
            observation = cls._latest_on_or_before(series[target], observed_on)
            if observation.value <= 0:
                raise ValueError("ECB reference exchange rates must be positive")
            return observation.value
        if target == "EUR":
            observation = cls._latest_on_or_before(series[source], observed_on)
            if observation.value <= 0:
                raise ValueError("ECB reference exchange rates must be positive")
            return Decimal(1) / observation.value

        source_values = {
            item.period_end: item.value for item in series[source].observations
        }
        target_values = {
            item.period_end: item.value for item in series[target].observations
        }
        common_dates = [
            item
            for item in source_values.keys() & target_values.keys()
            if item <= observed_on
        ]
        if not common_dates:
            raise ValueError(
                f"ECB returned no aligned {source}/{target} reference rate on or "
                f"before {observed_on.isoformat()}"
            )
        observed_on = max(common_dates)
        source_rate = source_values[observed_on]
        target_rate = target_values[observed_on]
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("ECB reference exchange rates must be positive")
        return target_rate / source_rate

    @staticmethod
    def _latest_on_or_before(series, observed_on):
        candidates = [
            item for item in series.observations if item.period_end <= observed_on
        ]
        if not candidates:
            raise ValueError(
                f"ECB returned no {series.currency or series.series_id} reference "
                f"rate on or before {observed_on.isoformat()}"
            )
        return max(candidates, key=lambda item: item.period_end)

    @classmethod
    def _convert_price(cls, price, source, target, series):
        factor = cls._factor_on(price.observed_on, source, target, series)
        return PriceBar(
            observed_on=price.observed_on,
            open=cls._scale(price.open, factor),
            high=cls._scale(price.high, factor),
            low=cls._scale(price.low, factor),
            close=price.close * factor,
            adjusted_close=cls._scale(price.adjusted_close, factor),
            volume=price.volume,
        )

    @classmethod
    def _convert_dividend(cls, dividend, source, target, series):
        factor = cls._factor_on(dividend.ex_date, source, target, series)
        return CashDividend(
            ex_date=dividend.ex_date,
            amount=dividend.amount * factor,
            currency=target,
            declaration_date=dividend.declaration_date,
            record_date=dividend.record_date,
            payment_date=dividend.payment_date,
        )

    @staticmethod
    def _scale(value: Decimal | None, factor: Decimal) -> Decimal | None:
        return value * factor if value is not None else None


__all__ = ["EcbMarketDataCurrencyConverter"]
