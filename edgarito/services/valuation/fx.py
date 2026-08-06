import asyncio
import datetime
from decimal import Decimal

from edgarito.schemas.market import (
    PriceBar,
    ReferenceMarketSeries,
    ReferenceSeriesKind,
    ReferenceValueUnit,
    SecurityMarketData,
)
from edgarito.services.providers.ecb import EcbClient


class EcbMarketDataCurrencyConverter:
    """Convert a latest market quote through ECB currency-per-euro rates."""

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
        latest_price = market_data.latest_price
        if latest_price is None:
            raise ValueError("FX conversion requires a latest market price")

        currencies = [currency for currency in (source, target) if currency != "EUR"]
        series = await asyncio.gather(
            *(
                self._currency_per_euro_series(
                    currency,
                    latest_price.observed_on,
                    use_cache=use_cache,
                    make_cache=make_cache,
                )
                for currency in currencies
            )
        )
        by_currency = dict(zip(currencies, series, strict=True))
        source_rate, target_rate, observed_on = self._aligned_rates(
            source,
            target,
            by_currency,
        )
        factor = target_rate / source_rate
        converted = PriceBar(
            observed_on=latest_price.observed_on,
            open=self._scale(latest_price.open, factor),
            high=self._scale(latest_price.high, factor),
            low=self._scale(latest_price.low, factor),
            close=latest_price.close * factor,
            adjusted_close=self._scale(latest_price.adjusted_close, factor),
            volume=latest_price.volume,
        )
        series_ids = ", ".join(item.series_id for item in series)
        source_version = (
            f"{market_data.source_version or 'unversioned'}; ECB FX {series_ids} "
            f"observed {observed_on.isoformat()}; factor {factor} {target}/{source}"
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
            prices=(converted,),
        )

    async def _currency_per_euro_series(
        self,
        currency: str,
        price_date: datetime.date,
        *,
        use_cache: bool,
        make_cache: bool,
    ) -> ReferenceMarketSeries:
        return await self._client.get_series(
            "EXR",
            f"D.{currency}.EUR.SP00.A",
            kind=ReferenceSeriesKind.EXCHANGE_RATE,
            unit=ReferenceValueUnit.CURRENCY_PER_CURRENCY,
            start_period=price_date - datetime.timedelta(days=14),
            end_period=price_date,
            use_cache=use_cache,
            make_cache=make_cache,
        )

    @staticmethod
    def _aligned_rates(source, target, series):
        if source == "EUR":
            observation = series[target].latest_observation
            if observation.value <= 0:
                raise ValueError("ECB reference exchange rates must be positive")
            return Decimal(1), observation.value, observation.period_end
        if target == "EUR":
            observation = series[source].latest_observation
            if observation.value <= 0:
                raise ValueError("ECB reference exchange rates must be positive")
            return observation.value, Decimal(1), observation.period_end

        source_values = {
            item.period_end: item.value for item in series[source].observations
        }
        target_values = {
            item.period_end: item.value for item in series[target].observations
        }
        common_dates = source_values.keys() & target_values.keys()
        if not common_dates:
            raise ValueError(
                f"ECB returned no aligned {source}/{target} reference-rate date"
            )
        observed_on = max(common_dates)
        source_rate = source_values[observed_on]
        target_rate = target_values[observed_on]
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("ECB reference exchange rates must be positive")
        return source_rate, target_rate, observed_on

    @staticmethod
    def _scale(value: Decimal | None, factor: Decimal) -> Decimal | None:
        return value * factor if value is not None else None


__all__ = ["EcbMarketDataCurrencyConverter"]
