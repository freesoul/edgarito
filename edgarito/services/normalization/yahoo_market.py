from decimal import Decimal
from typing import Optional

from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.market import (
    CashDividend,
    MarketDataFrequency,
    PriceBar,
    SecurityMarketData,
    StockSplit,
)
from edgarito.schemas.providers.yahoo.market import YahooMarketHistory


class YahooMarketNormalizer:
    """Normalize yfinance daily prices, dividends, and splits."""

    def normalize(
        self,
        history: YahooMarketHistory,
        *,
        identifiers: Optional[SecurityIdentifiers] = None,
    ) -> SecurityMarketData:
        currency, scale = self._currency_and_scale(history.currency)
        prices = tuple(
            self._price_bar(row, scale)
            for row in sorted(history.rows, key=lambda item: item.observed_on)
            if row.close is not None
        )
        dividends = tuple(
            CashDividend(
                ex_date=row.observed_on,
                amount=self._scaled(row.dividend, scale),
                currency=currency,
            )
            for row in sorted(history.rows, key=lambda item: item.observed_on)
            if row.dividend is not None
        )
        splits = tuple(
            StockSplit(
                effective_date=row.observed_on,
                from_shares=Decimal(1),
                to_shares=row.split_factor,
            )
            for row in sorted(history.rows, key=lambda item: item.observed_on)
            if row.split_factor is not None
        )
        return SecurityMarketData(
            provider="yahoo",
            provider_symbol=history.symbol,
            identifiers=identifiers or SecurityIdentifiers(ticker=history.symbol),
            currency=currency,
            exchange=history.exchange,
            frequency=MarketDataFrequency.DAILY,
            retrieved_at=history.retrieved_at,
            source_version=history.source_version,
            prices=prices,
            dividends=dividends,
            splits=splits,
        )

    @staticmethod
    def _currency_and_scale(currency: str) -> tuple[str, Decimal]:
        normalized = currency.strip()
        if normalized == "GBp" or normalized.upper() == "GBX":
            return "GBP", Decimal("0.01")
        if normalized == "ZAc":
            return "ZAR", Decimal("0.01")
        return normalized.upper(), Decimal(1)

    @staticmethod
    def _scaled(value: Optional[Decimal], scale: Decimal) -> Optional[Decimal]:
        return value * scale if value is not None else None

    @classmethod
    def _price_bar(cls, row, scale: Decimal) -> PriceBar:
        open_price = cls._scaled(row.open, scale)
        high = cls._scaled(row.high, scale)
        low = cls._scaled(row.low, scale)
        close = cls._scaled(row.close, scale)
        observed = [
            value for value in (open_price, high, low, close) if value is not None
        ]
        # Yahoo occasionally publishes an OHLC row whose close is a few ticks
        # outside the stated daily range. Preserve all fields while restoring
        # the basic candle invariant required by the normalized schema.
        if observed:
            high = max(observed) if high is not None else None
            low = min(observed) if low is not None else None
        return PriceBar(
            observed_on=row.observed_on,
            open=open_price,
            high=high,
            low=low,
            close=close,
            adjusted_close=cls._scaled(row.adjusted_close, scale),
            volume=row.volume,
        )


__all__ = ["YahooMarketNormalizer"]
