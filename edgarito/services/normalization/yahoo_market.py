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
            PriceBar(
                observed_on=row.observed_on,
                open=self._scaled(row.open, scale),
                high=self._scaled(row.high, scale),
                low=self._scaled(row.low, scale),
                close=self._scaled(row.close, scale),
                adjusted_close=self._scaled(row.adjusted_close, scale),
                volume=row.volume,
            )
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


__all__ = ["YahooMarketNormalizer"]
