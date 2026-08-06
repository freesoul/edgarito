import datetime
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
from edgarito.schemas.providers.alphavantage.market import (
    DailyTimeSeriesResponse,
    DividendResponse,
    GlobalQuoteResponse,
    SplitResponse,
)


class AlphaVantageMarketNormalizer:
    """Normalize raw Alpha Vantage prices and corporate actions."""

    def normalize(
        self,
        *,
        symbol: str,
        currency: str,
        daily_prices: Optional[DailyTimeSeriesResponse] = None,
        dividends: Optional[DividendResponse] = None,
        splits: Optional[SplitResponse] = None,
        identifiers: Optional[SecurityIdentifiers] = None,
        exchange: Optional[str] = None,
        retrieved_at: Optional[datetime.datetime] = None,
    ) -> SecurityMarketData:
        normalized_symbol = symbol.strip().upper()
        self._require_symbols(normalized_symbol, daily_prices, dividends, splits)
        prices = self.price_history(daily_prices) if daily_prices else ()
        normalized_dividends = (
            self.dividend_history(dividends, currency) if dividends else ()
        )
        normalized_splits = self.split_history(splits) if splits else ()
        source_version = (
            daily_prices.metadata.last_refreshed.isoformat()
            if daily_prices is not None
            else None
        )
        return SecurityMarketData(
            provider="alphavantage",
            provider_symbol=normalized_symbol,
            identifiers=identifiers or SecurityIdentifiers(ticker=normalized_symbol),
            currency=currency,
            exchange=exchange,
            frequency=MarketDataFrequency.DAILY,
            retrieved_at=retrieved_at or datetime.datetime.now(datetime.timezone.utc),
            source_version=source_version,
            prices=prices,
            dividends=normalized_dividends,
            splits=normalized_splits,
        )

    def normalize_quote(
        self,
        response: GlobalQuoteResponse,
        *,
        currency: str,
        identifiers: Optional[SecurityIdentifiers] = None,
        exchange: Optional[str] = None,
        retrieved_at: Optional[datetime.datetime] = None,
    ) -> SecurityMarketData:
        quote = response.quote
        symbol = quote.symbol.strip().upper()
        return SecurityMarketData(
            provider="alphavantage",
            provider_symbol=symbol,
            identifiers=identifiers or SecurityIdentifiers(ticker=symbol),
            currency=currency,
            exchange=exchange,
            frequency=MarketDataFrequency.SNAPSHOT,
            retrieved_at=retrieved_at or datetime.datetime.now(datetime.timezone.utc),
            source_version=quote.latest_trading_day.isoformat(),
            prices=(
                PriceBar(
                    observed_on=quote.latest_trading_day,
                    open=quote.open,
                    high=quote.high,
                    low=quote.low,
                    close=quote.price,
                    volume=quote.volume,
                ),
            ),
        )

    @staticmethod
    def price_history(
        response: DailyTimeSeriesResponse,
    ) -> tuple[PriceBar, ...]:
        return tuple(
            PriceBar(
                observed_on=observed_on,
                open=price.open,
                high=price.high,
                low=price.low,
                close=price.close,
                volume=price.volume,
            )
            for observed_on, price in sorted(response.time_series.items())
        )

    @staticmethod
    def dividend_history(
        response: DividendResponse, currency: str
    ) -> tuple[CashDividend, ...]:
        return tuple(
            CashDividend(
                ex_date=item.ex_dividend_date,
                declaration_date=item.declaration_date,
                record_date=item.record_date,
                payment_date=item.payment_date,
                amount=item.amount,
                currency=currency,
            )
            for item in sorted(response.data, key=lambda item: item.ex_dividend_date)
        )

    @staticmethod
    def split_history(response: SplitResponse) -> tuple[StockSplit, ...]:
        return tuple(
            StockSplit(
                effective_date=item.effective_date,
                from_shares=Decimal(1),
                to_shares=item.split_factor,
            )
            for item in sorted(response.data, key=lambda item: item.effective_date)
        )

    @staticmethod
    def _require_symbols(
        symbol: str,
        daily_prices: Optional[DailyTimeSeriesResponse],
        dividends: Optional[DividendResponse],
        splits: Optional[SplitResponse],
    ) -> None:
        response_symbols = [
            value
            for value in (
                daily_prices.metadata.symbol if daily_prices else None,
                dividends.symbol if dividends else None,
                splits.symbol if splits else None,
            )
            if value is not None
        ]
        if any(value.strip().upper() != symbol for value in response_symbols):
            raise ValueError("Alpha Vantage market responses must use the same symbol")


__all__ = ["AlphaVantageMarketNormalizer"]
