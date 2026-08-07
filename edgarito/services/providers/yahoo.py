import asyncio
import datetime
import re
from decimal import Decimal
from typing import Callable, Optional

import pandas as pd
import yfinance as yf

from edgarito.schemas.providers.yahoo.fundamentals import (
    YahooCompanyFinancials,
    YahooFinancialReport,
)
from edgarito.schemas.providers.yahoo.market import YahooMarketHistory, YahooPriceRow
from edgarito.services.cache.filesystem_cache import FileSystemCache


class YahooFinanceClient:
    """Retrieve keyless Yahoo fundamentals and daily market history via yfinance."""

    _SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._^-]*$")
    _PERIOD_PATTERN = re.compile(r"^(?:[0-9]+(?:D|MO|Y)|YTD|MAX)$")

    def __init__(
        self,
        cache: FileSystemCache,
        ticker_factory: Optional[Callable[[str], object]] = None,
    ):
        self._cache = cache
        self._ticker_factory = ticker_factory or yf.Ticker

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get_company_financials(
        self,
        symbol: str,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> YahooCompanyFinancials:
        normalized_symbol = self._normalize_symbol(symbol)
        cache_path = f"providers/yahoo/{normalized_symbol}/financials.json"
        if use_cache:
            cached = self._cache.read(cache_path)
            if cached is not None:
                return YahooCompanyFinancials.model_validate_json(cached)

        try:
            result = await asyncio.to_thread(
                self._fetch_company_financials, normalized_symbol
            )
        except Exception as exc:
            raise RuntimeError(
                "Yahoo financial-statement retrieval failed for "
                f"{normalized_symbol}: {exc}"
            ) from exc
        if make_cache:
            self._cache.save(cache_path, result.model_dump_json())
        return result

    async def get_price_history(
        self,
        symbol: str,
        *,
        period: str = "max",
        start: Optional[datetime.date] = None,
        end: Optional[datetime.date] = None,
        repair: bool = True,
        currency: Optional[str] = None,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> YahooMarketHistory:
        normalized_symbol = self._normalize_symbol(symbol)
        normalized_period = self._normalize_period(period)
        if start is not None and end is not None and start >= end:
            raise ValueError("Yahoo history start must be before end")
        variant = self._history_variant(normalized_period, start, end, repair)
        cache_path = f"providers/yahoo/{normalized_symbol}/history_{variant}.json"
        if use_cache:
            cached = self._cache.read(cache_path)
            if cached is not None:
                return YahooMarketHistory.model_validate_json(cached)

        try:
            result = await asyncio.to_thread(
                self._fetch_price_history,
                normalized_symbol,
                normalized_period,
                start,
                end,
                repair,
                currency,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Yahoo price-history retrieval failed for {normalized_symbol}: {exc}"
            ) from exc
        if make_cache:
            self._cache.save(cache_path, result.model_dump_json())
        return result

    async def get_latest_close(
        self,
        symbol: str,
        *,
        currency: Optional[str] = None,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> Decimal:
        history = await self.get_price_history(
            symbol,
            period="5d",
            currency=currency,
            use_cache=use_cache,
            make_cache=make_cache,
        )
        prices = [row for row in history.rows if row.close is not None]
        if not prices:
            raise RuntimeError(f"Yahoo returned no closing price for {history.symbol}")
        return max(prices, key=lambda row: row.observed_on).close

    def _fetch_company_financials(self, symbol: str) -> YahooCompanyFinancials:
        ticker = self._ticker_factory(symbol)
        statement_tables = {
            "annual_income_statements": ticker.get_income_stmt(freq="yearly"),
            "quarterly_income_statements": ticker.get_income_stmt(freq="quarterly"),
            "annual_balance_sheets": ticker.get_balance_sheet(freq="yearly"),
            "quarterly_balance_sheets": ticker.get_balance_sheet(freq="quarterly"),
            "annual_cash_flow_statements": ticker.get_cash_flow(freq="yearly"),
            "quarterly_cash_flow_statements": ticker.get_cash_flow(freq="quarterly"),
        }
        if not any(not table.empty for table in statement_tables.values()):
            raise ValueError("Yahoo returned no financial statements")

        info = ticker.get_info() or {}
        currency = info.get("financialCurrency") or info.get("currency")
        if not currency:
            raise ValueError("Yahoo did not identify the reporting currency")
        company_name = info.get("longName") or info.get("shortName") or symbol
        exchange = info.get("fullExchangeName") or info.get("exchange")
        return YahooCompanyFinancials(
            symbol=str(info.get("symbol") or symbol).upper(),
            company_name=str(company_name),
            currency=str(currency).upper(),
            exchange=str(exchange) if exchange else None,
            sector=str(info["sector"]) if info.get("sector") else None,
            industry=str(info["industry"]) if info.get("industry") else None,
            country=str(info["country"]) if info.get("country") else None,
            beta=(Decimal(str(info["beta"])) if info.get("beta") is not None else None),
            market_capitalization=(
                Decimal(str(info["marketCap"]))
                if info.get("marketCap") is not None
                else None
            ),
            retrieved_at=datetime.datetime.now(datetime.timezone.utc),
            **{
                name: self._dataframe_to_reports(table)
                for name, table in statement_tables.items()
            },
        )

    def _fetch_price_history(
        self,
        symbol: str,
        period: str,
        start: Optional[datetime.date],
        end: Optional[datetime.date],
        repair: bool,
        currency: Optional[str],
    ) -> YahooMarketHistory:
        ticker = self._ticker_factory(symbol)
        history_kwargs = {
            "interval": "1d",
            "actions": True,
            "auto_adjust": False,
            "repair": repair,
            "keepna": True,
            "raise_errors": True,
        }
        if start is not None or end is not None:
            history_kwargs.update(start=start, end=end)
        else:
            history_kwargs["period"] = period.lower()
        table = ticker.history(**history_kwargs)
        if table.empty:
            raise ValueError("Yahoo returned no price history")
        metadata = ticker.get_history_metadata(repair=repair) or {}
        source_currency = currency or metadata.get("currency")
        if not source_currency:
            raise ValueError("Yahoo did not identify the market currency")
        exchange = metadata.get("fullExchangeName") or metadata.get("exchangeName")

        rows = []
        for index, row in table.iterrows():
            values = {
                name: self._decimal(row.get(name))
                for name in (
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Dividends",
                    "Stock Splits",
                )
            }
            volume_value = self._decimal(row.get("Volume"))
            rows.append(
                YahooPriceRow(
                    observed_on=index.date(),
                    open=values["Open"],
                    high=values["High"],
                    low=values["Low"],
                    close=values["Close"],
                    adjusted_close=values["Adj Close"],
                    volume=int(volume_value) if volume_value is not None else None,
                    dividend=self._positive(values["Dividends"]),
                    split_factor=self._positive(values["Stock Splits"]),
                )
            )
        return YahooMarketHistory(
            symbol=symbol,
            currency=str(source_currency),
            exchange=str(exchange) if exchange else None,
            retrieved_at=datetime.datetime.now(datetime.timezone.utc),
            source_version=f"yfinance/{yf.__version__};repair={str(repair).lower()}",
            rows=tuple(rows),
        )

    @classmethod
    def _dataframe_to_reports(cls, table: pd.DataFrame):
        reports = []
        for column in table.columns:
            values = {}
            for source_concept, raw_value in table[column].items():
                value = cls._decimal(raw_value)
                if value is not None:
                    values[str(source_concept)] = value
            if values:
                reports.append(
                    YahooFinancialReport(period_end=column.date(), values=values)
                )
        return tuple(sorted(reports, key=lambda report: report.period_end))

    @staticmethod
    def _decimal(value) -> Optional[Decimal]:
        if value is None or pd.isna(value):
            return None
        normalized = Decimal(str(value))
        return normalized if normalized.is_finite() else None

    @staticmethod
    def _positive(value: Optional[Decimal]) -> Optional[Decimal]:
        return value if value is not None and value > 0 else None

    @classmethod
    def _normalize_symbol(cls, symbol: str) -> str:
        normalized = str(symbol).strip().upper()
        if not cls._SYMBOL_PATTERN.fullmatch(normalized):
            raise ValueError(f"Invalid Yahoo symbol: {symbol!r}")
        return normalized

    @classmethod
    def _normalize_period(cls, period: str) -> str:
        normalized = str(period).strip().upper()
        if not cls._PERIOD_PATTERN.fullmatch(normalized):
            raise ValueError(f"Invalid Yahoo history period: {period!r}")
        return normalized

    @staticmethod
    def _history_variant(
        period: str,
        start: Optional[datetime.date],
        end: Optional[datetime.date],
        repair: bool,
    ) -> str:
        date_range = (
            f"{start.isoformat() if start else 'open'}_"
            f"{end.isoformat() if end else 'open'}"
            if start is not None or end is not None
            else period.lower()
        )
        return f"{date_range}_repair-{str(repair).lower()}"


__all__ = ["YahooFinanceClient"]
