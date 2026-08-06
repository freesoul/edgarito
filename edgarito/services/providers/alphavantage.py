import asyncio
import json
import logging
import re
import time
from enum import Enum
from typing import Optional, TypeVar

import aiohttp
from pydantic import BaseModel

from edgarito.schemas.providers.alphavantage.fundamentals import (
    AlphaVantageCompanyFinancials,
    BalanceSheetResponse,
    CashFlowResponse,
    CompanyOverview,
    IncomeStatementResponse,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache


class AlphaVantageFunction(str, Enum):
    OVERVIEW = "OVERVIEW"
    INCOME_STATEMENT = "INCOME_STATEMENT"
    BALANCE_SHEET = "BALANCE_SHEET"
    CASH_FLOW = "CASH_FLOW"


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class AlphaVantageClient:
    """Retrieve company fundamentals from the Alpha Vantage REST API."""

    _BASE_URL = "https://www.alphavantage.co/query"
    _SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._-]*$")

    def __init__(
        self,
        cache: FileSystemCache,
        api_key: Optional[str],
        session: Optional[aiohttp.ClientSession] = None,
        min_request_interval: float = 1.05,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("An Alpha Vantage API key is required")
        if min_request_interval < 0:
            raise ValueError("min_request_interval cannot be negative")
        self._logger = logging.getLogger(__class__.__name__)
        self._cache = cache
        self._api_key = api_key
        self._session = session
        self._owns_session = session is None
        self._min_request_interval = min_request_interval
        self._request_lock = asyncio.Lock()
        self._last_request_at: Optional[float] = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._owns_session and self._session is not None:
            await self._session.close()

    async def get_company_financials(
        self,
        symbol: str,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> AlphaVantageCompanyFinancials:
        normalized_symbol = self._normalize_symbol(symbol)
        overview = await self.get_overview(
            normalized_symbol, use_cache=use_cache, make_cache=make_cache
        )
        income_statement = await self.get_income_statement(
            normalized_symbol, use_cache=use_cache, make_cache=make_cache
        )
        balance_sheet = await self.get_balance_sheet(
            normalized_symbol, use_cache=use_cache, make_cache=make_cache
        )
        cash_flow = await self.get_cash_flow(
            normalized_symbol, use_cache=use_cache, make_cache=make_cache
        )
        return AlphaVantageCompanyFinancials(
            overview=overview,
            income_statement=income_statement,
            balance_sheet=balance_sheet,
            cash_flow=cash_flow,
        )

    async def get_overview(
        self, symbol: str, use_cache: bool = True, make_cache: bool = True
    ) -> CompanyOverview:
        return await self._get(
            AlphaVantageFunction.OVERVIEW,
            symbol,
            CompanyOverview,
            use_cache,
            make_cache,
        )

    async def get_income_statement(
        self, symbol: str, use_cache: bool = True, make_cache: bool = True
    ) -> IncomeStatementResponse:
        return await self._get(
            AlphaVantageFunction.INCOME_STATEMENT,
            symbol,
            IncomeStatementResponse,
            use_cache,
            make_cache,
        )

    async def get_balance_sheet(
        self, symbol: str, use_cache: bool = True, make_cache: bool = True
    ) -> BalanceSheetResponse:
        return await self._get(
            AlphaVantageFunction.BALANCE_SHEET,
            symbol,
            BalanceSheetResponse,
            use_cache,
            make_cache,
        )

    async def get_cash_flow(
        self, symbol: str, use_cache: bool = True, make_cache: bool = True
    ) -> CashFlowResponse:
        return await self._get(
            AlphaVantageFunction.CASH_FLOW,
            symbol,
            CashFlowResponse,
            use_cache,
            make_cache,
        )

    async def _get(
        self,
        function: AlphaVantageFunction,
        symbol: str,
        response_model: type[ResponseModel],
        use_cache: bool,
        make_cache: bool,
    ) -> ResponseModel:
        normalized_symbol = self._normalize_symbol(symbol)
        cache_path = (
            f"providers/alphavantage/{normalized_symbol}/{function.value.lower()}.json"
        )
        if use_cache:
            cached_data = self._cache.read(cache_path)
            if cached_data is not None:
                return response_model.model_validate_json(cached_data)

        data = await self._fetch(function, normalized_symbol)
        response = response_model.model_validate(data)
        if make_cache:
            self._cache.save(cache_path, json.dumps(data))
        return response

    async def _fetch(self, function: AlphaVantageFunction, symbol: str) -> dict:
        if self._session is None:
            self._session = aiohttp.ClientSession()

        params = {
            "function": function.value,
            "symbol": symbol,
            "apikey": self._api_key,
        }
        for attempt in range(2):
            response_text = await self._request(function, symbol, params)
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Alpha Vantage returned invalid JSON for {function.value} {symbol}"
                ) from exc

            error_key = next(
                (
                    key
                    for key in ("Error Message", "Information", "Note")
                    if key in data
                ),
                None,
            )
            if error_key is not None:
                message = self._redact_secrets(str(data[error_key]))
                is_burst_limit = "1 request per second" in message.lower()
                if is_burst_limit and attempt == 0:
                    continue
                raise RuntimeError(f"Alpha Vantage: {message}")
            if not data:
                raise RuntimeError(
                    "Alpha Vantage returned an empty response for "
                    f"{function.value} {symbol}"
                )
            return data

        raise RuntimeError(
            f"Alpha Vantage request failed after retry for {function.value} {symbol}"
        )

    def _redact_secrets(self, message: str) -> str:
        return message.replace(self._api_key, "[REDACTED]")

    async def _request(
        self, function: AlphaVantageFunction, symbol: str, params: dict
    ) -> str:
        async with self._request_lock:
            if self._last_request_at is not None:
                elapsed = time.monotonic() - self._last_request_at
                delay = self._min_request_interval - elapsed
                if delay > 0:
                    await asyncio.sleep(delay)

            self._last_request_at = time.monotonic()
            try:
                async with self._session.get(
                    self._BASE_URL, params=params, timeout=20
                ) as response:
                    response_text = await response.text()
                    if response.status >= 400:
                        raise RuntimeError(
                            "Alpha Vantage request failed with HTTP "
                            f"{response.status} for {function.value} {symbol}"
                        )
                    return response_text
            except (aiohttp.ClientError, TimeoutError) as exc:
                raise RuntimeError(
                    f"Alpha Vantage request failed for {function.value} {symbol}"
                ) from exc

    @classmethod
    def _normalize_symbol(cls, symbol: str) -> str:
        normalized = symbol.strip().upper()
        if not cls._SYMBOL_PATTERN.fullmatch(normalized):
            raise ValueError(f"Invalid Alpha Vantage symbol: {symbol!r}")
        return normalized
