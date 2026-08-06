import json
import re
from enum import Enum
from typing import Optional, TypeVar

import aiohttp
from pydantic import RootModel

from edgarito.schemas.providers.fmp.fundamentals import (
    BalanceSheetResponse,
    CashFlowStatementResponse,
    CompanyProfile,
    CompanyProfileResponse,
    FmpCompanyFinancials,
    IncomeStatementResponse,
    SecuritySearchResult,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache


class FmpEndpoint(str, Enum):
    PROFILE = "profile"
    INCOME_STATEMENT = "income-statement"
    BALANCE_SHEET = "balance-sheet-statement"
    CASH_FLOW = "cash-flow-statement"


class FmpPeriod(str, Enum):
    ANNUAL = "annual"
    QUARTER = "quarter"


ResponseModel = TypeVar("ResponseModel", bound=RootModel)


class FmpClient:
    """Retrieve standardized company fundamentals from FMP's stable API."""

    _BASE_URL = "https://financialmodelingprep.com/stable"
    _SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._-]*$")

    def __init__(
        self,
        cache: FileSystemCache,
        api_key: Optional[str],
        session: Optional[aiohttp.ClientSession] = None,
        statement_limit: int = 5,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("An FMP API key is required")
        if statement_limit < 1:
            raise ValueError("statement_limit must be at least 1")
        self._cache = cache
        self._api_key = api_key
        self._session = session
        self._owns_session = session is None
        self._statement_limit = statement_limit

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
    ) -> FmpCompanyFinancials:
        normalized_symbol = self._normalize_symbol(symbol)
        profile = await self.get_profile(
            normalized_symbol, use_cache=use_cache, make_cache=make_cache
        )
        annual_income = await self.get_income_statements(
            normalized_symbol,
            FmpPeriod.ANNUAL,
            use_cache=use_cache,
            make_cache=make_cache,
        )
        quarterly_income = await self.get_income_statements(
            normalized_symbol,
            FmpPeriod.QUARTER,
            use_cache=use_cache,
            make_cache=make_cache,
        )
        annual_balance = await self.get_balance_sheets(
            normalized_symbol,
            FmpPeriod.ANNUAL,
            use_cache=use_cache,
            make_cache=make_cache,
        )
        quarterly_balance = await self.get_balance_sheets(
            normalized_symbol,
            FmpPeriod.QUARTER,
            use_cache=use_cache,
            make_cache=make_cache,
        )
        annual_cash_flow = await self.get_cash_flow_statements(
            normalized_symbol,
            FmpPeriod.ANNUAL,
            use_cache=use_cache,
            make_cache=make_cache,
        )
        quarterly_cash_flow = await self.get_cash_flow_statements(
            normalized_symbol,
            FmpPeriod.QUARTER,
            use_cache=use_cache,
            make_cache=make_cache,
        )
        return FmpCompanyFinancials(
            profile=profile,
            annual_income_statements=annual_income.root,
            quarterly_income_statements=quarterly_income.root,
            annual_balance_sheets=annual_balance.root,
            quarterly_balance_sheets=quarterly_balance.root,
            annual_cash_flow_statements=annual_cash_flow.root,
            quarterly_cash_flow_statements=quarterly_cash_flow.root,
        )

    async def get_profile(
        self, symbol: str, use_cache: bool = True, make_cache: bool = True
    ) -> CompanyProfile:
        response = await self._get(
            FmpEndpoint.PROFILE,
            symbol,
            CompanyProfileResponse,
            period=None,
            use_cache=use_cache,
            make_cache=make_cache,
        )
        if not response.root:
            raise ValueError(f"FMP did not find a profile for {symbol.upper()}")
        return response.root[0]

    async def search_isin(
        self, isin: str, use_cache: bool = True, make_cache: bool = True
    ) -> list[SecuritySearchResult]:
        return await self._search_identifier(
            "search-isin", "isin", isin.strip().upper(), use_cache, make_cache
        )

    async def search_cik(
        self, cik: int, use_cache: bool = True, make_cache: bool = True
    ) -> list[SecuritySearchResult]:
        return await self._search_identifier(
            "search-cik", "cik", str(cik).zfill(10), use_cache, make_cache
        )

    async def search_exchange_variants(
        self, symbol: str, use_cache: bool = True, make_cache: bool = True
    ) -> list[SecuritySearchResult]:
        return await self._search_identifier(
            "search-exchange-variants",
            "symbol",
            self._normalize_symbol(symbol),
            use_cache,
            make_cache,
        )

    async def get_income_statements(
        self,
        symbol: str,
        period: FmpPeriod,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> IncomeStatementResponse:
        return await self._get(
            FmpEndpoint.INCOME_STATEMENT,
            symbol,
            IncomeStatementResponse,
            period,
            use_cache,
            make_cache,
        )

    async def get_balance_sheets(
        self,
        symbol: str,
        period: FmpPeriod,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> BalanceSheetResponse:
        return await self._get(
            FmpEndpoint.BALANCE_SHEET,
            symbol,
            BalanceSheetResponse,
            period,
            use_cache,
            make_cache,
        )

    async def get_cash_flow_statements(
        self,
        symbol: str,
        period: FmpPeriod,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> CashFlowStatementResponse:
        return await self._get(
            FmpEndpoint.CASH_FLOW,
            symbol,
            CashFlowStatementResponse,
            period,
            use_cache,
            make_cache,
        )

    async def _get(
        self,
        endpoint: FmpEndpoint,
        symbol: str,
        response_model: type[ResponseModel],
        period: Optional[FmpPeriod],
        use_cache: bool,
        make_cache: bool,
    ) -> ResponseModel:
        normalized_symbol = self._normalize_symbol(symbol)
        suffix = f"_{period.value}" if period else ""
        cache_path = f"providers/fmp/{normalized_symbol}/{endpoint.value}{suffix}.json"
        if use_cache:
            cached_data = self._cache.read(cache_path)
            if cached_data is not None:
                return response_model.model_validate_json(cached_data)

        data = await self._fetch(endpoint, normalized_symbol, period)
        response = response_model.model_validate(data)
        if make_cache:
            self._cache.save(cache_path, json.dumps(data))
        return response

    async def _fetch(
        self,
        endpoint: FmpEndpoint,
        symbol: str,
        period: Optional[FmpPeriod],
    ) -> list:
        if self._session is None:
            self._session = aiohttp.ClientSession()

        params = {"symbol": symbol, "apikey": self._api_key}
        if period is not None:
            params.update(period=period.value, limit=str(self._statement_limit))
        url = f"{self._BASE_URL}/{endpoint.value}"
        try:
            async with self._session.get(url, params=params, timeout=20) as response:
                response_text = await response.text()
                if response.status == 404:
                    raise FileNotFoundError(
                        f"FMP endpoint not found: {endpoint.value} {symbol}"
                    )
                if response.status >= 400:
                    raise RuntimeError(
                        f"FMP request failed with HTTP {response.status} for "
                        f"{endpoint.value} {symbol}"
                    )
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise RuntimeError(
                f"FMP request failed for {endpoint.value} {symbol}"
            ) from exc

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"FMP returned invalid JSON for {endpoint.value} {symbol}"
            ) from exc
        if isinstance(data, dict):
            message = data.get("Error Message") or data.get("message") or data
            raise RuntimeError(f"FMP: {message}")
        if not isinstance(data, list):
            raise RuntimeError(
                f"FMP returned an unexpected response for {endpoint.value} {symbol}"
            )
        return data

    async def _search_identifier(
        self,
        endpoint: str,
        parameter: str,
        value: str,
        use_cache: bool,
        make_cache: bool,
    ) -> list[SecuritySearchResult]:
        cache_value = re.sub(r"[^A-Z0-9._-]", "_", value.upper())
        cache_path = f"providers/fmp/search/{endpoint}/{cache_value}.json"
        if use_cache:
            cached_data = self._cache.read(cache_path)
            if cached_data is not None:
                return [
                    SecuritySearchResult.model_validate(item)
                    for item in json.loads(cached_data)
                ]

        if self._session is None:
            self._session = aiohttp.ClientSession()
        url = f"{self._BASE_URL}/{endpoint}"
        params = {parameter: value, "apikey": self._api_key}
        try:
            async with self._session.get(url, params=params, timeout=20) as response:
                response_text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(
                        f"FMP identifier search failed with HTTP {response.status} "
                        f"for {parameter} {value}"
                    )
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise RuntimeError(
                f"FMP identifier search failed for {parameter} {value}"
            ) from exc

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("FMP identifier search returned invalid JSON") from exc
        if isinstance(data, dict):
            message = (
                data.get("Error Message") or data.get("error") or data.get("message")
            )
            raise RuntimeError(f"FMP identifier search failed: {message or data}")
        if not isinstance(data, list):
            raise RuntimeError("FMP identifier search returned an unexpected response")
        if make_cache:
            self._cache.save(cache_path, json.dumps(data))
        return [SecuritySearchResult.model_validate(item) for item in data]

    @classmethod
    def _normalize_symbol(cls, symbol: str) -> str:
        normalized = symbol.strip().upper()
        if not cls._SYMBOL_PATTERN.fullmatch(normalized):
            raise ValueError(f"Invalid FMP symbol: {symbol!r}")
        return normalized
