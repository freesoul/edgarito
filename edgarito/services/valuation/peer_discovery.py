from __future__ import annotations

import asyncio
import json
import math
import re
from typing import TYPE_CHECKING, Callable, Optional, Protocol

import aiohttp
import yfinance as yf
from pydantic import BaseModel, ConfigDict, Field

from edgarito.schemas.providers.yahoo.fundamentals import YahooCompanyFinancials
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.valuation.issuer_identity import issuer_identity_keys
from edgarito.services.valuation.models import PeerDiscoveryResult

if TYPE_CHECKING:
    from edgarito.services.openai import OpenAIClient


class PeerCandidateDiscoveryProvider(Protocol):
    async def discover(
        self, target: YahooCompanyFinancials, *, max_candidates: int = 30
    ) -> PeerDiscoveryResult: ...


class OpenAIPeerDiscoveryResponse(BaseModel):
    """Structured ticker suggestions returned by the OpenAI peer finder."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(
        default_factory=list,
        description=(
            "Public equity ticker symbols for economically comparable companies"
        ),
    )


class OpenAIPeerDiscoveryProvider:
    """Use OpenAI to suggest a bounded, unverified peer ticker universe."""

    _SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9._^-]*$")
    _INSTRUCTIONS = """
Suggest publicly traded companies that are economically comparable to the target
company. Prefer the same business model, industry, geography, and scale when that
information is available. Return only equity ticker symbols, not company names,
explanations, indices, funds, or analyst recommendations. Do not return the target
company or another listing, ADR, or depositary receipt of the same issuer. Ticker
suggestions are unverified candidates and will be validated against Yahoo Finance
and ranked by deterministic downstream valuation logic.
""".strip()

    def __init__(self, openai_client: OpenAIClient):
        self._openai = openai_client

    async def discover(
        self, target: YahooCompanyFinancials, *, max_candidates: int = 30
    ) -> PeerDiscoveryResult:
        symbol = target.symbol.strip().upper()
        if not self._SYMBOL.fullmatch(symbol):
            raise ValueError(f"Invalid OpenAI peer-discovery ticker: {target.symbol!r}")

        response = await self._openai.extract_structured(
            instructions=self._INSTRUCTIONS,
            content=self._content(target, symbol, max_candidates),
            response_model=OpenAIPeerDiscoveryResponse,
        )

        limit = max(0, max_candidates)
        target_keys = set(
            issuer_identity_keys(
                company_id=target.symbol,
                company_name=target.company_name,
                ticker=target.symbol,
                identifiers=target.identifiers,
            )
        )
        seen_issuer_keys = set(target_keys)
        candidates: list[str] = []
        invalid_count = 0
        duplicate_count = 0
        target_listing_count = 0

        for value in response.tickers:
            if not isinstance(value, str):
                invalid_count += 1
                continue
            candidate = value.strip().upper()
            if not self._SYMBOL.fullmatch(candidate):
                invalid_count += 1
                continue
            candidate_keys = issuer_identity_keys(
                company_id=candidate,
                ticker=candidate,
            )
            if candidate_keys & seen_issuer_keys:
                if candidate_keys & target_keys:
                    target_listing_count += 1
                else:
                    duplicate_count += 1
                continue
            if len(candidates) >= limit:
                break
            seen_issuer_keys.update(candidate_keys)
            candidates.append(candidate)

        confidence = (
            "high"
            if len(candidates) >= 8
            else "medium"
            if len(candidates) >= 5
            else "low"
        )
        warnings = []
        if invalid_count:
            warnings.append(
                f"OpenAI returned {invalid_count} invalid ticker symbol(s); "
                "they were skipped"
            )
        if target_listing_count:
            warnings.append(
                "OpenAI returned "
                f"{target_listing_count} target or cross-listing ticker(s); they were skipped"
            )
        if duplicate_count:
            warnings.append(
                f"OpenAI returned {duplicate_count} duplicate issuer ticker(s); "
                "they were skipped"
            )
        if confidence == "low":
            warnings.append(
                f"OpenAI returned only {len(candidates)} unique peer ticker(s)"
            )
        return PeerDiscoveryResult(
            provider="openai",
            target_ticker=symbol,
            candidate_tickers=tuple(candidates),
            methodology=(
                "OpenAI structured peer suggestions normalized and de-duplicated by "
                "issuer identity; Yahoo validation and downstream economic scoring "
                "remain required"
            ),
            confidence=confidence,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _content(
        target: YahooCompanyFinancials, symbol: str, max_candidates: int
    ) -> str:
        return "\n".join(
            (
                f"Target ticker: {symbol}",
                f"Target company: {target.company_name.strip()}",
                f"Sector: {(target.sector or 'unknown').strip()}",
                f"Industry: {(target.industry or 'unknown').strip()}",
                f"Country: {(target.country or 'unknown').strip()}",
                f"Exchange: {(target.exchange or 'unknown').strip()}",
                f"Currency: {target.currency.strip().upper()}",
                f"Return at most {max_candidates} candidate ticker symbols.",
            )
        )


class MassiveRelatedCompaniesPeerDiscoveryProvider:
    """Return U.S. discovery hints from Massive's related-companies endpoint."""

    _BASE_URL = "https://api.massive.com/v1/related-companies"
    _SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9._^-]*$")

    def __init__(
        self,
        cache: FileSystemCache,
        api_key: str,
        *,
        session: Optional[aiohttp.ClientSession] = None,
        use_cache: bool = True,
        make_cache: bool = True,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("A Massive API key is required")
        self._cache = cache
        self._api_key = api_key.strip()
        self._session = session
        self._use_cache = use_cache
        self._make_cache = make_cache

    async def discover(
        self, target: YahooCompanyFinancials, *, max_candidates: int = 30
    ) -> PeerDiscoveryResult:
        symbol = target.symbol.strip().upper()
        if not self._SYMBOL.fullmatch(symbol):
            raise ValueError(f"Invalid Massive ticker: {target.symbol!r}")
        cache_path = f"providers/massive/related-companies/{symbol}.json"
        payload = None
        if self._use_cache:
            cached = self._cache.read(cache_path)
            if cached is not None:
                try:
                    payload = json.loads(cached)
                except json.JSONDecodeError:
                    payload = None
        if payload is None:
            payload = await self._retrieve(symbol)
            if self._make_cache:
                self._cache.save(cache_path, json.dumps(payload, sort_keys=True))

        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise RuntimeError("Massive related companies returned an invalid response")
        candidates = []
        seen = set(
            issuer_identity_keys(
                company_id=target.symbol,
                company_name=target.company_name,
                ticker=target.symbol,
                identifiers=target.identifiers,
            )
        )
        for item in results:
            candidate = (
                str(item.get("ticker") or "").strip().upper()
                if isinstance(item, dict)
                else ""
            )
            candidate_keys = issuer_identity_keys(
                company_id=candidate,
                ticker=candidate,
            )
            if not self._SYMBOL.fullmatch(candidate) or candidate_keys & seen:
                continue
            seen.update(candidate_keys)
            candidates.append(candidate)
            if len(candidates) >= max_candidates:
                break
        confidence = (
            "high"
            if len(candidates) >= 8
            else "medium"
            if len(candidates) >= 5
            else "low"
        )
        warnings = (
            "Massive related-company results reflect news co-mentions and return "
            "relationships; they are discovery hints only, not comparable evidence. "
            "Industry/product-economics and observable-similarity gates remain "
            "required",
        )
        if confidence == "low":
            warnings = (
                *warnings,
                f"Massive returned only {len(candidates)} unique related tickers",
            )
        return PeerDiscoveryResult(
            provider="massive-related",
            target_ticker=symbol,
            candidate_tickers=tuple(candidates),
            methodology=(
                "Massive Related Tickers supplies discovery hints derived from news "
                "co-mentions and return relationships; it is not comparable evidence. "
                "Downstream industry/product-economics or observable-similarity "
                "gating and primary evidence-group selection are required"
            ),
            confidence=confidence,
            warnings=warnings,
        )

    async def _retrieve(self, symbol: str) -> dict:
        owns_session = self._session is None
        session = self._session or aiohttp.ClientSession()
        try:
            async with session.get(
                f"{self._BASE_URL}/{symbol}",
                params={"apiKey": self._api_key},
                timeout=20,
            ) as response:
                content = await response.text()
                if response.status == 429:
                    raise RuntimeError("Massive related companies rate limit reached")
                if response.status >= 400:
                    raise RuntimeError(
                        "Massive related companies failed with HTTP "
                        f"{response.status} for {symbol}"
                    )
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise RuntimeError(
                f"Massive related companies retrieval failed for {symbol}"
            ) from exc
        finally:
            if owns_session:
                await session.close()
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Massive related companies returned invalid JSON"
            ) from exc


class YahooScreenerPeerDiscoveryProvider:
    """Discover broad, provider-supported candidates for deterministic ranking."""

    _SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9._^-]*$")
    _PEER_GROUPS = (
        (
            re.compile(r"\bsoftware\b|information technology services"),
            "Software & Services",
        ),
        (re.compile(r"\bsemiconductors?\b"), "Semiconductors"),
        (
            re.compile(r"computer hardware|consumer electronics|electronic components"),
            "Technology Hardware",
        ),
        (re.compile(r"auto manufacturers?|automobiles?"), "Automobiles"),
        (re.compile(r"auto parts?|auto components?"), "Auto Components"),
        (re.compile(r"apparel|luxury goods|textiles?"), "Textiles & Apparel"),
        (re.compile(r"aerospace|defen[cs]e"), "Aerospace & Defense"),
        (re.compile(r"banks?|banking"), "Banks"),
        (re.compile(r"chemicals?"), "Chemicals"),
        (re.compile(r"building products?"), "Building Products"),
        (re.compile(r"construction materials?"), "Construction Materials"),
    )
    _EUROPE_REGIONS = frozenset(
        {
            "at",
            "be",
            "ch",
            "cz",
            "de",
            "dk",
            "ee",
            "es",
            "fi",
            "fr",
            "gb",
            "gr",
            "hu",
            "ie",
            "is",
            "it",
            "lt",
            "lv",
            "nl",
            "no",
            "pl",
            "pt",
            "ro",
            "se",
        }
    )
    _COUNTRY_REGIONS = {
        "austria": "at",
        "belgium": "be",
        "canada": "ca",
        "czech republic": "cz",
        "denmark": "dk",
        "finland": "fi",
        "france": "fr",
        "germany": "de",
        "ireland": "ie",
        "italy": "it",
        "netherlands": "nl",
        "norway": "no",
        "portugal": "pt",
        "spain": "es",
        "sweden": "se",
        "switzerland": "ch",
        "united kingdom": "gb",
        "united states": "us",
        "united states of america": "us",
        "usa": "us",
    }
    _SYMBOL_REGIONS = {
        ".AS": "nl",
        ".BR": "be",
        ".CO": "dk",
        ".DE": "de",
        ".HE": "fi",
        ".IR": "ie",
        ".L": "gb",
        ".LS": "pt",
        ".MC": "es",
        ".MI": "it",
        ".OL": "no",
        ".PA": "fr",
        ".PR": "cz",
        ".ST": "se",
        ".SW": "ch",
        ".VI": "at",
        ".WA": "pl",
    }

    def __init__(
        self,
        cache: FileSystemCache,
        *,
        screen: Callable = yf.screen,
        use_cache: bool = True,
        make_cache: bool = True,
    ):
        self._cache = cache
        self._screen = screen
        self._use_cache = use_cache
        self._make_cache = make_cache

    async def discover(
        self, target: YahooCompanyFinancials, *, max_candidates: int = 30
    ) -> PeerDiscoveryResult:
        sector = (target.sector or "").strip()
        peer_group = self._peer_group(target.industry)
        if not sector and peer_group is None:
            return self._unavailable(
                target.symbol,
                "Yahoo did not provide an industry or sector for broad discovery",
            )
        valid_sectors = yf.EquityQuery("eq", ["region", "us"]).valid_values["sector"]
        if peer_group is None and sector not in valid_sectors:
            return self._unavailable(
                target.symbol,
                f"Yahoo screener does not support the reported sector {sector!r}",
            )
        field = "peer_group" if peer_group is not None else "sector"
        filter_value = peer_group or sector
        cache_key = re.sub(r"[^a-z0-9]+", "-", filter_value.casefold()).strip("-")
        base_query = yf.EquityQuery("eq", [field, filter_value])
        target_region = self._region(target)
        region_group = self._region_group(target_region)
        query = self._regional_query(base_query, target_region)
        scope = region_group or target_region or "global"
        cache_path = f"providers/yahoo/screeners/{field}-{cache_key}-{scope}.json"
        warnings = [
            "Yahoo peer-group/sector screening supplies discovery support only, not "
            "comparable evidence; downstream industry/product-economics or "
            "observable-similarity gates remain required"
        ]
        try:
            response = await self._response(query, cache_path)
        except Exception as exc:
            if target_region is None:
                return self._unavailable(
                    target.symbol, f"Yahoo sector screener failed: {exc}"
                )
            warnings.append(
                f"Yahoo {scope} industry screen failed; global screen used: {exc}"
            )
            try:
                response = await self._response(
                    base_query,
                    f"providers/yahoo/screeners/{field}-{cache_key}-global.json",
                )
            except Exception as fallback_exc:
                return self._unavailable(
                    target.symbol,
                    f"Yahoo regional and global screens failed: {fallback_exc}",
                )

        quotes = response.get("quotes", []) if isinstance(response, dict) else []
        minimum_regional = min(5, max_candidates)
        if target_region is not None and len(quotes) < minimum_regional:
            try:
                global_response = await self._response(
                    base_query,
                    f"providers/yahoo/screeners/{field}-{cache_key}-global.json",
                )
                global_quotes = (
                    global_response.get("quotes", [])
                    if isinstance(global_response, dict)
                    else []
                )
                quotes = [*quotes, *global_quotes]
                warnings.append(
                    f"Yahoo {scope} screen returned fewer than {minimum_regional} "
                    "quotes; global discovery candidates were added for coverage"
                )
            except Exception as exc:
                warnings.append(f"Yahoo global fallback screen failed: {exc}")

        target_cap = target.market_capitalization
        preferred_exchanges = self._preferred_exchanges(target.exchange)
        target_issuer_keys = issuer_identity_keys(
            company_id=target.symbol,
            company_name=target.company_name,
            ticker=target.symbol,
            identifiers=target.identifiers,
        )
        ranked_candidates = []
        out_of_band_market_caps = 0
        primary_names = {
            str(quote.get("symbol") or "").strip().upper(): quote.get("longName")
            for quote in quotes
            if quote.get("longName")
        }
        for quote in quotes:
            symbol = str(quote.get("symbol") or "").strip().upper()
            if (
                not self._SYMBOL.fullmatch(symbol)
                or symbol == target.symbol.upper()
                or quote.get("quoteType") not in {None, "EQUITY"}
            ):
                continue
            market_cap = quote.get("marketCap") or quote.get("intradaymarketcap")
            try:
                candidate_cap = float(market_cap) if market_cap is not None else None
            except (TypeError, ValueError):
                candidate_cap = None
            if (
                target_cap is not None
                and target_cap > 0
                and candidate_cap
                and candidate_cap > 0
            ):
                cap_ratio = candidate_cap / float(target_cap)
                distance = abs(math.log10(cap_ratio))
                if cap_ratio < 0.25 or cap_ratio > 4:
                    out_of_band_market_caps += 1
            else:
                distance = math.inf
            exchange = str(quote.get("exchange") or "").upper()
            exchange_rank = (
                0
                if exchange in preferred_exchanges
                else 1
                if "." not in symbol and symbol.isalpha()
                else 2
            )
            candidate_region = self._quote_region(quote, symbol)
            geography_rank = (
                0
                if target_region and candidate_region == target_region
                else 1
                if region_group and self._region_group(candidate_region) == region_group
                else 2
            )
            primary_symbol = symbol.split(".", maxsplit=1)[0]
            name = str(
                quote.get("longName")
                or primary_names.get(primary_symbol)
                or quote.get("shortName")
                or symbol
            ).casefold()
            candidate_issuer_keys = issuer_identity_keys(
                company_id=symbol,
                company_name=name,
                ticker=symbol,
            )
            if target_issuer_keys & candidate_issuer_keys:
                continue
            rank = (geography_rank, exchange_rank, distance, symbol)
            ranked_candidates.append((rank, symbol, candidate_issuer_keys))
        symbols = []
        seen_issuer_keys = set()
        for _rank, symbol, candidate_issuer_keys in sorted(ranked_candidates):
            if candidate_issuer_keys & seen_issuer_keys:
                continue
            seen_issuer_keys.update(candidate_issuer_keys)
            symbols.append(symbol)
            if len(symbols) >= max_candidates:
                break
        symbols = tuple(symbols)
        confidence = (
            "high" if len(symbols) >= 15 else "medium" if len(symbols) >= 5 else "low"
        )
        if confidence == "low":
            warnings.append(
                f"Only {len(symbols)} provider-supported sector candidates were discovered",
            )
        if out_of_band_market_caps:
            warnings.append(
                f"{out_of_band_market_caps} Yahoo discovery candidate(s) fell outside "
                "the default 0.25x-4x market-cap proximity band; they were retained "
                "for the selector's configured market-cap range and relaxation policy"
            )
        return PeerDiscoveryResult(
            provider="yahoo-screener",
            target_ticker=target.symbol,
            candidate_tickers=symbols,
            methodology=(
                f"Yahoo {scope} {field.replace('_', ' ')} {filter_value!r} equity "
                "screen used as discovery support with global fallback when regional "
                "coverage is sparse; candidates ordered deterministically "
                "by geography, market-cap log-distance and ticker before "
                "downstream economic/evidence-group gating; Massive network hints "
                "are not comparable evidence"
            ),
            confidence=confidence,
            warnings=tuple(warnings),
        )

    async def _response(self, query, cache_path):
        if self._use_cache:
            cached = self._cache.read(cache_path)
            if cached is not None:
                try:
                    payload = json.loads(cached)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    return payload
        response = await asyncio.to_thread(
            self._screen,
            query,
            size=100,
            sortField="intradaymarketcap",
            sortAsc=False,
        )
        if not isinstance(response, dict):
            raise RuntimeError("Yahoo screener returned an invalid response")
        if self._make_cache:
            self._cache.save(cache_path, json.dumps(response, sort_keys=True))
        return response

    @staticmethod
    def _unavailable(symbol, warning):
        return PeerDiscoveryResult(
            provider="yahoo-screener",
            target_ticker=symbol,
            methodology="Provider-backed sector/peer-group discovery support unavailable",
            confidence="low",
            warnings=(warning,),
        )

    @classmethod
    def _peer_group(cls, industry):
        normalized = (industry or "").casefold()
        return next(
            (
                group
                for pattern, group in cls._PEER_GROUPS
                if pattern.search(normalized)
            ),
            None,
        )

    @staticmethod
    def _preferred_exchanges(exchange):
        normalized = (exchange or "").casefold()
        if "nasdaq" in normalized:
            return {"NMS", "NGM", "NCM"}
        if "nyse" in normalized or "new york" in normalized:
            return {"NYQ"}
        return set()

    @classmethod
    def _region(cls, target):
        country = (target.country or "").strip().casefold()
        if country in cls._COUNTRY_REGIONS:
            return cls._COUNTRY_REGIONS[country]
        if len(country) == 2:
            return country
        symbol = target.symbol.strip().upper()
        return next(
            (
                region
                for suffix, region in cls._SYMBOL_REGIONS.items()
                if symbol.endswith(suffix)
            ),
            (
                "us"
                if any(
                    exchange in (target.exchange or "").casefold()
                    for exchange in ("nasdaq", "nyse")
                )
                else None
            ),
        )

    @classmethod
    def _regional_query(cls, base_query, region):
        if region is None:
            return base_query
        regions = cls._EUROPE_REGIONS if region in cls._EUROPE_REGIONS else (region,)
        filters = [yf.EquityQuery("eq", ["region", item]) for item in sorted(regions)]
        region_query = (
            filters[0] if len(filters) == 1 else yf.EquityQuery("or", filters)
        )
        return yf.EquityQuery("and", [region_query, base_query])

    @classmethod
    def _quote_region(cls, quote, symbol):
        region = str(quote.get("region") or "").strip().casefold()
        if region:
            return cls._COUNTRY_REGIONS.get(
                region, region if len(region) == 2 else None
            )
        return next(
            (
                value
                for suffix, value in cls._SYMBOL_REGIONS.items()
                if symbol.endswith(suffix)
            ),
            None,
        )

    @classmethod
    def _region_group(cls, region):
        return "europe" if region in cls._EUROPE_REGIONS else region


class MarketAwarePeerDiscoveryProvider:
    """Combine provider discovery hints without treating either as peer evidence."""

    def __init__(
        self,
        yahoo: PeerCandidateDiscoveryProvider,
        massive: PeerCandidateDiscoveryProvider | None = None,
        *,
        minimum_candidates: int = 5,
    ):
        self._yahoo = yahoo
        self._massive = massive
        self._minimum_candidates = minimum_candidates

    async def discover(
        self, target: YahooCompanyFinancials, *, max_candidates: int = 30
    ) -> PeerDiscoveryResult:
        if not self._is_us(target):
            yahoo = await self._yahoo.discover(target, max_candidates=max_candidates)
            return yahoo.model_copy(
                update={
                    "methodology": (
                        "Non-U.S. issuer bypassed the U.S.-only Massive source; "
                        + yahoo.methodology
                    )
                }
            )

        fallback_warnings = []
        massive = None
        if self._massive is None:
            fallback_warnings.append(
                "Massive API key is not configured; Yahoo peer-group discovery used"
            )
        else:
            try:
                massive = await self._massive.discover(
                    target, max_candidates=max_candidates
                )
            except (RuntimeError, ValueError) as exc:
                fallback_warnings.append(f"Massive discovery failed: {exc}")
            if massive is not None:
                fallback_warnings.extend(massive.warnings)
                if len(massive.candidate_tickers) < min(
                    self._minimum_candidates, max_candidates
                ):
                    fallback_warnings.append(
                        "Massive returned too few candidates; Yahoo peer-group "
                        "discovery supplemented its hints"
                    )
                else:
                    fallback_warnings.append(
                        "Massive supplied enough network hints for coverage, but Yahoo "
                        "peer-group discovery was still retrieved for supplementation; "
                        "Massive hints are not comparable evidence"
                    )

        try:
            yahoo = await self._yahoo.discover(target, max_candidates=max_candidates)
        except (RuntimeError, ValueError) as exc:
            fallback_warnings.append(f"Yahoo fallback failed: {exc}")
            yahoo = PeerDiscoveryResult(
                provider="yahoo-screener",
                target_ticker=target.symbol,
                methodology="Yahoo fallback discovery support unavailable",
                confidence="low",
            )
        combined = self._deduplicate(
            (
                *yahoo.candidate_tickers,
                *(massive.candidate_tickers if massive is not None else ()),
            ),
            target,
            max_candidates,
        )
        fallback_warnings.extend(yahoo.warnings)
        confidence = (
            "high" if len(combined) >= 10 else "medium" if len(combined) >= 5 else "low"
        )
        return PeerDiscoveryResult(
            provider=(
                "massive-related+yahoo-screener"
                if massive is not None and massive.candidate_tickers
                else "yahoo-screener"
            ),
            target_ticker=target.symbol,
            candidate_tickers=combined,
            methodology=(
                "U.S. discovery combines Yahoo peer-group/region screening with "
                "Massive Related Tickers network hints; Yahoo candidates are retained "
                "before non-evidentiary network hints, both sources are discovery "
                "support only (Massive hints are not comparable evidence), and "
                "candidates are de-duplicated before required "
                "industry/product-economics or observable-similarity and primary "
                "evidence-group gating"
            ),
            confidence=confidence,
            warnings=tuple(dict.fromkeys(fallback_warnings)),
        )

    @staticmethod
    def _deduplicate(candidates, target, maximum):
        target_symbol = (
            target.symbol if isinstance(target, YahooCompanyFinancials) else str(target)
        )
        seen = set(
            issuer_identity_keys(
                company_id=target_symbol,
                company_name=(
                    target.company_name
                    if isinstance(target, YahooCompanyFinancials)
                    else None
                ),
                ticker=target_symbol,
                identifiers=(
                    target.identifiers
                    if isinstance(target, YahooCompanyFinancials)
                    else None
                ),
            )
        )
        selected = []
        for value in candidates:
            symbol = value.strip().upper()
            candidate_keys = issuer_identity_keys(
                company_id=symbol,
                ticker=symbol,
            )
            if not symbol or candidate_keys & seen:
                continue
            seen.update(candidate_keys)
            selected.append(symbol)
            if len(selected) >= maximum:
                break
        return tuple(selected)

    @staticmethod
    def _is_us(target):
        country = (target.country or "").strip().casefold()
        if country:
            return country in {
                "us",
                "usa",
                "united states",
                "united states of america",
            }
        exchange = (target.exchange or "").casefold()
        return any(
            value in exchange for value in ("nasdaq", "nyse", "new york", "amex")
        )


__all__ = [
    "MarketAwarePeerDiscoveryProvider",
    "MassiveRelatedCompaniesPeerDiscoveryProvider",
    "OpenAIPeerDiscoveryProvider",
    "OpenAIPeerDiscoveryResponse",
    "PeerCandidateDiscoveryProvider",
    "YahooScreenerPeerDiscoveryProvider",
]
