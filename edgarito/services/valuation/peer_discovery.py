from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Callable, Protocol

import yfinance as yf

from edgarito.schemas.providers.yahoo.fundamentals import YahooCompanyFinancials
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.valuation.models import PeerDiscoveryResult


class PeerCandidateDiscoveryProvider(Protocol):
    async def discover(
        self, target: YahooCompanyFinancials, *, max_candidates: int = 30
    ) -> PeerDiscoveryResult: ...


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
        cache_path = f"providers/yahoo/screeners/{field}-{cache_key}.json"
        response = None
        if self._use_cache:
            cached = self._cache.read(cache_path)
            if cached is not None:
                response = json.loads(cached)
        if response is None:
            query = yf.EquityQuery("eq", [field, filter_value])
            try:
                response = await asyncio.to_thread(
                    self._screen,
                    query,
                    size=100,
                    sortField="intradaymarketcap",
                    sortAsc=False,
                )
            except Exception as exc:
                return self._unavailable(
                    target.symbol, f"Yahoo sector screener failed: {exc}"
                )
            if self._make_cache:
                self._cache.save(cache_path, json.dumps(response, sort_keys=True))

        target_cap = target.market_capitalization
        preferred_exchanges = self._preferred_exchanges(target.exchange)
        ranked_by_issuer = {}
        quotes = response.get("quotes", [])
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
                distance = abs(math.log10(candidate_cap / float(target_cap)))
            else:
                distance = math.inf
            exchange = str(quote.get("exchange") or "").upper()
            listing_rank = (
                0
                if exchange in preferred_exchanges
                else 1
                if "." not in symbol and symbol.isalpha()
                else 2
            )
            primary_symbol = symbol.split(".", maxsplit=1)[0]
            name = str(
                quote.get("longName")
                or primary_names.get(primary_symbol)
                or quote.get("shortName")
                or symbol
            ).casefold()
            issuer_key = re.sub(r"[^a-z0-9]", "", name)
            rank = (listing_rank, distance, symbol)
            if (
                issuer_key not in ranked_by_issuer
                or rank < ranked_by_issuer[issuer_key]
            ):
                ranked_by_issuer[issuer_key] = rank
        symbols = tuple(
            item[2] for item in sorted(ranked_by_issuer.values())[:max_candidates]
        )
        confidence = (
            "high" if len(symbols) >= 15 else "medium" if len(symbols) >= 5 else "low"
        )
        warnings = ()
        if confidence == "low":
            warnings = (
                f"Only {len(symbols)} provider-supported sector candidates were discovered",
            )
        return PeerDiscoveryResult(
            provider="yahoo-screener",
            target_ticker=target.symbol,
            candidate_tickers=symbols,
            methodology=(
                f"Yahoo {field.replace('_', ' ')} {filter_value!r} equity screener; "
                "candidates ordered deterministically "
                "by geography signal, market-cap proximity and ticker before economic scoring"
            ),
            confidence=confidence,
            warnings=warnings,
        )

    @staticmethod
    def _unavailable(symbol, warning):
        return PeerDiscoveryResult(
            provider="yahoo-screener",
            target_ticker=symbol,
            methodology="Provider-backed sector screening unavailable",
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


__all__ = [
    "PeerCandidateDiscoveryProvider",
    "YahooScreenerPeerDiscoveryProvider",
]
