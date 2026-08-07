import argparse
import asyncio
import datetime
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional

from edgarito.cli.presentation.console import ComparableMultiplesConsolePresenter
from edgarito.config.valuation import ValuationProfileLoader
from edgarito.schemas.market import SecurityMarketData
from edgarito.schemas.normalization.financials import NormalizedCompanyFinancials
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization.classification import (
    CompanyClassificationNormalizer,
)
from edgarito.services.normalization.yahoo import YahooFinancialsNormalizer
from edgarito.services.normalization.yahoo_market import YahooMarketNormalizer
from edgarito.services.providers.ecb import EcbClient
from edgarito.services.providers.yahoo import YahooFinanceClient
from edgarito.services.valuation import (
    ComparableMultiplesReport,
    ComparableMultiplesService,
    EcbMarketDataCurrencyConverter,
    LtmMultiplesService,
    MarketAwarePeerDiscoveryProvider,
    MassiveRelatedCompaniesPeerDiscoveryProvider,
    PeerDiscoveryResult,
    PeerSelectionParameters,
    PeerUniverseSelector,
    ValuationProfileBuilder,
    ValuationProfileOverrides,
    YahooScreenerPeerDiscoveryProvider,
)
from edgarito.settings import MASSIVE_API_KEY


@dataclass(frozen=True)
class ComparableReportBundle:
    report: ComparableMultiplesReport
    target_financials: NormalizedCompanyFinancials
    target_market: SecurityMarketData
    peer_sources: dict[str, tuple[NormalizedCompanyFinancials, SecurityMarketData]]

    @property
    def reliable_peer_roics(self) -> tuple[Decimal, ...]:
        if self.report.universe.discovery_confidence == "low":
            return ()
        return tuple(
            peer.fundamentals.return_on_invested_capital
            for peer in self.report.peers
            if peer.fundamentals.return_on_invested_capital is not None
        )


def _resolve_comparable_peer_symbols(args, valuation_profile, target_symbol):
    cli_peers = getattr(args, "peer", None) or ()
    configured_peers = valuation_profile.comparables.peers
    source = None if cli_peers else "valuation-profile" if configured_peers else None
    values = cli_peers or configured_peers
    target = target_symbol.strip().upper()
    symbols = []
    seen = {target}
    for value in values:
        symbol = value.strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols, source


async def _run_comparables(args: argparse.Namespace) -> int:
    valuation_profile, _, _ = ValuationProfileLoader.load_for_ticker(
        args.ticker, args.profile
    )
    target_symbol = args.ticker.strip().upper()
    peer_symbols, peer_source = _resolve_comparable_peer_symbols(
        args,
        valuation_profile,
        target_symbol,
    )
    bundle = await _build_comparable_report(
        args,
        valuation_profile,
        target_symbol,
        peer_symbols,
        peer_source=peer_source,
        as_of=args.as_of,
    )
    print(ComparableMultiplesConsolePresenter().render(bundle.report))
    return 0


async def _build_comparable_report(
    args,
    valuation_profile,
    target_symbol,
    peer_symbols,
    *,
    peer_source=None,
    as_of=None,
):
    configuration = valuation_profile.comparables
    selection_configuration = valuation_profile.model_selection
    parameters = PeerSelectionParameters(
        max_peers=(
            getattr(args, "max_peers", None)
            if getattr(args, "max_peers", None) is not None
            else configuration.max_peers
        ),
        preferred_minimum=(
            getattr(args, "preferred_minimum", None)
            if getattr(args, "preferred_minimum", None) is not None
            else configuration.preferred_minimum
        ),
        minimum_score=(
            getattr(args, "minimum_score", None)
            if getattr(args, "minimum_score", None) is not None
            else configuration.minimum_score
        ),
        require_same_sector=(
            getattr(args, "require_same_sector", None)
            if getattr(args, "require_same_sector", None) is not None
            else configuration.require_same_sector
        ),
    )
    discovery = (
        PeerDiscoveryResult(
            provider="valuation-profile",
            target_ticker=target_symbol,
            candidate_tickers=tuple(peer_symbols),
            methodology=(
                f"Saved comparable peers from valuation profile {valuation_profile.name!r}"
            ),
            confidence="high",
        )
        if peer_symbols and peer_source == "valuation-profile"
        else None
    )
    cache = FileSystemCache(Path(args.cache_dir))
    if not peer_symbols:
        async with YahooFinanceClient(cache) as client:
            try:
                target_source = await client.get_company_financials(
                    target_symbol,
                    use_cache=not args.refresh,
                    make_cache=True,
                )
                if target_source.market_capitalization is None:
                    target_source = await client.get_company_financials(
                        target_symbol,
                        use_cache=False,
                        make_cache=True,
                    )
                yahoo_discovery = YahooScreenerPeerDiscoveryProvider(
                    cache,
                    use_cache=not args.refresh,
                    make_cache=True,
                )
                massive_discovery = (
                    MassiveRelatedCompaniesPeerDiscoveryProvider(
                        cache,
                        MASSIVE_API_KEY,
                        use_cache=not args.refresh,
                        make_cache=True,
                    )
                    if MASSIVE_API_KEY
                    else None
                )
                discovery = await MarketAwarePeerDiscoveryProvider(
                    yahoo_discovery,
                    massive_discovery,
                    minimum_candidates=parameters.preferred_minimum,
                ).discover(
                    target_source,
                    max_candidates=max(12, parameters.max_peers * 3),
                )
                peer_symbols = list(discovery.candidate_tickers)
            except (RuntimeError, ValueError) as exc:
                discovery = PeerDiscoveryResult(
                    provider="yahoo-screener",
                    target_ticker=target_symbol,
                    methodology="Provider-backed discovery failed",
                    confidence="low",
                    warnings=(f"Automatic peer discovery failed: {exc}",),
                )
    symbols = [target_symbol, *peer_symbols]
    async with YahooFinanceClient(cache) as client:
        results = await asyncio.gather(
            *(
                _retrieve_yahoo_comparable_source(
                    client,
                    symbol,
                    as_of,
                    use_cache=not args.refresh,
                    history_period="5y",
                )
                for symbol in symbols
            ),
            return_exceptions=True,
        )

    if isinstance(results[0], BaseException):
        raise RuntimeError(f"Target retrieval failed for {target_symbol}: {results[0]}")

    profile_builder = ValuationProfileBuilder()
    classification_normalizer = CompanyClassificationNormalizer()
    financials_normalizer = YahooFinancialsNormalizer()
    market_normalizer = YahooMarketNormalizer()
    multiples_service = LtmMultiplesService()
    bundles = {}
    normalized_financials = {}
    normalized_markets = {}
    retrieval_warnings = []
    for symbol, result in zip(symbols, results, strict=True):
        if isinstance(result, BaseException):
            retrieval_warnings.append(f"{symbol} retrieval failed: {result}")
            continue
        source, history = result
        financials = financials_normalizer.normalize(source)
        classification = classification_normalizer.normalize_yahoo(source)
        profile = profile_builder.build(
            financials,
            classification,
            (
                ValuationProfileOverrides(
                    sector=selection_configuration.sector,
                    industry=selection_configuration.industry,
                )
                if symbol == target_symbol
                else None
            ),
        )
        market_data = market_normalizer.normalize(history)
        if market_data.currency != source.currency:
            try:
                async with EcbClient(cache) as ecb:
                    market_data = await EcbMarketDataCurrencyConverter(ecb).convert(
                        market_data,
                        source.currency,
                        use_cache=not args.refresh,
                        make_cache=True,
                    )
            except (RuntimeError, ValueError) as exc:
                retrieval_warnings.append(
                    f"{symbol} market currency could not be aligned to "
                    f"{source.currency}: {exc}"
                )
        try:
            multiples = multiples_service.compute(financials, market_data, as_of)
        except ValueError as exc:
            multiples = None
            retrieval_warnings.append(f"{symbol} multiples unavailable: {exc}")
        bundles[symbol] = (profile, multiples)
        normalized_financials[symbol] = financials
        normalized_markets[symbol] = market_data

    target_profile, target_multiples = bundles[target_symbol]
    if target_multiples is None:
        raise ValueError(f"LTM multiples could not be computed for {target_symbol}")
    candidate_profiles = [
        bundles[symbol][0] for symbol in peer_symbols if symbol in bundles
    ]
    universe = PeerUniverseSelector().select(
        target_profile,
        candidate_profiles,
        parameters,
        target_multiples=target_multiples,
        candidate_multiples={
            symbol: bundles[symbol][1]
            for symbol in peer_symbols
            if symbol in bundles and bundles[symbol][1] is not None
        },
        discovery=discovery,
    )
    peer_multiples = [
        bundles[symbol][1]
        for symbol in peer_symbols
        if symbol in bundles and bundles[symbol][1] is not None
    ]
    report = ComparableMultiplesService().build(
        universe,
        target_multiples,
        peer_multiples,
    )
    if retrieval_warnings:
        report = report.model_copy(
            update={"warnings": [*report.warnings, *retrieval_warnings]}
        )
    return ComparableReportBundle(
        report=report,
        target_financials=normalized_financials[target_symbol],
        target_market=normalized_markets[target_symbol],
        peer_sources={
            symbol: (normalized_financials[symbol], normalized_markets[symbol])
            for symbol in universe.selected_tickers
            if symbol in normalized_financials and symbol in normalized_markets
        },
    )


async def _retrieve_yahoo_comparable_source(
    client: YahooFinanceClient,
    symbol: str,
    as_of: Optional[datetime.date],
    *,
    use_cache: bool,
    history_period: str = "1mo",
):
    history_arguments = {"period": history_period}
    if as_of is not None and history_period == "1mo":
        history_arguments = {
            "start": as_of - datetime.timedelta(days=14),
            "end": as_of + datetime.timedelta(days=1),
        }
    return await asyncio.gather(
        client.get_company_financials(symbol, use_cache=use_cache, make_cache=True),
        client.get_price_history(
            symbol,
            **history_arguments,
            use_cache=use_cache,
            make_cache=True,
        ),
    )


__all__ = [
    "_build_comparable_report",
    "_resolve_comparable_peer_symbols",
    "_run_comparables",
]
