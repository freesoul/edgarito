"""Layered provider retrieval for forward annual revenue estimates."""

from __future__ import annotations

import datetime
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

from edgarito.schemas.forecasting import ForwardGrowthEvidence
from edgarito.schemas.forward import (
    ForwardEstimateProviderDiagnostic,
    ForwardEstimateProviderStatus,
    ForwardRevenueEstimate,
    ForwardRevenueEstimateResult,
)
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.providers.yahoo.fundamentals import (
    YahooRevenueEstimateResponse,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization.alphavantage import AlphaVantageNormalizer
from edgarito.services.normalization.yahoo import YahooFinancialsNormalizer
from edgarito.services.providers.alphavantage import AlphaVantageClient
from edgarito.services.providers.yahoo import YahooFinanceClient


class ForwardRevenueEstimateService:
    """Resolve annual revenue consensus without coupling to historical data.

    Provider preference is intentionally local to this service:

    ``Alpha Vantage -> Yahoo/yfinance -> unavailable``.

    A failed estimate provider never raises out of :meth:`resolve`; its
    diagnostic is retained and the next provider is attempted.
    """

    def __init__(
        self,
        cache: FileSystemCache | str | Path | None = None,
        alphavantage_api_key: str | None = None,
        *,
        alpha_vantage_api_key: str | None = None,
        alpha_client: Any | None = None,
        alphavantage_client: Any | None = None,
        alpha_provider: Any | None = None,
        yahoo_client: Any | None = None,
        yahoo_provider: Any | None = None,
        alpha_normalizer: AlphaVantageNormalizer | None = None,
        yahoo_normalizer: YahooFinancialsNormalizer | None = None,
    ) -> None:
        self._cache = (
            cache
            if isinstance(cache, FileSystemCache)
            else FileSystemCache(cache or "cache")
        )
        self._alphavantage_api_key = alphavantage_api_key or alpha_vantage_api_key
        self._alpha_client = alpha_client or alphavantage_client or alpha_provider
        self._yahoo_client = yahoo_client or yahoo_provider
        self._alpha_normalizer = alpha_normalizer or AlphaVantageNormalizer()
        self._yahoo_normalizer = yahoo_normalizer or YahooFinancialsNormalizer()

    async def resolve(
        self,
        symbol: str | None = None,
        *,
        ticker: str | None = None,
        financials: NormalizedCompanyFinancials | None = None,
        forecast_years: tuple[int, ...] | list[int] = (),
        current_fiscal_year: int | None = None,
        base_revenue: Decimal | None = None,
        currency: str | None = None,
        provider_symbols: Mapping[Any, str] | None = None,
        as_of: datetime.date | datetime.datetime | None = None,
        use_cache: bool = True,
        make_cache: bool = True,
    ) -> ForwardRevenueEstimateResult:
        """Retrieve and normalize the first usable annual estimate set."""

        del as_of  # Estimate endpoints expose their own retrieval timestamp.
        normalized_symbol = (symbol or ticker or "").strip().upper()
        provider_symbols = provider_symbols or {}
        alpha_symbol = self._provider_symbol(
            provider_symbols, "alphavantage", normalized_symbol
        )
        yahoo_symbol = self._provider_symbol(
            provider_symbols, "yahoo", normalized_symbol
        )
        target_years = tuple(dict.fromkeys(int(year) for year in forecast_years))
        if current_fiscal_year is None and target_years:
            current_fiscal_year = target_years[0]
        fiscal_end_month = self._fiscal_end_month(financials)
        diagnostics: list[ForwardEstimateProviderDiagnostic] = []
        warnings: list[str] = []

        alpha_estimates: tuple[ForwardRevenueEstimate, ...] = ()
        if not alpha_symbol:
            diagnostics.append(
                self._diagnostic(
                    "alphavantage",
                    ForwardEstimateProviderStatus.UNAVAILABLE,
                    "ticker/symbol unavailable",
                    credentials_available=bool(self._alphavantage_api_key),
                )
            )
        elif self._alpha_client is None and not self._alphavantage_api_key:
            diagnostics.append(
                self._diagnostic(
                    "alphavantage",
                    ForwardEstimateProviderStatus.UNAVAILABLE,
                    "API key missing",
                    credentials_available=False,
                )
            )
        else:
            try:
                alpha_client, owned = self._alpha_provider()
                raw = await self._call_provider(
                    alpha_client,
                    ("get_earnings_estimates", "get_revenue_estimates"),
                    alpha_symbol,
                    use_cache=use_cache,
                    make_cache=make_cache,
                )
                alpha_estimates = self._normalize_alpha(
                    raw,
                    observed_at=datetime.datetime.now(datetime.timezone.utc),
                    fiscal_end_month=fiscal_end_month,
                )
                alpha_estimates, rejected_reason = self._usable_estimates(
                    alpha_estimates,
                    target_years=target_years,
                    currency=currency,
                    fiscal_end_month=fiscal_end_month,
                )
                if alpha_estimates:
                    diagnostics.append(
                        self._diagnostic(
                            "alphavantage",
                            ForwardEstimateProviderStatus.SUCCESS,
                            None,
                            estimate_count=len(alpha_estimates),
                            years=tuple(item.fiscal_year for item in alpha_estimates),
                            credentials_available=True,
                        )
                    )
                    if owned:
                        await self._close(alpha_client)
                    diagnostics.append(
                        self._diagnostic(
                            "yahoo",
                            ForwardEstimateProviderStatus.NOT_NEEDED,
                            "primary provider returned usable estimates",
                            attempted=False,
                        )
                    )
                    return ForwardRevenueEstimateResult(
                        estimates=alpha_estimates,
                        selected_provider="alphavantage",
                        diagnostics=tuple(diagnostics),
                        warnings=tuple(warnings),
                    )
                diagnostics.append(
                    self._diagnostic(
                        "alphavantage",
                        ForwardEstimateProviderStatus.UNAVAILABLE,
                        rejected_reason or "no usable annual revenue estimates",
                        credentials_available=True,
                    )
                )
                if owned:
                    await self._close(alpha_client)
            except Exception as exc:
                if "alpha_client" in locals() and locals().get("owned", False):
                    await self._close(alpha_client)
                diagnostics.append(
                    self._diagnostic(
                        "alphavantage",
                        ForwardEstimateProviderStatus.FAILED,
                        self._safe_reason(exc),
                        credentials_available=True,
                    )
                )
                if "rate" in str(exc).casefold() or "limit" in str(exc).casefold():
                    warnings.append("Alpha Vantage forward estimates were rate limited")

        yahoo_estimates: tuple[ForwardRevenueEstimate, ...] = ()
        if not yahoo_symbol:
            diagnostics.append(
                self._diagnostic(
                    "yahoo",
                    ForwardEstimateProviderStatus.UNAVAILABLE,
                    "ticker/symbol unavailable",
                )
            )
        else:
            try:
                yahoo_client, owned = self._yahoo_provider()
                raw = await self._call_provider(
                    yahoo_client,
                    ("get_revenue_estimates", "get_revenue_estimate"),
                    yahoo_symbol,
                    use_cache=use_cache,
                    make_cache=make_cache,
                )
                yahoo_estimates = self._normalize_yahoo(
                    raw,
                    current_fiscal_year=current_fiscal_year,
                    forecast_years=target_years,
                    fiscal_end_month=fiscal_end_month,
                )
                yahoo_estimates, rejected_reason = self._usable_estimates(
                    yahoo_estimates,
                    target_years=target_years,
                    currency=currency,
                    fiscal_end_month=fiscal_end_month,
                )
                if yahoo_estimates:
                    diagnostics.append(
                        self._diagnostic(
                            "yahoo",
                            ForwardEstimateProviderStatus.SUCCESS,
                            None,
                            estimate_count=len(yahoo_estimates),
                            years=tuple(item.fiscal_year for item in yahoo_estimates),
                        )
                    )
                    if owned:
                        await self._close(yahoo_client)
                    return ForwardRevenueEstimateResult(
                        estimates=yahoo_estimates,
                        selected_provider="yahoo",
                        diagnostics=tuple(diagnostics),
                        warnings=tuple(warnings),
                        fallback_reason="Alpha Vantage did not return usable estimates",
                    )
                diagnostics.append(
                    self._diagnostic(
                        "yahoo",
                        ForwardEstimateProviderStatus.UNAVAILABLE,
                        rejected_reason or "no usable annual revenue estimates",
                    )
                )
                if owned:
                    await self._close(yahoo_client)
            except Exception as exc:
                if "yahoo_client" in locals() and locals().get("owned", False):
                    await self._close(yahoo_client)
                diagnostics.append(
                    self._diagnostic(
                        "yahoo",
                        ForwardEstimateProviderStatus.FAILED,
                        self._safe_reason(exc),
                    )
                )

        fallback_reason = "; ".join(
            f"{item.provider}: {item.reason}"
            for item in diagnostics
            if item.reason
            and item.status
            in {
                ForwardEstimateProviderStatus.UNAVAILABLE,
                ForwardEstimateProviderStatus.FAILED,
            }
        )
        return ForwardRevenueEstimateResult(
            diagnostics=tuple(diagnostics),
            warnings=tuple(warnings),
            fallback_reason=fallback_reason or "no usable forward revenue estimates",
        )

    @staticmethod
    def to_growth_evidence(
        result: ForwardRevenueEstimateResult,
        *,
        forecast_years: tuple[int, ...] | list[int],
        base_revenue: Decimal,
        seed_revenues: Mapping[int, Decimal] | None = None,
        seed_growth_path: tuple[Decimal, ...] | list[Decimal] = (),
        lifecycle: str = "unknown",
        backlog: bool = False,
        capacity: bool = False,
        growth_visibility: Decimal = Decimal(0),
    ) -> ForwardGrowthEvidence:
        """Convert absolute annual estimates into correctly indexed growth.

        A gap before the first estimate is represented with the existing seed
        year's growth only when a later estimate can be mapped to it.  Once a
        consensus path starts, a gap ends the explicit prefix; the adaptive
        service then owns the transition toward terminal growth.
        """

        years = tuple(int(year) for year in forecast_years)
        seed_revenues = dict(seed_revenues or {})
        seed_growth = tuple(seed_growth_path)
        by_year = {item.fiscal_year: item for item in result.estimates}
        first_estimate_index = next(
            (index for index, year in enumerate(years) if year in by_year), None
        )
        values: list[Decimal] = []
        used_estimate_growth: list[Decimal] = []
        used_estimate_years: list[int] = []
        growth_by_year: list[tuple[int, Decimal]] = []
        previous_revenue = base_revenue
        consensus_started = False
        for index, year in enumerate(years):
            estimate = by_year.get(year)
            if estimate is not None and estimate.midpoint is not None:
                value = estimate.midpoint
                prior = previous_revenue
                if not consensus_started and index > 0:
                    prior = seed_revenues.get(year - 1, prior)
                if prior <= 0:
                    break
                growth = (value / prior - Decimal(1)) * Decimal(100)
                values.append(growth)
                growth_by_year.append((year, growth))
                used_estimate_growth.append(growth)
                used_estimate_years.append(year)
                previous_revenue = value
                consensus_started = True
                continue

            if consensus_started:
                break
            if first_estimate_index is None or index >= first_estimate_index:
                break
            seed_revenue = seed_revenues.get(year)
            if seed_revenue is None or seed_revenue <= 0 or previous_revenue <= 0:
                break
            growth = (seed_revenue / previous_revenue - Decimal(1)) * Decimal(100)
            values.append(growth)
            growth_by_year.append((year, growth))
            previous_revenue = seed_revenue
            if index < len(seed_growth) and seed_growth[index] != growth:
                # The model revenue is authoritative for fiscal alignment;
                # the discrepancy is retained in the path rather than hidden.
                pass

        confidence = ForwardRevenueEstimateService._path_confidence(
            result.estimates,
            used_estimate_years=tuple(used_estimate_years),
        )
        source = (
            f"analyst_consensus / {result.selected_provider}"
            if used_estimate_years and result.selected_provider
            else None
        )
        return ForwardGrowthEvidence(
            guidance=False,
            backlog=backlog,
            capacity=capacity,
            growth_visibility=growth_visibility,
            lifecycle=lifecycle,
            growth_path=tuple(values),
            growth_path_by_year=tuple(growth_by_year),
            confidence=confidence,
            source=source,
            forward_revenue_estimates=result.estimates,
            forward_estimate_provider=result.selected_provider,
            forward_estimate_diagnostics=result.diagnostics,
            forward_estimate_years=tuple(used_estimate_years),
            forward_estimate_growth_path=tuple(used_estimate_growth),
        )

    @staticmethod
    def estimates_to_growth_path(
        estimates: Iterable[ForwardRevenueEstimate],
        *,
        base_revenue: Decimal,
        forecast_years: tuple[int, ...] | list[int],
    ) -> tuple[Decimal, ...]:
        """Small pure helper for consumers that only need a compact prefix."""

        result = ForwardRevenueEstimateResult(
            estimates=tuple(estimates), selected_provider="unknown"
        )
        evidence = ForwardRevenueEstimateService.to_growth_evidence(
            result,
            forecast_years=tuple(forecast_years),
            base_revenue=base_revenue,
        )
        return evidence.growth_path

    def _alpha_provider(self) -> tuple[Any, bool]:
        if self._alpha_client is not None:
            return self._alpha_client, False
        return (
            AlphaVantageClient(self._cache, self._alphavantage_api_key or ""),
            True,
        )

    def _yahoo_provider(self) -> tuple[Any, bool]:
        if self._yahoo_client is not None:
            return self._yahoo_client, False
        return YahooFinanceClient(self._cache), True

    @staticmethod
    def _provider_symbol(
        provider_symbols: Mapping[Any, str], provider: str, fallback: str
    ) -> str:
        for key, value in provider_symbols.items():
            key_value = getattr(key, "value", key)
            if str(key_value).casefold() == provider:
                return str(value).strip().upper()
        return fallback

    @staticmethod
    async def _call_provider(
        client: Any,
        method_names: tuple[str, ...],
        symbol: str,
        *,
        use_cache: bool,
        make_cache: bool,
    ):
        method = next(
            (
                getattr(client, name, None)
                for name in method_names
                if hasattr(client, name)
            ),
            None,
        )
        if method is None:
            raise AttributeError(
                f"forward estimate provider has none of: {', '.join(method_names)}"
            )
        result = method(symbol, use_cache=use_cache, make_cache=make_cache)
        if hasattr(result, "__await__"):
            return await result
        return result

    def _normalize_alpha(
        self,
        raw: Any,
        *,
        observed_at: datetime.datetime,
        fiscal_end_month: int | None,
    ) -> tuple[ForwardRevenueEstimate, ...]:
        if isinstance(raw, (tuple, list)) and all(
            isinstance(item, ForwardRevenueEstimate) for item in raw
        ):
            return tuple(raw)
        return self._alpha_normalizer.normalize_earnings_estimates(
            raw,
            observed_at=observed_at,
            fiscal_end_month=fiscal_end_month,
        )

    def _normalize_yahoo(
        self,
        raw: Any,
        *,
        current_fiscal_year: int | None,
        forecast_years: tuple[int, ...],
        fiscal_end_month: int | None,
    ) -> tuple[ForwardRevenueEstimate, ...]:
        if isinstance(raw, (tuple, list)) and all(
            isinstance(item, ForwardRevenueEstimate) for item in raw
        ):
            return tuple(raw)
        if not isinstance(raw, YahooRevenueEstimateResponse):
            if hasattr(raw, "iterrows"):
                rows = YahooFinanceClient._revenue_estimate_rows(raw)
                raw = YahooRevenueEstimateResponse(
                    symbol="",
                    rows=tuple(rows),
                    retrieved_at=datetime.datetime.now(datetime.timezone.utc),
                )
            elif isinstance(raw, dict) and "rows" not in raw:
                rows = YahooFinanceClient._revenue_estimate_rows(raw)
                raw = YahooRevenueEstimateResponse(
                    symbol="",
                    rows=tuple(rows),
                    retrieved_at=datetime.datetime.now(datetime.timezone.utc),
                )
            else:
                raw = YahooRevenueEstimateResponse.model_validate(raw)
        return self._yahoo_normalizer.normalize_revenue_estimates(
            raw,
            current_fiscal_year=current_fiscal_year,
            forecast_years=forecast_years,
            fiscal_end_month=fiscal_end_month,
        )

    @staticmethod
    def _usable_estimates(
        estimates: tuple[ForwardRevenueEstimate, ...],
        *,
        target_years: tuple[int, ...],
        currency: str | None,
        fiscal_end_month: int | None,
    ) -> tuple[tuple[ForwardRevenueEstimate, ...], str | None]:
        selected: dict[int, ForwardRevenueEstimate] = {}
        rejected: list[str] = []
        normalized_currency = currency.strip().upper() if currency else None
        for item in estimates:
            value = item.midpoint
            if target_years and item.fiscal_year not in target_years:
                continue
            if value is None or value <= 0:
                continue
            if (
                item.period_end is not None
                and fiscal_end_month is not None
                and item.mapping_method == "explicit_fiscal_date_mismatched_fiscal_end"
            ):
                rejected.append(
                    f"FY{item.fiscal_year} fiscal date does not match issuer fiscal year end"
                )
                continue
            if (
                normalized_currency
                and item.currency
                and item.currency.casefold() != normalized_currency.casefold()
            ):
                rejected.append(
                    f"FY{item.fiscal_year} currency {item.currency} does not match "
                    f"forecast currency {normalized_currency}"
                )
                continue
            existing = selected.get(item.fiscal_year)
            if existing is None or ForwardRevenueEstimateService._estimate_key(item) > (
                ForwardRevenueEstimateService._estimate_key(existing)
            ):
                selected[item.fiscal_year] = item.model_copy(
                    update={
                        "confidence": ForwardRevenueEstimateService._confidence(item)
                    }
                )
        ordered = tuple(selected[year] for year in sorted(selected))
        return ordered, "; ".join(rejected) if rejected and not ordered else None

    @staticmethod
    def _estimate_key(item: ForwardRevenueEstimate) -> tuple[int, Decimal]:
        return (item.analyst_count or 0, item.midpoint or Decimal(0))

    @staticmethod
    def _confidence(item: ForwardRevenueEstimate) -> str:
        analysts = item.analyst_count
        confidence = (
            "high"
            if item.source.casefold() == "alphavantage"
            and analysts is not None
            and analysts >= 10
            else "medium"
            if analysts is None or analysts >= 3
            else "low"
        )
        value = item.midpoint
        if value and item.low is not None and item.high is not None:
            dispersion = (item.high - item.low) / value
            if dispersion > Decimal("0.50"):
                return "low"
            if dispersion > Decimal("0.25") and confidence == "high":
                return "medium"
        if item.source.casefold() == "yahoo" and confidence == "high":
            return "medium"
        return confidence

    @staticmethod
    def _path_confidence(
        estimates: tuple[ForwardRevenueEstimate, ...],
        *,
        used_estimate_years: tuple[int, ...],
    ) -> str | None:
        selected = [
            item for item in estimates if item.fiscal_year in set(used_estimate_years)
        ]
        if not selected:
            return None
        ranks = {"low": 0, "medium": 1, "high": 2}
        return min(
            (item.confidence or "medium" for item in selected),
            key=lambda value: ranks.get(value, 1),
        )

    @staticmethod
    def _fiscal_end_month(
        financials: NormalizedCompanyFinancials | None,
    ) -> int | None:
        if financials is None:
            return None
        dates = [
            item.period_end
            for item in financials.observations
            if item.granularity.value == "annual"
            and item.fiscal_period.value == "FY"
            and item.concept == FinancialConcept.REVENUE
        ]
        if not dates:
            dates = [
                item.period_end
                for item in financials.observations
                if item.granularity.value == "annual"
                and item.fiscal_period.value == "FY"
            ]
        return max(dates).month if dates else None

    @staticmethod
    def _diagnostic(
        provider: str,
        status: ForwardEstimateProviderStatus,
        reason: str | None,
        *,
        estimate_count: int = 0,
        years: tuple[int, ...] = (),
        credentials_available: bool | None = None,
        attempted: bool = True,
    ) -> ForwardEstimateProviderDiagnostic:
        return ForwardEstimateProviderDiagnostic(
            provider=provider,
            status=status,
            reason=reason,
            estimate_count=estimate_count,
            years=years,
            credentials_available=credentials_available,
            attempted=attempted,
        )

    @staticmethod
    def _safe_reason(exc: Exception) -> str:
        reason = str(exc).strip() or exc.__class__.__name__
        lowered = reason.casefold()
        if "rate" in lowered and "limit" in lowered:
            return "rate limited"
        if "api key" in lowered or "apikey" in lowered or "credential" in lowered:
            return "API key rejected"
        if (
            "unsupported ticker" in lowered
            or "symbol" in lowered
            and "not found" in lowered
        ):
            return "unsupported ticker"
        reason = re.sub(r"(?i)(api[- ]?key\s*[=:]?\s*)\S+", r"\1[REDACTED]", reason)
        return reason[:240]

    @staticmethod
    async def _close(client: Any) -> None:
        close = getattr(client, "__aexit__", None)
        if close is not None:
            try:
                result = close(None, None, None)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                # Provider cleanup must never suppress the original result or
                # prevent the next fallback provider from being attempted.
                return


__all__ = ["ForwardRevenueEstimateService"]
