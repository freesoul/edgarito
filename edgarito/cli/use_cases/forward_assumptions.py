"""Forward evidence, guidance overlays, and automatic assumption use cases."""

from __future__ import annotations

import argparse
import asyncio
import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

from edgarito.cli.use_cases.context import call_with_context, dependency
from edgarito.cli.use_cases.financial_retrieval import parse_provider_symbols
from edgarito.cli.use_cases.forecast import market_for_args
from edgarito.enums.market import Market
from edgarito.schemas.forecasting import (
    FcffForecastParameters,
    ForecastAssumptionSource,
    ForwardGrowthEvidence,
)
from edgarito.schemas.forward import (
    ForwardEstimateProviderDiagnostic,
    ForwardRevenueEstimateResult,
)
from edgarito.schemas.guidance.management import GuidanceOverlayResult
from edgarito.schemas.market import ReferenceSeriesKind, ReferenceValueUnit
from edgarito.schemas.normalization.financials import NormalizedCompanyFinancials
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.forecasting.forward_estimates import (
    ForwardRevenueEstimateService,
)
from edgarito.services.guidance.extraction import ManagementGuidanceExtractor
from edgarito.services.guidance.overlay import GuidanceForecastOverlay
from edgarito.services.guidance.service import ManagementGuidanceService
from edgarito.services.normalization.classification import (
    CompanyClassificationNormalizer,
)
from edgarito.services.normalization.yahoo_market import YahooMarketNormalizer
from edgarito.services.openai import OpenAIClient
from edgarito.services.providers.damodaran import DamodaranClient
from edgarito.services.providers.ecb import EcbClient
from edgarito.services.providers.edgar import EdgarClient
from edgarito.services.providers.fred import FredClient
from edgarito.services.providers.treasury import TreasuryClient
from edgarito.services.providers.yahoo import YahooFinanceClient
from edgarito.services.valuation import (
    DiscountRateService,
    EcbMarketDataCurrencyConverter,
)
from edgarito.settings import (
    ALPHAVANTAGE_API_KEY,
    FRED_API_KEY,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_REASONING_EFFORT,
)


async def retrieve_forward_estimates(
    args: argparse.Namespace,
    financials: NormalizedCompanyFinancials,
    forecast,
    *,
    use_cache: bool,
    make_cache: bool,
    context=None,
) -> ForwardRevenueEstimateResult:
    forecast_years = tuple(item.fiscal_year for item in forecast.observations)
    cache_type = dependency(context, "FileSystemCache", FileSystemCache)
    service_type = dependency(
        context,
        "ForwardRevenueEstimateService",
        ForwardRevenueEstimateService,
    )
    symbols_parser = dependency(
        context,
        "_parse_provider_symbols",
        parse_provider_symbols,
    )
    try:
        return await service_type(
            cache=cache_type(Path(args.cache_dir)),
            alphavantage_api_key=dependency(
                context, "ALPHAVANTAGE_API_KEY", ALPHAVANTAGE_API_KEY
            ),
        ).resolve(
            financials.ticker or args.ticker,
            financials=financials,
            forecast_years=forecast_years,
            current_fiscal_year=(
                forecast.current_fiscal_year
                or (forecast_years[0] if forecast_years else None)
            ),
            base_revenue=forecast.base_revenue,
            currency=forecast.unit,
            provider_symbols=call_with_context(
                symbols_parser,
                args.provider_symbol,
                context=context,
            ),
            as_of=datetime.date.today(),
            use_cache=use_cache,
            make_cache=make_cache,
        )
    except Exception as exc:
        return ForwardRevenueEstimateResult(
            diagnostics=(
                ForwardEstimateProviderDiagnostic(
                    provider="alphavantage",
                    status="failed",
                    reason=f"resolver failure: {exc}",
                ),
                ForwardEstimateProviderDiagnostic(
                    provider="yahoo",
                    status="failed",
                    reason="not attempted because the resolver failed before provider fallback",
                ),
            ),
            warnings=(f"Forward estimate retrieval failed: {exc}",),
            fallback_reason=str(exc),
        )


def merge_forward_growth_evidence(
    management: ForwardGrowthEvidence,
    consensus: ForwardGrowthEvidence,
    *,
    management_has_revenue_guidance: bool,
    forecast_years: tuple[int, ...] = (),
) -> ForwardGrowthEvidence:
    del management_has_revenue_guidance
    consensus_metadata = {
        "forward_revenue_estimates": consensus.forward_revenue_estimates,
        "forward_estimate_provider": consensus.forward_estimate_provider,
        "forward_estimate_years": consensus.forward_estimate_years,
        "forward_estimate_growth_path": consensus.forward_estimate_growth_path,
        "forward_estimate_diagnostics": consensus.forward_estimate_diagnostics,
    }
    management_is_quantitative = bool(
        management.growth_path or management.guidance_growth_path
    )
    if management_is_quantitative:
        management_by_year = dict(management.growth_path_by_year)
        consensus_by_year = dict(consensus.growth_path_by_year)
        if management_by_year and consensus_by_year:
            years = forecast_years or tuple(
                sorted(set(management_by_year) | set(consensus_by_year))
            )
            merged_by_year = tuple(
                (year, management_by_year.get(year, consensus_by_year.get(year)))
                for year in years
                if management_by_year.get(year, consensus_by_year.get(year)) is not None
            )
            return management.model_copy(
                update={
                    **consensus_metadata,
                    "growth_path": tuple(value for _year, value in merged_by_year),
                    "growth_path_by_year": merged_by_year,
                }
            )
        return management.model_copy(update=consensus_metadata)
    if consensus.growth_path:
        return consensus.model_copy(
            update={
                "backlog": management.backlog,
                "capacity": management.capacity,
                "growth_visibility": max(
                    management.growth_visibility, consensus.growth_visibility
                ),
                "lifecycle": management.lifecycle,
            }
        )
    return management.model_copy(update=consensus_metadata)


def materialize_forward_revenue_anchors(
    parameters: FcffForecastParameters,
    estimates,
    forecast,
    warnings: list[str],
) -> FcffForecastParameters:
    if parameters.revenue_growth is not None:
        return parameters
    anchors = dict(parameters.revenue_anchors)
    sources = dict(parameters.revenue_anchor_sources)
    forecast_years = {item.fiscal_year for item in forecast.observations}
    ytd_anchor = getattr(forecast, "ytd_anchor", None)
    first_fiscal_year = (
        forecast.observations[0].fiscal_year if forecast.observations else None
    )
    for estimate in estimates:
        value = estimate.midpoint
        year = estimate.fiscal_year
        if value is None or year not in forecast_years or year in anchors:
            continue
        if (
            ytd_anchor is not None
            and year == first_fiscal_year
            and value < ytd_anchor.actual_revenue
        ):
            warnings.append(
                f"Forward consensus FY{year} revenue anchor ({value:,.0f}) is "
                f"below reported YTD revenue ({ytd_anchor.actual_revenue:,.0f}); "
                "the estimate was rejected"
            )
            continue
        anchors[year] = value
        sources[year] = ForecastAssumptionSource.FORWARD_EVIDENCE
    if anchors == parameters.revenue_anchors:
        return parameters
    return parameters.model_copy(
        update={
            "revenue_anchors": anchors,
            "revenue_anchor_sources": sources,
        }
    )


def financial_snapshot_warnings(
    financials: NormalizedCompanyFinancials,
    args: argparse.Namespace,
) -> list[str]:
    if financials.provider.casefold() != "yahoo":
        return []
    max_age_hours = getattr(args, "financial_snapshot_max_age_hours", 24)
    if max_age_hours <= 0:
        raise ValueError("--financial-snapshot-max-age-hours must be positive")
    retrieved_at = financials.retrieved_at
    if retrieved_at is None:
        return [
            "Yahoo financial snapshot retrieval time is unavailable; cache freshness "
            "cannot be established. Use --refresh for a current snapshot"
        ]
    age = datetime.datetime.now(datetime.timezone.utc) - retrieved_at.astimezone(
        datetime.timezone.utc
    )
    if age < datetime.timedelta(0):
        return [
            "Yahoo financial snapshot retrieval time is in the future; provenance "
            "should be checked"
        ]
    threshold = datetime.timedelta(hours=max_age_hours)
    if age <= threshold:
        return []
    age_hours = int(age.total_seconds() // 3600)
    return [
        f"Yahoo financial snapshot is stale ({age_hours} hours old; retrieved "
        f"{retrieved_at.astimezone(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC). "
        "Use --refresh for a current snapshot"
    ]


async def management_guidance_overlay(
    args: argparse.Namespace,
    financials: NormalizedCompanyFinancials,
    parameters: FcffForecastParameters,
    baseline,
    valuation_date: datetime.date,
    *,
    market: Market | str | None = None,
    context=None,
) -> tuple[FcffForecastParameters, GuidanceOverlayResult]:
    market = (
        call_with_context(
            dependency(context, "_market_for_args", market_for_args),
            args,
            context=context,
        )
        if market is None
        else Market(market)
    )
    if market != Market.US:
        return parameters, GuidanceOverlayResult(
            warnings=(
                f"SEC/EDGAR management guidance skipped for the {market.value} market",
            )
        )
    if not args.user_agent:
        return parameters, GuidanceOverlayResult(
            warnings=(
                "Management guidance skipped: SEC retrieval requires --user-agent",
            )
        )
    cache_type = dependency(context, "FileSystemCache", FileSystemCache)
    cache = cache_type(args.cache_dir)
    openai_type = dependency(context, "OpenAIClient", OpenAIClient)
    edgar_type = dependency(context, "EdgarClient", EdgarClient)
    guidance_service_type = dependency(
        context, "ManagementGuidanceService", ManagementGuidanceService
    )
    extractor_type = dependency(
        context, "ManagementGuidanceExtractor", ManagementGuidanceExtractor
    )
    overlay_type = dependency(context, "GuidanceForecastOverlay", GuidanceForecastOverlay)
    openai_client = openai_type(
        api_key=dependency(context, "OPENAI_API_KEY", OPENAI_API_KEY),
        model=dependency(context, "OPENAI_MODEL", OPENAI_MODEL),
        reasoning_effort=dependency(
            context, "OPENAI_REASONING_EFFORT", OPENAI_REASONING_EFFORT
        ),
    )
    try:
        async with edgar_type(cache, args.user_agent) as edgar:
            discovery = await guidance_service_type(
                edgar,
                extractor_type(openai_client, cache),
            ).retrieve(
                ticker=args.ticker or financials.ticker,
                cik=args.cik,
                as_of=valuation_date,
                refresh_sec=args.refresh,
            )
        overlaid, overlay = overlay_type().apply(
            discovery.records,
            baseline=baseline,
            parameters=parameters,
        )
        validation_rejections = tuple(
            f"Extraction rejected: {item.reason}" for item in discovery.rejected
        )
        return overlaid, overlay.model_copy(
            update={
                "rejected_reasons": tuple(
                    dict.fromkeys([*overlay.rejected_reasons, *validation_rejections])
                ),
                "warnings": tuple(
                    dict.fromkeys([*overlay.warnings, *discovery.warnings])
                ),
                "cache_hits": discovery.cache_hits,
                "cache_misses": discovery.cache_misses,
                "filings_inspected": discovery.filings_inspected,
                "documents_inspected": discovery.documents_inspected,
                "extracted_guidance_records": discovery.extracted_guidance_records,
                "rejected_records": (
                    discovery.rejected_records + len(overlay.rejected_reasons)
                ),
                "document_audits": getattr(discovery, "document_audits", ()),
            }
        )
    finally:
        await openai_client.close()


def forward_growth_evidence(
    lifecycle,
    economic_traits,
    guidance_overlay,
    forecast_years: tuple[int, ...] = (),
) -> ForwardGrowthEvidence:
    records = ()
    if guidance_overlay is not None:
        records = (*guidance_overlay.applications, *guidance_overlay.evidence_only)
    evidence_records = tuple(getattr(item, "guidance", item) for item in records)
    traits = {getattr(item, "value", str(item)) for item in economic_traits}
    backlog = "backlog_driven" in traits
    for item in guidance_overlay.applications if guidance_overlay is not None else ():
        metric = getattr(item.guidance.metric, "value", "")
        backlog = backlog or metric in {"backlog", "bookings"}
    guidance = any(
        getattr(getattr(item, "metric", None), "value", "")
        in {"revenue", "revenue_growth"}
        for item in evidence_records
    )
    text = " ".join(
        getattr(item, "supporting_text", "").casefold() for item in evidence_records
    )
    capacity = any(term in text for term in ("capacity", "cleanroom", "fab", "shipment"))
    visible_years = {
        item.fiscal_year
        for item in evidence_records
        if getattr(item, "fiscal_year", None) is not None
        and getattr(getattr(item, "period_type", None), "value", "")
        in {"multi_year_target", "long_term_target"}
    }
    growth_visibility = min(Decimal("1"), Decimal(len(visible_years)) / Decimal("3"))
    growth_applications = {
        application.fiscal_year: application.value
        for application in (
            guidance_overlay.applications if guidance_overlay is not None else ()
        )
        if application.driver == "revenue_growth"
        and getattr(getattr(application.guidance, "metric", None), "value", "")
        == "revenue_growth"
        and getattr(getattr(application.guidance, "period_type", None), "value", "")
        == "fiscal_year"
        and application.guidance.fiscal_year == application.fiscal_year
        and application.value is not None
    }
    forward_growth_records = []
    for index, fiscal_year in enumerate(forecast_years):
        value = growth_applications.get(fiscal_year)
        if value is None:
            if any(year in growth_applications for year in forecast_years[index + 1 :]):
                forward_growth_records = []
            break
        forward_growth_records.append((fiscal_year, value))
    return ForwardGrowthEvidence(
        backlog=backlog,
        guidance=guidance,
        capacity=capacity,
        growth_visibility=growth_visibility,
        lifecycle=getattr(lifecycle, "value", str(lifecycle)),
        growth_path=tuple(value for _year, value in forward_growth_records),
        growth_path_by_year=tuple(forward_growth_records),
        guidance_growth_path=tuple(
            growth_applications[year] for year in sorted(growth_applications)
        ),
        guidance_growth_path_by_year=tuple(sorted(growth_applications.items())),
        confidence="high" if forward_growth_records else None,
    )


async def retrieve_automatic_assumption_inputs(
    args: argparse.Namespace,
    financials: NormalizedCompanyFinancials,
    currency: str,
    *,
    needs_wacc: bool,
    needs_terminal: bool,
    sector_override=None,
    industry_override: Optional[str] = None,
    context=None,
) -> dict:
    inputs = {
        "classification": None,
        "market_data": None,
        "risk_free_series": None,
        "inflation_series": None,
        "country_snapshot": None,
        "industry_snapshot": None,
        "company_beta": None,
    }
    if not (needs_wacc or needs_terminal):
        return inputs

    cache_type = dependency(context, "FileSystemCache", FileSystemCache)
    cache = cache_type(Path(args.cache_dir))
    use_cache = not args.refresh
    symbol = financials.ticker or args.ticker
    if needs_wacc:
        if not symbol:
            raise ValueError(
                "Automatic WACC requires a ticker to retrieve Yahoo classification "
                "and market capitalization; provide --ticker or explicit WACC inputs"
            )
        try:
            yahoo_type = dependency(context, "YahooFinanceClient", YahooFinanceClient)
            normalizer_type = dependency(
                context,
                "CompanyClassificationNormalizer",
                CompanyClassificationNormalizer,
            )
            market_normalizer_type = dependency(
                context, "YahooMarketNormalizer", YahooMarketNormalizer
            )
            async with yahoo_type(cache) as yahoo:
                source, history = await asyncio.gather(
                    yahoo.get_company_financials(
                        symbol, use_cache=use_cache, make_cache=True
                    ),
                    yahoo.get_price_history(
                        symbol,
                        period="1mo",
                        use_cache=use_cache,
                        make_cache=True,
                    ),
                )
                classification = normalizer_type().normalize_yahoo(
                    source
                )
                if (
                    classification.industry is None
                    or classification.country is None
                    or source.beta is None
                ):
                    source = await yahoo.get_company_financials(
                        symbol, use_cache=False, make_cache=True
                    )
                    classification = normalizer_type().normalize_yahoo(
                        source
                    )
                classification = call_with_context(
                    dependency(
                        context,
                        "_apply_classification_overrides",
                        apply_classification_overrides,
                    ),
                    classification,
                    sector=sector_override,
                    industry=industry_override,
                    context=context,
                )
            inputs["classification"] = classification
            inputs["company_beta"] = source.beta
            market_data = market_normalizer_type().normalize(history)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "Automatic WACC could not retrieve Yahoo classification/price data; "
                "set WACC or the missing CAPM/capital-weight inputs in the profile. "
                f"Cause: {exc}"
            ) from exc
        if market_data.currency != currency.strip().upper():
            try:
                ecb_type = dependency(context, "EcbClient", EcbClient)
                converter_type = dependency(
                    context,
                    "EcbMarketDataCurrencyConverter",
                    EcbMarketDataCurrencyConverter,
                )
                async with ecb_type(cache) as ecb:
                    market_data = await converter_type(ecb).convert(
                        market_data,
                        currency,
                        use_cache=use_cache,
                        make_cache=True,
                    )
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"Automatic WACC could not align the Yahoo quote currency "
                    f"({market_data.currency}) with the financial-statement currency "
                    f"({currency}); provide an explicit WACC or market-value equity. "
                    f"Cause: {exc}"
                ) from exc
        inputs["market_data"] = market_data
        try:
            damodaran_type = dependency(context, "DamodaranClient", DamodaranClient)
            async with damodaran_type(cache) as damodaran:
                country_snapshot, industry_snapshot = await asyncio.gather(
                    damodaran.get_country_risk_premiums(
                        use_cache=use_cache, make_cache=True
                    ),
                    damodaran.get_industry_betas(use_cache=use_cache, make_cache=True),
                )
            inputs["country_snapshot"] = country_snapshot
            inputs["industry_snapshot"] = industry_snapshot
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                "Automatic WACC could not retrieve the versioned Damodaran country "
                "and industry references; set beta, ERP, country premium, and tax "
                f"inputs in the profile. Cause: {exc}"
            ) from exc

    normalized_currency = currency.strip().upper()
    try:
        if normalized_currency == "EUR":
            start = datetime.date.today() - datetime.timedelta(days=365 * 6)
            ecb_type = dependency(context, "EcbClient", EcbClient)
            async with ecb_type(cache) as ecb:
                risk_free_task = ecb.get_series(
                    "YC",
                    "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
                    kind=ReferenceSeriesKind.GOVERNMENT_YIELD,
                    unit=ReferenceValueUnit.PERCENTAGE_POINTS,
                    start_period=datetime.date.today() - datetime.timedelta(days=45),
                    end_period=datetime.date.today(),
                    use_cache=use_cache,
                    make_cache=True,
                )
                if needs_terminal:
                    risk_free, inflation = await asyncio.gather(
                        risk_free_task,
                        ecb.get_series(
                            "HICP",
                            "M.U2.N.000000.4D0.ANR",
                            kind=ReferenceSeriesKind.INFLATION_RATE,
                            unit=ReferenceValueUnit.PERCENT_CHANGE,
                            start_period=start,
                            end_period=datetime.date.today(),
                            use_cache=use_cache,
                            make_cache=True,
                        ),
                    )
                    inputs["inflation_series"] = inflation
                else:
                    risk_free = await risk_free_task
            inputs["risk_free_series"] = risk_free
        elif normalized_currency == "DKK":
            today = datetime.date.today()
            inflation_start = today - datetime.timedelta(days=365 * 6)
            ecb_type = dependency(context, "EcbClient", EcbClient)
            async with ecb_type(cache) as ecb:
                risk_free_task = ecb.get_series(
                    "IRS",
                    "M.DK.L.L40.CI.0000.DKK.N.Z",
                    kind=ReferenceSeriesKind.GOVERNMENT_YIELD,
                    unit=ReferenceValueUnit.PERCENTAGE_POINTS,
                    start_period=today - datetime.timedelta(days=120),
                    end_period=today,
                    use_cache=use_cache,
                    make_cache=True,
                )
                if needs_terminal:
                    risk_free, inflation = await asyncio.gather(
                        risk_free_task,
                        ecb.get_series(
                            "HICP",
                            "M.DK.N.000000.4D0.ANR",
                            kind=ReferenceSeriesKind.INFLATION_RATE,
                            unit=ReferenceValueUnit.PERCENT_CHANGE,
                            start_period=inflation_start,
                            end_period=today,
                            use_cache=use_cache,
                            make_cache=True,
                        ),
                    )
                    inputs["inflation_series"] = inflation
                else:
                    risk_free = await risk_free_task
            inputs["risk_free_series"] = risk_free
        elif normalized_currency == "USD":
            treasury_type = dependency(context, "TreasuryClient", TreasuryClient)
            async with treasury_type(cache) as treasury:
                inputs["risk_free_series"] = await treasury.get_par_yield(
                    120,
                    use_cache=use_cache,
                    make_cache=True,
                )
            fred_key = dependency(context, "FRED_API_KEY", FRED_API_KEY)
            fred_type = dependency(context, "FredClient", FredClient)
            if needs_terminal and fred_key:
                async with fred_type(cache, fred_key) as fred:
                    inputs["inflation_series"] = await fred.get_series(
                        "FPCPITOTLZGUSA",
                        kind=ReferenceSeriesKind.INFLATION_RATE,
                        unit=ReferenceValueUnit.PERCENT_CHANGE,
                        observation_start=datetime.date.today()
                        - datetime.timedelta(days=365 * 15),
                        observation_end=datetime.date.today(),
                        country="US",
                        use_cache=use_cache,
                        make_cache=True,
                    )
        else:
            raise ValueError(
                f"automatic macro assumptions currently support DKK, EUR, and USD, not "
                f"{normalized_currency}; set risk_free_rate/WACC and terminal growth "
                "in the profile"
            )
    except RuntimeError as exc:
        raise ValueError(
            "Automatic valuation assumptions could not retrieve the sovereign-yield "
            "or inflation series; provide risk_free_rate/WACC and terminal growth in "
            f"the profile. Cause: {exc}"
        ) from exc
    return inputs


def apply_classification_overrides(
    classification,
    *,
    sector=None,
    industry: Optional[str] = None,
    context=None,
):
    del context
    updates = {}
    if sector is not None:
        updates["sector"] = sector
        updates["sector_taxonomy"] = "valuation-profile"
    if industry is not None:
        updates["industry"] = industry
        updates["industry_taxonomy"] = "valuation-profile"
    return classification.model_copy(update=updates) if updates else classification


def resolve_wacc(
    override: Optional[Decimal], configuration, *, context=None
) -> tuple[Decimal, str]:
    if override is not None:
        return override, "explicit CLI override"
    if configuration.wacc is not None:
        return configuration.wacc, "explicit valuation profile"
    beta = configuration.levered_beta
    if beta is None and configuration.unlevered_beta is not None:
        required = {
            "market_value_debt": configuration.market_value_debt,
            "market_value_equity": configuration.market_value_equity,
            "normalized_tax_rate": configuration.normalized_tax_rate,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("Levering the profile beta requires: " + ", ".join(missing))
        assert configuration.market_value_debt is not None
        assert configuration.market_value_equity is not None
        assert configuration.normalized_tax_rate is not None
        discount_rate_service = dependency(
            context, "DiscountRateService", DiscountRateService
        )
        beta = discount_rate_service.lever_beta(
            configuration.unlevered_beta,
            configuration.market_value_debt,
            configuration.market_value_equity,
            configuration.normalized_tax_rate,
        )
    cost_of_equity = configuration.cost_of_equity
    if cost_of_equity is None:
        capm_inputs = {
            "risk_free_rate": configuration.risk_free_rate,
            "levered_beta": beta,
            "equity_risk_premium": configuration.equity_risk_premium,
        }
        missing = [name for name, value in capm_inputs.items() if value is None]
        if missing:
            raise ValueError(
                "FCFF DCF requires WACC. Provide --wacc, set valuation.discount_rates.wacc, "
                "or complete the profile CAPM/WACC inputs. Missing: "
                + ", ".join(missing)
            )
        assert configuration.risk_free_rate is not None
        assert beta is not None
        assert configuration.equity_risk_premium is not None
        discount_rate_service = dependency(
            context, "DiscountRateService", DiscountRateService
        )
        cost_of_equity = discount_rate_service.cost_of_equity(
            configuration.risk_free_rate,
            beta,
            configuration.equity_risk_premium,
            configuration.country_risk_premium or Decimal(0),
        ).cost_of_equity
    wacc_inputs = {
        "pretax_cost_of_debt": configuration.pretax_cost_of_debt,
        "normalized_tax_rate": configuration.normalized_tax_rate,
        "market_value_equity": configuration.market_value_equity,
        "market_value_debt": configuration.market_value_debt,
    }
    missing = [name for name, value in wacc_inputs.items() if value is None]
    if missing:
        raise ValueError(
            "FCFF DCF WACC calculation is missing profile inputs: " + ", ".join(missing)
        )
    assert configuration.pretax_cost_of_debt is not None
    assert configuration.normalized_tax_rate is not None
    assert configuration.market_value_equity is not None
    assert configuration.market_value_debt is not None
    discount_rate_service = dependency(
        context, "DiscountRateService", DiscountRateService
    )
    result = discount_rate_service.wacc(
        cost_of_equity,
        configuration.pretax_cost_of_debt,
        configuration.normalized_tax_rate,
        configuration.market_value_equity,
        configuration.market_value_debt,
    )
    return result.wacc, "derived from valuation profile CAPM and capital weights"


# Private aliases match the former application module.
_retrieve_forward_estimates = retrieve_forward_estimates
_merge_forward_growth_evidence = merge_forward_growth_evidence
_materialize_forward_revenue_anchors = materialize_forward_revenue_anchors
_financial_snapshot_warnings = financial_snapshot_warnings
_management_guidance_overlay = management_guidance_overlay
_forward_growth_evidence = forward_growth_evidence
_retrieve_automatic_assumption_inputs = retrieve_automatic_assumption_inputs
_apply_classification_overrides = apply_classification_overrides
_resolve_wacc = resolve_wacc


__all__ = [
    "_apply_classification_overrides",
    "_financial_snapshot_warnings",
    "_forward_growth_evidence",
    "_management_guidance_overlay",
    "_materialize_forward_revenue_anchors",
    "_merge_forward_growth_evidence",
    "_resolve_wacc",
    "_retrieve_automatic_assumption_inputs",
    "_retrieve_forward_estimates",
    "apply_classification_overrides",
    "financial_snapshot_warnings",
    "forward_growth_evidence",
    "management_guidance_overlay",
    "materialize_forward_revenue_anchors",
    "merge_forward_growth_evidence",
    "resolve_wacc",
    "retrieve_automatic_assumption_inputs",
    "retrieve_forward_estimates",
]
