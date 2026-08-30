"""CLI use cases for structured operating evidence and activation audits."""

from __future__ import annotations

import argparse
import inspect
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator, Mapping

from edgarito.cli.use_cases.context import call_with_context, dependency
from edgarito.cli.use_cases.forecast import market_for_args
from edgarito.enums.market import Market
from edgarito.schemas.normalization.financials import NormalizedCompanyFinancials
from edgarito.schemas.vocabulary import KpiVocabularyAudit
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.openai import OpenAIClient
from edgarito.services.operating._discovery.service import (
    OperatingEvidenceDiscoveryService,
)
from edgarito.services.operating.extraction import OperatingEvidenceExtractor
from edgarito.services.operating.integration import OperatingForecastPipelineService
from edgarito.services.operating.vocabulary import (
    KpiVocabularyProvider,
    normalize_industry_namespace,
)
from edgarito.services.providers.edgar import EdgarClient
from edgarito.settings import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_REASONING_EFFORT,
)


async def retrieve_operating_evidence(
    financials: NormalizedCompanyFinancials,
    forecast,
    as_of,
    *,
    provider=None,
    args: argparse.Namespace | None = None,
    metadata: Mapping[str, object] | None = None,
    fiscal_years=None,
    availability_mode=None,
    context=None,
) -> tuple[object | None, tuple[str, ...]]:
    provider = (
        dependency(context, "OPERATING_EVIDENCE_PROVIDER", None)
        if provider is None
        else provider
    )
    if provider is None:
        return None, ()
    resolver = getattr(provider, "discover", None) or getattr(provider, "retrieve", None)
    if resolver is None and callable(provider):
        resolver = provider
    if resolver is None:
        return None, (
            "Structured operating evidence provider is invalid: expected "
            "discover, retrieve, or a callable",
        )
    try:
        resolver_kwargs = {
            "financials": financials,
            "company_id": financials.company_id,
            "as_of": as_of,
            "fiscal_years": (
                tuple(item.fiscal_year for item in forecast.observations)
                if forecast is not None
                else tuple(fiscal_years or ())
            ),
            "industry": getattr(financials, "industry", None),
            "business_archetype": getattr(financials, "business_archetype", None),
            **(metadata or {}),
        }
        if availability_mode is not None:
            resolver_kwargs["availability_mode"] = availability_mode
        if args is not None:
            resolver_kwargs.update(
                {
                    "ticker": args.ticker or financials.ticker,
                    "cik": args.cik,
                    "refresh_sec": args.refresh
                    or getattr(args, "refresh_sec", False),
                }
            )
        evidence = call_with_context(
            dependency(
                context,
                "_call_with_supported_kwargs",
                call_with_supported_kwargs,
            ),
            resolver,
            resolver_kwargs,
            context=context,
        )
        if hasattr(evidence, "__await__"):
            evidence = await evidence
        warnings = tuple(
            evidence.get("warnings", ())
            if isinstance(evidence, Mapping)
            else getattr(evidence, "warnings", ())
        )
        return evidence, warnings
    except Exception as exc:
        return None, (f"Structured operating evidence unavailable: {exc}",)


def operating_quality_audit(
    evidence,
    *,
    discovery_warnings: tuple[str, ...] = (),
    context=None,
):
    value = evidence if isinstance(evidence, Mapping) else {}
    values = {
        "definitions": tuple(
            (
                value.get("definitions", ())
                if value
                else getattr(evidence, "definitions", ())
            )
            or ()
        ),
        "observations": tuple(
            (
                value.get("observations", ())
                if value
                else getattr(evidence, "observations", ())
            )
            or ()
        ),
        "history_audit": (
            value.get("history_audit")
            if value
            else getattr(evidence, "history_audit", None)
        ),
        "exhibits_found": int(
            value.get("exhibits_found", 0)
            if value
            else getattr(evidence, "exhibits_found", 0)
        ),
        "gaps_resolved_sec": tuple(
            (
                value.get("gaps_resolved_sec", ())
                if value
                else getattr(evidence, "gaps_resolved_sec", ())
            )
            or ()
        ),
        "gaps_resolved_ir": tuple(
            (
                value.get("gaps_resolved_ir", ())
                if value
                else getattr(evidence, "gaps_resolved_ir", ())
            )
            or ()
        ),
        "ir_diagnostic": (
            value.get("ir_diagnostic")
            if value
            else getattr(evidence, "ir_diagnostic", None)
        ),
    }
    pipeline_service = dependency(
        context,
        "OperatingForecastPipelineService",
        OperatingForecastPipelineService,
    )
    preliminary = pipeline_service.quality_gate(values)
    return replace(
        preliminary,
        warnings=tuple(
            dict.fromkeys(
                (
                    *discovery_warnings,
                    *(
                        (
                            value.get("warnings", ())
                            if value
                            else getattr(evidence, "warnings", ())
                        )
                        or ()
                    ),
                )
            )
        ),
        audit_records=tuple(
            (
                value.get("audit_records", ())
                if value
                else getattr(evidence, "audit_records", ())
            )
            or ()
        ),
        document_audits=tuple(
            (
                value.get("document_audits", ())
                if value
                else getattr(evidence, "document_audits", ())
            )
            or ()
        ),
        modeled_revenue_share=(
            value.get("modeled_revenue_share", None)
            if value
            else getattr(evidence, "modeled_revenue_share", None)
        ),
        unusable_evidence=tuple(
            (
                value.get("unusable_evidence", ())
                if value
                else getattr(evidence, "unusable_evidence", ())
            )
            or ()
        ),
        history_audit=(
            value.get("history_audit")
            if value
            else getattr(evidence, "history_audit", None)
        ),
        cache_hits=int(
            value.get("cache_hits", 0) if value else getattr(evidence, "cache_hits", 0)
        ),
        cache_misses=int(
            value.get("cache_misses", 0)
            if value
            else getattr(evidence, "cache_misses", 0)
        ),
        filings_inspected=int(
            value.get("filings_inspected", 0)
            if value
            else getattr(evidence, "filings_inspected", 0)
        ),
        documents_inspected=int(
            value.get("documents_inspected", 0)
            if value
            else getattr(evidence, "documents_inspected", 0)
        ),
        raw_filings_received=int(
            value.get("raw_filings_received", 0)
            if value
            else getattr(evidence, "raw_filings_received", 0)
        ),
        raw_filings_in_range=int(
            value.get("raw_filings_in_range", 0)
            if value
            else getattr(evidence, "raw_filings_in_range", 0)
        ),
        candidate_filings=int(
            value.get("candidate_filings", 0)
            if value
            else getattr(evidence, "candidate_filings", 0)
        ),
        filing_inventory_cache_bypass=bool(
            value.get("filing_inventory_cache_bypass", False)
            if value
            else getattr(evidence, "filing_inventory_cache_bypass", False)
        ),
        filing_inventory_fetched_live=bool(
            value.get("filing_inventory_fetched_live", False)
            if value
            else getattr(evidence, "filing_inventory_fetched_live", False)
        ),
        filing_inventory_metadata=tuple(
            value.get("filing_inventory_metadata", ())
            if value
            else getattr(evidence, "filing_inventory_metadata", ())
        ),
        vocabulary_audit=(
            value.get("vocabulary_audit")
            if value
            else getattr(evidence, "vocabulary_audit", None)
        ),
        vocabulary_terms=tuple(
            (
                value.get("vocabulary_terms", ())
                if value
                else getattr(evidence, "vocabulary_terms", ())
            )
            or ()
        ),
        exhibits_found=int(values.get("exhibits_found", 0) or 0),
        gaps_resolved_sec=tuple(values.get("gaps_resolved_sec") or ()),
        gaps_resolved_ir=tuple(values.get("gaps_resolved_ir") or ()),
        ir_diagnostic=values.get("ir_diagnostic"),
    )


def retain_operating_audit_metadata(current, discovered):
    if discovered is None:
        return current
    updates = {}
    for field in (
        "vocabulary_audit",
        "vocabulary_terms",
        "raw_filings_received",
        "raw_filings_in_range",
        "candidate_filings",
        "filing_inventory_cache_bypass",
        "filing_inventory_fetched_live",
        "filing_inventory_metadata",
    ):
        value = getattr(discovered, field, None)
        if field.startswith("raw_") or field in {
            "candidate_filings",
            "filing_inventory_cache_bypass",
            "filing_inventory_fetched_live",
            "filing_inventory_metadata",
        }:
            if value:
                updates[field] = value
            continue
        if field == "vocabulary_audit" and value is not None:
            if getattr(value, "global_count", 0) == 0:
                continue
        if value:
            updates[field] = value
    return replace(current, **updates) if updates else current


def default_operating_vocabulary_audit(profile, *, context=None):
    industry = getattr(profile.model_selection, "industry", None)
    archetype = getattr(profile.model_selection, "business_archetype", None)
    provider_type = dependency(context, "KpiVocabularyProvider", KpiVocabularyProvider)
    provider = provider_type()
    terms = provider.normal_terms(industry, archetype)
    return KpiVocabularyAudit(
        global_count=len(provider.GLOBAL_KPI_TERMS),
        industry_count=max(0, len(terms) - len(provider.GLOBAL_KPI_TERMS)),
        terms=tuple(term for term, _metric in terms),
        raw_industry=str(industry or ""),
        normalized_industry=normalize_industry_namespace(industry),
        selected_archetype=str(getattr(archetype, "value", archetype) or ""),
        cache_status="not_needed",
    )


@asynccontextmanager
async def operating_evidence_provider(
    args: argparse.Namespace,
    financials: NormalizedCompanyFinancials,
    *,
    market: Market | str | None = None,
    context=None,
) -> AsyncIterator[tuple[object | None, str | None]]:
    del financials
    market_resolver = dependency(context, "_market_for_args", market_for_args)
    market = (
        call_with_context(market_resolver, args, context=context)
        if market is None
        else Market(market)
    )
    if market != Market.US:
        yield None, f"SEC-backed operating evidence skipped for the {market.value} market"
        return
    configured_provider = dependency(context, "OPERATING_EVIDENCE_PROVIDER", None)
    if configured_provider is not None:
        yield configured_provider, None
        return
    api_key = dependency(context, "OPENAI_API_KEY", OPENAI_API_KEY)
    if not api_key or not str(api_key).strip():
        yield None, "OpenAI API key missing"
        return
    user_agent = getattr(args, "user_agent", None)
    if not user_agent or not str(user_agent).strip():
        yield None, "SEC user-agent missing"
        return
    cache_type = dependency(context, "FileSystemCache", FileSystemCache)
    openai_type = dependency(context, "OpenAIClient", OpenAIClient)
    edgar_type = dependency(context, "EdgarClient", EdgarClient)
    discovery_type = dependency(
        context,
        "OperatingEvidenceDiscoveryService",
        OperatingEvidenceDiscoveryService,
    )
    extractor_type = dependency(
        context,
        "OperatingEvidenceExtractor",
        OperatingEvidenceExtractor,
    )
    vocabulary_type = dependency(
        context,
        "KpiVocabularyProvider",
        KpiVocabularyProvider,
    )
    cache = cache_type(Path(args.cache_dir))
    try:
        openai_client = openai_type(
            api_key=api_key,
            model=dependency(context, "OPENAI_MODEL", OPENAI_MODEL),
            reasoning_effort=dependency(
                context, "OPENAI_REASONING_EFFORT", OPENAI_REASONING_EFFORT
            ),
        )
    except Exception as exc:
        yield None, f"Operating forecast provider unavailable: {exc}"
        return
    try:
        async with edgar_type(cache, str(user_agent)) as edgar:
            yield (
                discovery_type(
                    edgar,
                    extractor_type(openai_client, cache),
                    vocabulary_provider=vocabulary_type(openai_client, cache),
                ),
                None,
            )
    finally:
        await openai_client.close()


def call_with_supported_kwargs(resolver, kwargs: dict[str, object]):
    try:
        signature = inspect.signature(resolver)
    except (TypeError, ValueError):
        return resolver(**kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return resolver(**kwargs)
    return resolver(
        **{
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters
        }
    )


# Private aliases match the former application module.
_retrieve_operating_evidence = retrieve_operating_evidence
_operating_quality_audit = operating_quality_audit
_retain_operating_audit_metadata = retain_operating_audit_metadata
_default_operating_vocabulary_audit = default_operating_vocabulary_audit
_operating_evidence_provider = operating_evidence_provider
_call_with_supported_kwargs = call_with_supported_kwargs


__all__ = [
    "_call_with_supported_kwargs",
    "_default_operating_vocabulary_audit",
    "_operating_evidence_provider",
    "_operating_quality_audit",
    "_retrieve_operating_evidence",
    "_retain_operating_audit_metadata",
    "call_with_supported_kwargs",
    "default_operating_vocabulary_audit",
    "operating_evidence_provider",
    "operating_quality_audit",
    "retrieve_operating_evidence",
    "retain_operating_audit_metadata",
]
