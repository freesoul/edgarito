"""Orchestrate failure-isolated SEC and IR discovery for operating evidence."""

# Private helper imports are retained as compatibility re-exports.
# ruff: noqa: F401

from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any

from edgarito.schemas.operating import (
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
)
from edgarito.services.guidance.documents import (
    GuidanceDocumentSelector,
    clean_document_text,
    extract_operating_context,
    guidance_keyword_hits,
    is_exhibit_document,
    is_periodic_filing,
)
from edgarito.services.openai import OpenAIAuthenticationError
from edgarito.services.operating._discovery import audit, documents, ir, recovery
from edgarito.services.operating._discovery.audit import (
    finalize_history_audit,
    merge_vocabulary_audits,
)
from edgarito.services.operating._discovery.contracts import (
    DiscoveryState,
    OperatingForecastDiscovery,
    OperatingForecastDiscoveryResult,
    OperatingIrFallback,
)
from edgarito.services.operating._discovery.documents import (
    collect_sec_documents,
    initialize_vocabulary_audit,
    prepare_filings,
)
from edgarito.services.operating._discovery.ir import recover_from_ir
from edgarito.services.operating._discovery.recovery import (
    _assemble,
    append_unique,
    gap_terms,
    merge_extraction_entry,
    merge_segment_identity,
    recover_sec_gaps,
    retry_vocabulary,
    targeted_gap_context,
    targeted_gap_documents,
)
from edgarito.services.operating.extraction import (
    OperatingEvidenceExtractor,
    operating_keyword_hits,
)
from edgarito.services.operating.history import OperatingHistoryAssembler
from edgarito.services.operating.vocabulary import KpiVocabularyProvider
from edgarito.services.providers.edgar import EdgarClient


class OperatingEvidenceDiscoveryService:
    """Discover first-party operating evidence without owning forecasting."""

    def __init__(
        self,
        edgar: EdgarClient,
        extractor: OperatingEvidenceExtractor,
        *,
        selector=None,
        lookback_days: int = 1825,
        max_filings: int = 24,
        max_documents_per_filing: int = 4,
        max_documents: int = 12,
        max_gap_documents: int = 4,
        history_assembler: OperatingHistoryAssembler | None = None,
        vocabulary_provider: KpiVocabularyProvider | None = None,
        ir_fallback: OperatingIrFallback | None = None,
    ) -> None:
        self._edgar = edgar
        self._extractor = extractor
        self._selector = selector or GuidanceDocumentSelector()
        self.lookback_days = max(0, lookback_days)
        self.max_filings = max(0, max_filings)
        self.max_documents_per_filing = max(0, max_documents_per_filing)
        self.max_documents = max(0, max_documents)
        self.max_gap_documents = max(0, max_gap_documents)
        self._history_assembler = history_assembler or OperatingHistoryAssembler()
        self._vocabulary = vocabulary_provider
        self._ir_fallback = ir_fallback

    async def discover(
        self,
        *,
        ticker: str | None = None,
        cik: int | None = None,
        financials: Any | None = None,
        company_id: str | int | None = None,
        as_of: datetime.date,
        refresh_sec: bool = False,
        valuation_date: datetime.date | None = None,
        fiscal_years: tuple[int, ...] | None = None,
        industry: str | None = None,
        sector: str | None = None,
        business_archetype: Any | None = None,
        profile_metadata: Mapping[str, Any] | None = None,
    ) -> OperatingForecastDiscoveryResult:
        """Coordinate inventory, extraction, recovery, and audit assembly."""

        del sector  # retained in the public seam for provider-neutral callers
        if valuation_date is not None and valuation_date != as_of:
            return OperatingForecastDiscoveryResult(
                warnings=(
                    "Operating evidence discovery skipped: as_of and "
                    "valuation_date differ",
                )
            )
        warnings: list[str] = []
        if self._vocabulary is not None:
            vocabulary_context = self._vocabulary.normal_terms(
                industry=industry,
                business_archetype=business_archetype,
            )
            if not vocabulary_context:
                warnings.append("KPI vocabulary is explicitly disabled")

        preparation, preparation_warning = await prepare_filings(
            self,
            ticker=ticker,
            cik=cik,
            financials=financials,
            company_id=company_id,
            as_of=as_of,
            refresh_sec=refresh_sec,
            fiscal_years=fiscal_years,
        )
        if preparation is None:
            return OperatingForecastDiscoveryResult(
                warnings=tuple((*warnings, preparation_warning or ""))
            )

        state = DiscoveryState(
            warnings=warnings,
            raw_filings_received=preparation.raw_filings_received,
            raw_filings_in_range=preparation.raw_filings_in_range,
            candidate_filings=preparation.candidate_filings,
        )
        initialize_vocabulary_audit(
            self,
            state,
            industry=industry,
            business_archetype=business_archetype,
        )
        await collect_sec_documents(
            self,
            preparation,
            state,
            as_of=as_of,
            refresh_sec=refresh_sec,
            industry=industry,
            business_archetype=business_archetype,
        )

        company_key = str(company_id or preparation.cik or "company")
        history = _assemble(self, state, company_key)
        historical_revenue = history.engine_historical_revenue
        state.warnings.extend(history.audit.warnings)
        if history.audit.missing_pairs:
            state.warnings.append(
                "Operating history missing required KPI pairs: "
                + ", ".join(history.audit.missing_pairs)
            )

        history, initial_history, detected_gaps = await recover_sec_gaps(
            self,
            state,
            history=history,
            company_id=company_key,
            as_of=as_of,
            extraction_years=preparation.extraction_years,
            industry=industry,
            business_archetype=business_archetype,
        )
        historical_revenue = history.engine_historical_revenue
        history, _retry_reason = await retry_vocabulary(
            self,
            state,
            history=history,
            company_id=company_key,
            as_of=as_of,
            extraction_years=preparation.extraction_years,
            industry=industry,
            business_archetype=business_archetype,
        )
        historical_revenue = history.engine_historical_revenue

        history, ir_resolved, ir_diagnostic = await recover_from_ir(
            self,
            state,
            history=history,
            financials=financials,
            profile_metadata=profile_metadata,
            company_id=company_key,
            as_of=as_of,
            extraction_years=preparation.extraction_years,
        )
        historical_revenue = history.engine_historical_revenue
        history, resolved_gaps, unresolved_gaps = finalize_history_audit(
            history,
            initial_history,
            detected_gaps,
            ir_resolved,
        )
        if resolved_gaps:
            state.warnings.append(
                "Operating evidence gaps resolved: "
                + ", ".join(item.label for item in resolved_gaps)
            )
        if unresolved_gaps:
            state.warnings.append(
                "Operating evidence gaps unresolved: "
                + ", ".join(item.label for item in unresolved_gaps)
            )
        if ir_diagnostic:
            state.warnings.append(ir_diagnostic)
        if state.unsupported:
            state.warnings.append(
                f"Operating evidence rejected {len(set(state.unsupported))} "
                "unsupported claim(s)"
            )
        if state.missing:
            state.warnings.append(
                f"Operating evidence rejected {len(set(state.missing))} item(s) "
                "with missing support"
            )
        if state.unusable:
            state.warnings.append(
                f"Operating evidence marked {len(set(state.unusable))} item(s) "
                "unusable"
            )

        return OperatingForecastDiscoveryResult(
            segments=tuple(state.segments),
            definitions=tuple(state.definitions),
            observations=tuple(state.observations),
            investment_programs=tuple(state.programs),
            management_constraints=tuple(state.management_constraints),
            historical_revenue=historical_revenue,
            history_audit=history.audit,
            rejected=tuple(state.rejected),
            warnings=tuple(dict.fromkeys(state.warnings)),
            unsupported_evidence=tuple(dict.fromkeys(state.unsupported)),
            missing_evidence=tuple(dict.fromkeys(state.missing)),
            unusable_evidence=tuple(dict.fromkeys(state.unusable)),
            audit_records=tuple(state.audit_records),
            document_audits=tuple(state.document_audits),
            cache_hits=state.hits,
            cache_misses=state.misses,
            filings_inspected=state.filings_inspected,
            raw_filings_received=state.raw_filings_received,
            raw_filings_in_range=state.raw_filings_in_range,
            candidate_filings=state.candidate_filings,
            filing_inventory_cache_bypass=refresh_sec,
            filing_inventory_fetched_live=refresh_sec,
            filing_inventory_metadata=tuple(
                f"{item.filing_date.isoformat()} | {item.form} | "
                f"{item.accession_number} | {item.primary_document}"
                for item in preparation.filings
            ),
            documents_inspected=state.documents_inspected,
            vocabulary_audit=merge_vocabulary_audits(state.vocabulary_audits),
            vocabulary_terms=tuple(
                item for item in state.vocabulary_terms if hasattr(item, "raw_term")
            ),
            gaps_detected=history.audit.gaps_detected,
            gaps_resolved=history.audit.gaps_resolved,
            gaps_unresolved=history.audit.gaps_unresolved,
            exhibits_found=state.exhibits_found,
            gaps_resolved_sec=history.audit.gaps_resolved_sec,
            gaps_resolved_ir=ir_resolved,
            ir_diagnostic=ir_diagnostic,
        )

    async def retrieve(self, **kwargs: Any) -> OperatingForecastDiscoveryResult:
        """ManagementGuidanceService-compatible name for discovery callers."""

        return await self.discover(**kwargs)

    async def discover_for_company(
        self,
        company_id: str | int,
        *,
        as_of: datetime.date,
        **kwargs: Any,
    ) -> OperatingForecastDiscoveryResult:
        """Discover by a provider-neutral company identifier when it is a CIK."""

        try:
            cik = int(company_id)
        except (TypeError, ValueError):
            return OperatingForecastDiscoveryResult(
                warnings=(
                    "Operating evidence discovery requires a numeric SEC CIK "
                    "when no ticker resolver is supplied",
                )
            )
        return await self.discover(cik=cik, as_of=as_of, **kwargs)

    # Compatibility seams for callers that used the former monolith's helpers.
    _append_unique = staticmethod(append_unique)
    _gap_terms = staticmethod(gap_terms)
    _targeted_gap_documents = classmethod(
        lambda cls, retry_documents, gaps, vocabulary_terms: targeted_gap_documents(
            retry_documents, gaps, vocabulary_terms
        )
    )
    _targeted_gap_context = classmethod(
        lambda cls, clean_text, gaps, vocabulary_terms, **kwargs: targeted_gap_context(
            clean_text, gaps, vocabulary_terms, **kwargs
        )
    )
    _merge_extraction_entry = classmethod(
        lambda cls, entry, **kwargs: merge_extraction_entry(
            entry,
            state=DiscoveryState(
                segments=kwargs["segments"],
                definitions=kwargs["definitions"],
                observations=kwargs["observations"],
                programs=kwargs["programs"],
                management_constraints=kwargs["management_constraints"],
                rejected=kwargs["rejected"],
                audit_records=kwargs["audit_records"],
                seen_segments=kwargs["seen_segments"],
                seen_definitions=kwargs["seen_definitions"],
                seen_observations=kwargs["seen_observations"],
                seen_programs=kwargs["seen_programs"],
                unsupported=kwargs["unsupported"],
                missing=kwargs["missing"],
                unusable=kwargs["unusable"],
                warnings=kwargs["warnings"],
            ),
        )
    )


# Descriptive aliases for the common naming conventions.
OperatingForecastDiscoveryService = OperatingEvidenceDiscoveryService
OperatingDriverDiscoveryService = OperatingEvidenceDiscoveryService
OperatingEvidenceDiscovery = OperatingEvidenceDiscoveryService
_merge_segment_identity = merge_segment_identity
_merge_vocabulary_audits = merge_vocabulary_audits


__all__ = [
    "OperatingDriverDiscoveryService",
    "OperatingEvidenceDiscovery",
    "OperatingEvidenceDiscoveryService",
    "OperatingForecastDiscovery",
    "OperatingForecastDiscoveryResult",
    "OperatingForecastDiscoveryService",
    "OperatingIrFallback",
]
