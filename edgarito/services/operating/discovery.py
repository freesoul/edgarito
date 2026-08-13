"""Failure-isolated SEC discovery for operating-driver evidence."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from edgarito.schemas.operating import (
    OperatingArchetype,
    OperatingDocumentAudit,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingEvidenceAuditRecord,
    OperatingEvidenceRejection,
    OperatingInvestmentProgram,
    OperatingSegment,
)
from edgarito.schemas.operating_history import OperatingHistoryAudit
from edgarito.schemas.vocabulary import KpiVocabularyAudit
from edgarito.services.guidance.documents import (
    GuidanceDocumentSelector,
    clean_document_text,
    extract_operating_context,
    guidance_keyword_hits,
    is_periodic_filing,
)
from edgarito.services.openai import OpenAIAuthenticationError
from edgarito.services.operating.extraction import (
    OperatingEvidenceExtractor,
    operating_keyword_hits,
)
from edgarito.services.operating.history import OperatingHistoryAssembler
from edgarito.services.operating.vocabulary import (
    KpiVocabularyProvider,
    normalize_industry_namespace,
)
from edgarito.services.providers.edgar import EdgarClient


@dataclass(frozen=True)
class OperatingForecastDiscoveryResult:
    """Normalized operating evidence and content-free discovery diagnostics."""

    segments: tuple[OperatingSegment, ...] = ()
    definitions: tuple[OperatingDriverDefinition, ...] = ()
    observations: tuple[OperatingDriverObservation, ...] = ()
    investment_programs: tuple[OperatingInvestmentProgram, ...] = ()
    management_constraints: tuple[OperatingDriverObservation, ...] = ()
    historical_revenue: Mapping[Any, Any] | None = None
    history_audit: OperatingHistoryAudit | None = None
    rejected: tuple[OperatingEvidenceRejection, ...] = ()
    warnings: tuple[str, ...] = ()
    unsupported_evidence: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    unusable_evidence: tuple[str, ...] = ()
    audit_records: tuple[OperatingEvidenceAuditRecord, ...] = ()
    document_audits: tuple[OperatingDocumentAudit, ...] = ()
    cache_hits: int = 0
    cache_misses: int = 0
    filings_inspected: int = 0
    documents_inspected: int = 0
    vocabulary_audit: Any | None = None
    vocabulary_terms: tuple[Any, ...] = ()

    @property
    def drivers(self) -> tuple[OperatingDriverDefinition, ...]:
        """Compatibility alias for callers that call definitions drivers."""

        return self.definitions

    @property
    def archetypes(self) -> tuple[OperatingArchetype, ...]:
        """Unique archetypes in stable definition order for concise audits."""

        result: list[OperatingArchetype] = []
        for definition in self.definitions:
            if definition.archetype not in result:
                result.append(definition.archetype)
        return tuple(result)

    @property
    def extracted_records(self) -> int:
        return (
            len(self.segments)
            + len(self.definitions)
            + len(self.observations)
            + len(self.investment_programs)
        )

    @property
    def rejected_records(self) -> int:
        return len(self.rejected)

    @property
    def audits(self) -> tuple[OperatingEvidenceAuditRecord, ...]:
        return self.audit_records

    @property
    def available(self) -> bool:
        return bool(
            self.segments
            or self.definitions
            or self.observations
            or self.investment_programs
        )

    def integration_inputs(self) -> dict[str, Any]:
        """Return only the contracts consumed by the forecast integration seam."""

        return {
            "segments": self.segments,
            "definitions": self.definitions,
            "observations": self.observations,
            "management_constraints": self.management_constraints,
            "historical_revenue": self.historical_revenue,
            "history_audit": self.history_audit,
            "audit_records": self.audit_records,
            "document_audits": self.document_audits,
            "unusable_evidence": self.unusable_evidence,
        }


class OperatingForecastDiscovery(Protocol):
    def discover(self, *args: Any, **kwargs: Any) -> Any:
        """Return normalized operating evidence for the integration seam."""


class OperatingEvidenceDiscoveryService:
    """Discover first-party operating evidence without owning forecasting.

    SEC retrieval, document selection, bounded context, extraction, and cache
    diagnostics are deliberately isolated here.  Any optional discovery
    failure becomes a warning and an empty result; the deterministic operating
    engine and valuation can continue with other evidence sources.
    """

    def __init__(
        self,
        edgar: EdgarClient,
        extractor: OperatingEvidenceExtractor,
        *,
        selector: GuidanceDocumentSelector | None = None,
        lookback_days: int = 1825,
        max_filings: int = 6,
        max_documents_per_filing: int = 4,
        max_documents: int = 12,
        history_assembler: OperatingHistoryAssembler | None = None,
        vocabulary_provider: KpiVocabularyProvider | None = None,
    ) -> None:
        self._edgar = edgar
        self._extractor = extractor
        self._selector = selector or GuidanceDocumentSelector()
        self.lookback_days = max(0, lookback_days)
        self.max_filings = max(0, max_filings)
        self.max_documents_per_filing = max(0, max_documents_per_filing)
        self.max_documents = max(0, max_documents)
        self._history_assembler = history_assembler or OperatingHistoryAssembler()
        self._vocabulary = vocabulary_provider

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
    ) -> OperatingForecastDiscoveryResult:
        """Return evidence for one valuation date, isolating provider failures."""

        if ticker is None and financials is not None:
            ticker = getattr(financials, "ticker", None)
        if cik is None and company_id is not None:
            try:
                cik = int(company_id)
            except (TypeError, ValueError):
                pass
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
        if cik is None:
            if not ticker:
                return OperatingForecastDiscoveryResult(
                    warnings=(
                        "Operating evidence discovery skipped: no SEC identifier",
                    )
                )
            try:
                cik = await self._edgar.get_cik(
                    ticker,
                    use_cache=not refresh_sec,
                    make_cache=True,
                )
            except Exception as exc:
                return OperatingForecastDiscoveryResult(
                    warnings=(
                        f"Operating evidence discovery skipped: SEC identifier lookup failed: {exc}",
                    )
                )

        try:
            get_filings = getattr(self._edgar, "get_operating_filings", None)
            if get_filings is None:
                get_filings = self._edgar.get_guidance_filings
            filings = await get_filings(
                cik,
                as_of=as_of,
                lookback_days=self.lookback_days,
                use_cache=not refresh_sec,
                make_cache=True,
            )
        except Exception as exc:
            return OperatingForecastDiscoveryResult(
                warnings=(
                    f"Operating evidence discovery skipped: SEC filing retrieval failed: {exc}",
                )
            )

        try:
            selected_filings = self._selector.select_operating_filings(
                filings,
                limit=self.max_filings,
            )
        except Exception as exc:
            return OperatingForecastDiscoveryResult(
                warnings=(
                    f"Operating evidence discovery skipped: filing selection failed: {exc}",
                )
            )

        # Match management-guidance ordering: preserve periodic primary reports
        # before exhibits consume the global document budget.
        selected_filings = [
            *[filing for filing in selected_filings if is_periodic_filing(filing)],
            *[filing for filing in selected_filings if not is_periodic_filing(filing)],
        ]
        extraction_years = (
            tuple(
                sorted(
                    {
                        *(fiscal_years or ()),
                        *(
                            filing.report_date.year
                            for filing in selected_filings
                            if filing.report_date is not None
                        ),
                        *(filing.filing_date.year for filing in selected_filings),
                    }
                )
            )
            or fiscal_years
        )

        segments: list[OperatingSegment] = []
        definitions: list[OperatingDriverDefinition] = []
        observations: list[OperatingDriverObservation] = []
        programs: list[OperatingInvestmentProgram] = []
        management_constraints: list[OperatingDriverObservation] = []
        rejected: list[OperatingEvidenceRejection] = []
        audit_records: list[OperatingEvidenceAuditRecord] = []
        document_audits: list[OperatingDocumentAudit] = []
        vocabulary_terms: list[Any] = []
        vocabulary_audits: list[Any] = []
        retry_documents: list[tuple[Any, Any, str]] = []
        discovered_by_document: dict[str, tuple[Any, ...]] = {}
        if self._vocabulary is not None:
            normal = self._vocabulary.normal_terms(industry, business_archetype)
            vocabulary_audits.append(
                KpiVocabularyAudit(
                    global_count=len(self._vocabulary.GLOBAL_KPI_TERMS)
                    if hasattr(self._vocabulary, "GLOBAL_KPI_TERMS")
                    else 0,
                    industry_count=self._vocabulary.industry_term_count(
                        industry, business_archetype
                    ),
                    terms=tuple(term for term, _metric in normal),
                    cache_status="not_needed",
                    raw_industry=str(industry or ""),
                    normalized_industry=normalize_industry_namespace(industry),
                    selected_archetype=str(
                        getattr(business_archetype, "value", business_archetype) or ""
                    ),
                )
            )
        unsupported: list[str] = []
        missing: list[str] = []
        unusable: list[str] = []
        hits = 0
        misses = 0
        documents_inspected = 0
        filings_inspected = 0
        seen_segments: dict[str, OperatingSegment] = {}
        seen_definitions: dict[tuple[str, str], OperatingDriverDefinition] = {}
        seen_observations: dict[
            tuple[str, str, int, str, str | None], OperatingDriverObservation
        ] = {}
        seen_programs: dict[
            tuple[str, int | None, str], OperatingInvestmentProgram
        ] = {}

        for filing_index, filing in enumerate(selected_filings):
            if documents_inspected >= self.max_documents:
                break
            filings_inspected += 1
            try:
                populated = await self._edgar.get_filing_documents(
                    filing,
                    use_cache=not refresh_sec,
                    make_cache=True,
                )
            except Exception as exc:
                warnings.append(
                    f"Operating document retrieval skipped for {filing.accession_number}: {exc}"
                )
                continue

            remaining_capacity = self.max_documents - documents_inspected
            remaining_filings = len(selected_filings) - filing_index - 1
            reserved_for_later = min(
                remaining_filings,
                max(0, remaining_capacity - 1),
            )
            document_limit = max(
                0,
                min(
                    self.max_documents_per_filing,
                    remaining_capacity - reserved_for_later,
                ),
            )
            try:
                documents = self._selector.select_operating_documents(
                    populated,
                    limit=document_limit,
                )
            except Exception as exc:
                warnings.append(
                    f"Operating document selection skipped for {filing.accession_number}: {exc}"
                )
                continue
            if not documents and any(
                document.is_pdf for document in populated.documents
            ):
                warnings.append(
                    f"{filing.form} {filing.accession_number} has no selected "
                    "HTML/text operating evidence document; PDF extraction is unsupported"
                )

            for document in documents:
                if documents_inspected >= self.max_documents:
                    break
                documents_inspected += 1
                clean_text = clean_document_text(document.content)
                context_text = extract_operating_context(clean_text)
                retry_documents.append((filing, document, clean_text))
                if self._vocabulary is not None:
                    try:
                        discovered, vocabulary_audit = await self._vocabulary.discover(
                            context=context_text,
                            source_document=document.filename,
                            source_text=clean_text,
                            industry=industry,
                            business_archetype=business_archetype,
                            as_of=as_of,
                        )
                        vocabulary_terms.extend(discovered)
                        discovered_by_document[document.filename] = tuple(discovered)
                        vocabulary_audits.append(vocabulary_audit)
                        if discovered:
                            context_text += (
                                "\n\nAdditional grounded KPI terminology: "
                                + ", ".join(item.raw_term for item in discovered)
                            )
                    except Exception as exc:
                        warnings.append(f"KPI vocabulary discovery skipped: {exc}")
                audit = OperatingDocumentAudit(
                    filing_form=filing.form,
                    filing_date=filing.filing_date,
                    accession_number=filing.accession_number,
                    filename=document.filename,
                    document_type=document.document_type,
                    is_primary=self._selector.is_primary(filing, document),
                    cleaned_size=len(clean_text),
                    bounded_context_size=len(context_text),
                    keyword_hits={
                        **guidance_keyword_hits(clean_text),
                        **operating_keyword_hits(clean_text),
                    },
                )
                if document.is_pdf:
                    warnings.append(
                        f"Operating evidence skipped for {filing.accession_number} "
                        f"{document.filename}: PDF extraction is unsupported"
                    )
                    document_audits.append(audit)
                    continue
                if not clean_text:
                    warnings.append(
                        f"Operating evidence skipped for {filing.accession_number} "
                        f"{document.filename}: document text is empty"
                    )
                    document_audits.append(audit)
                    continue
                try:
                    entry, cache_hit = await self._extractor.extract(
                        filing,
                        document,
                        clean_text,
                        valuation_date=as_of,
                        source_text=clean_text,
                        context_text=context_text,
                        fiscal_years=extraction_years,
                    )
                except OpenAIAuthenticationError as exc:
                    # Discovery is optional.  Unlike the existing valuation
                    # guidance path, this service is explicitly failure-isolated.
                    warnings.append(
                        f"Operating evidence extraction skipped for {filing.accession_number} "
                        f"{document.filename}: OpenAI authentication failed ({exc})"
                    )
                    document_audits.append(audit)
                    continue
                except Exception as exc:
                    warnings.append(
                        f"Operating evidence extraction skipped for {filing.accession_number} "
                        f"{document.filename}: {exc}"
                    )
                    document_audits.append(audit)
                    continue

                hits += int(cache_hit)
                misses += int(not cache_hit)
                for item in entry.segments:
                    item = _merge_segment_identity(
                        seen_segments.get(item.segment_id), item
                    )
                    self._append_unique(
                        segments,
                        seen_segments,
                        item.segment_id,
                        item,
                        warnings,
                        "segment",
                    )
                for item in entry.definitions:
                    self._append_unique(
                        definitions,
                        seen_definitions,
                        (item.segment_id, item.driver_id),
                        item,
                        warnings,
                        "driver definition",
                    )
                for item in entry.observations:
                    key = (
                        item.segment_id,
                        item.driver_id,
                        item.fiscal_year,
                        item.fiscal_period,
                        item.period_key,
                    )
                    added = self._append_unique(
                        observations,
                        seen_observations,
                        key,
                        item,
                        warnings,
                        "operating observation",
                    )
                    if added and item.origin == "management_guidance":
                        management_constraints.append(item)
                for item in entry.investment_programs:
                    key = (item.program_id, item.fiscal_year, item.fiscal_period)
                    self._append_unique(
                        programs,
                        seen_programs,
                        key,
                        item,
                        warnings,
                        "investment program",
                    )
                rejected.extend(entry.rejected)
                unsupported.extend(entry.unsupported_evidence)
                missing.extend(entry.missing_evidence)
                unusable.extend(entry.unusable_reasons)
                audit_records.extend(entry.audit_records)
                document_audits.append(
                    audit.model_copy(
                        update={
                            "accepted_segments": len(entry.segments),
                            "accepted_definitions": len(entry.definitions),
                            "accepted_observations": len(entry.observations),
                            "accepted_investment_programs": len(
                                entry.investment_programs
                            ),
                            "rejected_records": len(entry.rejected),
                            "unsupported_evidence": len(entry.unsupported_evidence),
                            "missing_evidence": len(entry.missing_evidence),
                            "unusable_evidence": len(entry.unusable_reasons),
                        }
                    )
                )
                for reason in entry.unusable_reasons:
                    warnings.append(f"Operating evidence unusable: {reason}")

        history = self._history_assembler.assemble(
            observations,
            segments=segments,
            definitions=definitions,
            company_id=str(company_id or cik or "company"),
        )
        # The assembler canonicalizes names and deduplicates repeated filing
        # identities. Keep the same canonical segment set for the downstream
        # forecast boundary; otherwise the raw per-document list can reintroduce
        # duplicate IDs after history assembly.
        segments = list(history.segments)
        definitions = list(history.definitions)
        observations = list(history.observations)
        historical_revenue = history.engine_historical_revenue
        warnings.extend(history.audit.warnings)
        if history.audit.missing_pairs:
            warnings.append(
                "Operating history missing required KPI pairs: "
                + ", ".join(history.audit.missing_pairs)
            )

        # Terminology discovery is evidence-quality aware.  A deterministic
        # keyword hit is not sufficient when extraction produced no usable
        # operating history, so retry the same documents with grounded terms.
        non_revenue = sum(
            1
            for item in observations
            if item.driver_id
            not in {
                "revenue",
                "segment_revenue",
                "sales",
                "net_sales",
                "total_revenue",
                "revenue_growth",
                "sales_growth",
            }
        )
        retry_reason = ""
        if not observations or non_revenue == 0:
            retry_reason = "accepted non-revenue observations are zero"
        elif not history.audit.accepted_pairs or history.audit.missing_pairs:
            retry_reason = "usable reconstruction pairs or coverage unavailable"
        if self._vocabulary is not None and retry_reason and retry_documents:
            retry_terms: list[Any] = []
            for filing, document, clean_text in retry_documents:
                try:
                    discovered = discovered_by_document.get(document.filename)
                    if not discovered:
                        discovered, audit = await self._vocabulary.discover(
                            context=extract_operating_context(clean_text),
                            source_document=document.filename,
                            source_text=clean_text,
                            industry=industry,
                            business_archetype=business_archetype,
                            as_of=as_of,
                            force=True,
                            fallback_reason=retry_reason,
                        )
                    else:
                        discovered = tuple(discovered)
                    audit = KpiVocabularyAudit(
                        global_count=len(self._vocabulary.GLOBAL_KPI_TERMS),
                        terms=tuple(
                            term
                            for term, _ in self._vocabulary.normal_terms(
                                industry, business_archetype
                            )
                        ),
                        raw_industry=str(industry or ""),
                        normalized_industry=normalize_industry_namespace(industry),
                        selected_archetype=str(
                            getattr(business_archetype, "value", business_archetype)
                            or ""
                        ),
                        fallback_triggered=True,
                        fallback_reason=retry_reason,
                        retry=True,
                        discovered_count=len(discovered),
                        validated_terms=tuple(item.raw_term for item in discovered),
                        cache_status="retry",
                    )
                    retry_terms.extend(discovered)
                    vocabulary_audits.append(audit)
                    if not discovered:
                        continue
                    context_text = (
                        extract_operating_context(clean_text)
                        + "\n\nAdditional grounded KPI terminology: "
                        + ", ".join(item.raw_term for item in discovered)
                    )
                    entry, cache_hit = await self._extractor.extract(
                        filing,
                        document,
                        clean_text,
                        valuation_date=as_of,
                        source_text=clean_text,
                        context_text=context_text,
                        fiscal_years=extraction_years,
                    )
                    hits += int(cache_hit)
                    misses += int(not cache_hit)
                    for item in entry.segments:
                        item = _merge_segment_identity(
                            seen_segments.get(item.segment_id), item
                        )
                        self._append_unique(
                            segments,
                            seen_segments,
                            item.segment_id,
                            item,
                            warnings,
                            "segment",
                        )
                    for item in entry.definitions:
                        self._append_unique(
                            definitions,
                            seen_definitions,
                            (item.segment_id, item.driver_id),
                            item,
                            warnings,
                            "driver definition",
                        )
                    for item in entry.observations:
                        key = (
                            item.segment_id,
                            item.driver_id,
                            item.fiscal_year,
                            item.fiscal_period,
                            item.period_key,
                        )
                        if (
                            self._append_unique(
                                observations,
                                seen_observations,
                                key,
                                item,
                                warnings,
                                "operating observation",
                            )
                            and item.origin == "management_guidance"
                        ):
                            management_constraints.append(item)
                    rejected.extend(entry.rejected)
                    unsupported.extend(entry.unsupported_evidence)
                    missing.extend(entry.missing_evidence)
                    unusable.extend(entry.unusable_reasons)
                    audit_records.extend(entry.audit_records)
                except Exception as exc:
                    warnings.append(
                        f"KPI vocabulary retry skipped for {document.filename}: {exc}"
                    )
            if retry_terms:
                vocabulary_terms.extend(retry_terms)
                vocabulary_audits.append(
                    KpiVocabularyAudit(
                        global_count=len(self._vocabulary.GLOBAL_KPI_TERMS),
                        industry_count=self._vocabulary.industry_term_count(
                            industry, business_archetype
                        ),
                        terms=tuple(
                            item[0]
                            for item in self._vocabulary.normal_terms(
                                industry, business_archetype
                            )
                        ),
                        discovered_count=len(retry_terms),
                        validated_terms=tuple(item.raw_term for item in retry_terms),
                        raw_industry=str(industry or ""),
                        normalized_industry=normalize_industry_namespace(industry),
                        selected_archetype=str(
                            getattr(business_archetype, "value", business_archetype)
                            or ""
                        ),
                        fallback_triggered=True,
                        fallback_reason=retry_reason,
                        retry=True,
                        new_observations=len(observations)
                        - history.audit.accepted_observations,
                        cache_status="retry",
                    )
                )

            history = self._history_assembler.assemble(
                observations,
                segments=segments,
                definitions=definitions,
                company_id=str(company_id or cik or "company"),
            )
            segments = list(history.segments)
            definitions = list(history.definitions)
            observations = list(history.observations)
            historical_revenue = history.engine_historical_revenue

        if unsupported:
            warnings.append(
                f"Operating evidence rejected {len(set(unsupported))} unsupported claim(s)"
            )
        if missing:
            warnings.append(
                f"Operating evidence rejected {len(set(missing))} item(s) with missing support"
            )
        if unusable:
            warnings.append(
                f"Operating evidence marked {len(set(unusable))} item(s) unusable"
            )
        return OperatingForecastDiscoveryResult(
            segments=tuple(segments),
            definitions=tuple(definitions),
            observations=tuple(observations),
            investment_programs=tuple(programs),
            management_constraints=tuple(management_constraints),
            historical_revenue=historical_revenue,
            history_audit=history.audit,
            rejected=tuple(rejected),
            warnings=tuple(dict.fromkeys(warnings)),
            unsupported_evidence=tuple(dict.fromkeys(unsupported)),
            missing_evidence=tuple(dict.fromkeys(missing)),
            unusable_evidence=tuple(dict.fromkeys(unusable)),
            audit_records=tuple(audit_records),
            document_audits=tuple(document_audits),
            cache_hits=hits,
            cache_misses=misses,
            filings_inspected=filings_inspected,
            documents_inspected=documents_inspected,
            vocabulary_audit=_merge_vocabulary_audits(vocabulary_audits),
            vocabulary_terms=tuple(
                item for item in vocabulary_terms if hasattr(item, "raw_term")
            ),
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

    @staticmethod
    def _append_unique(
        target: list[Any],
        seen: dict[Any, Any],
        key: Any,
        item: Any,
        warnings: list[str],
        label: str,
    ) -> bool:
        previous = seen.get(key)
        if previous is None:
            seen[key] = item
            target.append(item)
            return True
        if previous != item:
            warnings.append(f"Conflicting duplicate {label} retained from first source")
        return False


def _merge_segment_identity(
    previous: OperatingSegment | None,
    current: OperatingSegment,
) -> OperatingSegment:
    if previous is None or previous.name != previous.segment_id:
        return current
    if current.name == current.segment_id:
        return previous
    return previous.model_copy(update={"name": current.name})


def _merge_vocabulary_audits(audits: list[Any]) -> Any | None:
    if not audits:
        return None
    first = audits[0]
    return first.model_copy(
        update={
            "global_count": max(item.global_count for item in audits),
            "industry_count": max(item.industry_count for item in audits),
            "discovered_count": sum(item.discovered_count for item in audits),
            "rejected_count": sum(item.rejected_count for item in audits),
            "cache_status": (
                "hit" if any(item.cache_status == "hit" for item in audits) else "miss"
            ),
            "terms": tuple(
                dict.fromkeys(term for item in audits for term in item.terms)
            ),
            "diagnostics": tuple(
                dict.fromkeys(
                    diagnostic for item in audits for diagnostic in item.diagnostics
                )
            ),
            "raw_industry": next(
                (item.raw_industry for item in audits if item.raw_industry), ""
            ),
            "normalized_industry": next(
                (
                    item.normalized_industry
                    for item in audits
                    if item.normalized_industry
                ),
                "",
            ),
            "selected_archetype": next(
                (item.selected_archetype for item in audits if item.selected_archetype),
                "",
            ),
            "fallback_triggered": any(item.fallback_triggered for item in audits),
            "fallback_reason": next(
                (item.fallback_reason for item in audits if item.fallback_reason), ""
            ),
            "retry": any(item.retry for item in audits),
            "new_observations": sum(item.new_observations for item in audits),
            "validated_terms": tuple(
                dict.fromkeys(term for item in audits for term in item.validated_terms)
            ),
        }
    )


# Descriptive aliases for the two common naming conventions.
OperatingForecastDiscoveryService = OperatingEvidenceDiscoveryService
OperatingDriverDiscoveryService = OperatingEvidenceDiscoveryService
OperatingEvidenceDiscovery = OperatingEvidenceDiscoveryService


__all__ = [
    "OperatingDriverDiscoveryService",
    "OperatingEvidenceDiscovery",
    "OperatingEvidenceDiscoveryService",
    "OperatingForecastDiscovery",
    "OperatingForecastDiscoveryResult",
    "OperatingForecastDiscoveryService",
]
