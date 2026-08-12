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
    ) -> None:
        self._edgar = edgar
        self._extractor = extractor
        self._selector = selector or GuidanceDocumentSelector()
        self.lookback_days = max(0, lookback_days)
        self.max_filings = max(0, max_filings)
        self.max_documents_per_filing = max(0, max_documents_per_filing)
        self.max_documents = max(0, max_documents)

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

        segments: list[OperatingSegment] = []
        definitions: list[OperatingDriverDefinition] = []
        observations: list[OperatingDriverObservation] = []
        programs: list[OperatingInvestmentProgram] = []
        management_constraints: list[OperatingDriverObservation] = []
        rejected: list[OperatingEvidenceRejection] = []
        audit_records: list[OperatingEvidenceAuditRecord] = []
        document_audits: list[OperatingDocumentAudit] = []
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
                        fiscal_years=fiscal_years,
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
