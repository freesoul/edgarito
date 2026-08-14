"""Contracts and mutable accumulation state for operating discovery."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from dataclasses import dataclass, field
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
from edgarito.schemas.operating_history import (
    OperatingEvidenceGap,
    OperatingHistoryAudit,
)


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
    raw_filings_received: int = 0
    raw_filings_in_range: int = 0
    candidate_filings: int = 0
    filing_inventory_cache_bypass: bool = False
    filing_inventory_fetched_live: bool = False
    filing_inventory_metadata: tuple[str, ...] = ()
    documents_inspected: int = 0
    vocabulary_audit: Any | None = None
    vocabulary_terms: tuple[Any, ...] = ()
    gaps_detected: tuple[OperatingEvidenceGap, ...] = ()
    gaps_resolved: tuple[OperatingEvidenceGap, ...] = ()
    gaps_unresolved: tuple[OperatingEvidenceGap, ...] = ()
    exhibits_found: int = 0
    gaps_resolved_sec: tuple[OperatingEvidenceGap, ...] = ()
    gaps_resolved_ir: tuple[OperatingEvidenceGap, ...] = ()
    ir_diagnostic: str | None = None

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
        """Return only contracts consumed by the forecast integration seam."""

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
            "gaps_detected": self.gaps_detected,
            "gaps_resolved": self.gaps_resolved,
            "gaps_unresolved": self.gaps_unresolved,
            "exhibits_found": self.exhibits_found,
            "gaps_resolved_sec": self.gaps_resolved_sec,
            "gaps_resolved_ir": self.gaps_resolved_ir,
            "ir_diagnostic": self.ir_diagnostic,
            "raw_filings_received": self.raw_filings_received,
            "raw_filings_in_range": self.raw_filings_in_range,
            "candidate_filings": self.candidate_filings,
            "filing_inventory_cache_bypass": self.filing_inventory_cache_bypass,
            "filing_inventory_fetched_live": self.filing_inventory_fetched_live,
            "filing_inventory_metadata": self.filing_inventory_metadata,
        }


class OperatingIrFallback(Protocol):
    async def retrieve(
        self, *, url: str, gaps: tuple[OperatingEvidenceGap, ...], as_of: datetime.date
    ) -> Any:
        """Return provider-neutral IR documents for requested gaps."""


class OperatingForecastDiscovery(Protocol):
    def discover(self, *args: Any, **kwargs: Any) -> Any:
        """Return normalized operating evidence for the integration seam."""


@dataclass
class DiscoveryState:
    """Mutable state passed between the document and recovery components."""

    segments: list[OperatingSegment] = field(default_factory=list)
    definitions: list[OperatingDriverDefinition] = field(default_factory=list)
    observations: list[OperatingDriverObservation] = field(default_factory=list)
    programs: list[OperatingInvestmentProgram] = field(default_factory=list)
    management_constraints: list[OperatingDriverObservation] = field(default_factory=list)
    rejected: list[OperatingEvidenceRejection] = field(default_factory=list)
    audit_records: list[OperatingEvidenceAuditRecord] = field(default_factory=list)
    document_audits: list[OperatingDocumentAudit] = field(default_factory=list)
    vocabulary_terms: list[Any] = field(default_factory=list)
    vocabulary_audits: list[Any] = field(default_factory=list)
    retry_documents: list[tuple[Any, Any, str]] = field(default_factory=list)
    candidate_documents: list[tuple[Any, Any, str]] = field(default_factory=list)
    discovered_by_document: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    unsupported: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unusable: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seen_segments: dict[str, OperatingSegment] = field(default_factory=dict)
    seen_definitions: dict[tuple[str, str], OperatingDriverDefinition] = field(
        default_factory=dict
    )
    seen_observations: dict[Any, OperatingDriverObservation] = field(
        default_factory=dict
    )
    seen_programs: dict[Any, OperatingInvestmentProgram] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    documents_inspected: int = 0
    filings_inspected: int = 0
    exhibits_found: int = 0
    raw_filings_received: int = 0
    raw_filings_in_range: int = 0
    candidate_filings: int = 0


__all__ = [
    "DiscoveryState",
    "OperatingForecastDiscovery",
    "OperatingForecastDiscoveryResult",
    "OperatingIrFallback",
]
