"""Deterministic precedence-aware reconciliation of research evidence."""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Sequence
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgarito.services.research.contracts import (
    SOURCE_PRECEDENCE,
    CompetitorObservation,
    EstimateRange,
    EvidenceConfidence,
    EvidenceContext,
    EvidenceItem,
    EvidenceKind,
    EvidenceProvenance,
    EvidenceSourceType,
    MarketGrowthEvidence,
    MarketShareEvidence,
    MarketSizeEvidence,
    PricingObservation,
    ProductionCapacityEvidence,
    ResearchEvidence,
)


def source_priority(source_type: EvidenceSourceType) -> int:
    """Return the exact precedence rank; larger values govern consensus."""

    if isinstance(source_type, str):
        source_type = EvidenceSourceType(source_type.strip().lower())
    try:
        return len(SOURCE_PRECEDENCE) - SOURCE_PRECEDENCE.index(source_type)
    except ValueError as exc:
        raise ValueError(f"Unsupported evidence source type: {source_type!r}") from exc


class EvidenceContributor(BaseModel):
    """One preserved input, including inputs below the governing source tier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence: EvidenceItem
    governing: bool
    evidence_id: str | None = None
    source_date: datetime.date
    source_type: EvidenceSourceType
    low: Decimal
    base: Decimal
    high: Decimal
    provenance: EvidenceProvenance
    confidence: EvidenceConfidence
    unit: str
    context: EvidenceContext

    @model_validator(mode="before")
    @classmethod
    def populate_from_evidence(cls, value: object) -> object:
        if not isinstance(value, dict) or "evidence" not in value:
            return value
        evidence = value["evidence"]
        if not isinstance(evidence, ResearchEvidence):
            return value
        populated = dict(value)
        for name in (
            "evidence_id",
            "source_date",
            "source_type",
            "low",
            "base",
            "high",
            "provenance",
            "confidence",
            "unit",
            "context",
        ):
            populated.setdefault(name, getattr(evidence, name))
        return populated

    @model_validator(mode="after")
    def validate_copy(self) -> "EvidenceContributor":
        expected = self.evidence
        if (
            self.evidence_id != expected.evidence_id
            or self.source_date != expected.source_date
            or self.source_type != expected.source_type
            or self.low != expected.low
            or self.base != expected.base
            or self.high != expected.high
            or self.provenance != expected.provenance
            or self.confidence != expected.confidence
            or self.unit != expected.unit
            or self.context != expected.context
        ):
            raise ValueError("Evidence contributor fields must match evidence")
        return self

    @property
    def source(self) -> str:
        return self.evidence.source

    @property
    def estimate(self) -> EstimateRange:
        return EstimateRange(low=self.low, base=self.base, high=self.high)


class EvidenceConsensus(BaseModel):
    """Auditable consensus range and all of its preserved contributors.

    ``number_sources`` and ``sources`` describe every supplied observation.
    Only ``governing_source_type`` contributors determine the reconciled
    low/base/high values; lower-tier contributors remain in ``contributors``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EvidenceKind
    unit: str
    context: EvidenceContext
    low: Decimal
    base: Decimal
    high: Decimal
    number_sources: int = Field(ge=1)
    dispersion: Decimal = Field(ge=0)
    confidence: EvidenceConfidence
    sources: tuple[str, ...]
    contributors: tuple[EvidenceContributor, ...]
    governing_source_type: EvidenceSourceType
    governing_number_sources: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_result(self) -> "EvidenceConsensus":
        EstimateRange(low=self.low, base=self.base, high=self.high)
        if self.number_sources != len(self.contributors):
            raise ValueError("Consensus number_sources must match contributors")
        if self.number_sources != len(self.sources):
            raise ValueError("Consensus source list must include every contributor")
        if self.governing_number_sources > self.number_sources:
            raise ValueError("Governing source count cannot exceed number_sources")
        if any(not value.is_finite() for value in (self.dispersion,)):
            raise ValueError("Consensus dispersion must be finite")
        return self

    @property
    def estimate(self) -> EstimateRange:
        return EstimateRange(low=self.low, base=self.base, high=self.high)

    @property
    def source_list(self) -> tuple[str, ...]:
        """Alias emphasizing that source order is stable and auditable."""

        return self.sources

    @property
    def source_count(self) -> int:
        """Return the count of all preserved source observations."""

        return self.number_sources

    @property
    def governing_sources(self) -> tuple[EvidenceContributor, ...]:
        """Return only the contributors from the governing source tier."""

        return self.active_contributors

    @property
    def active_contributors(self) -> tuple[EvidenceContributor, ...]:
        return tuple(
            contributor for contributor in self.contributors if contributor.governing
        )

    @property
    def lower_priority_contributors(self) -> tuple[EvidenceContributor, ...]:
        return tuple(
            contributor
            for contributor in self.contributors
            if not contributor.governing
        )

    @property
    def source_provenance(self) -> tuple[EvidenceProvenance, ...]:
        return tuple(contributor.provenance for contributor in self.contributors)


def _median(values: Sequence[Decimal]) -> Decimal:
    if not values:
        raise ValueError("Cannot calculate a median for empty evidence")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)


def _sort_key(item: EvidenceItem) -> tuple[object, ...]:
    """Provide order independent of caller input order."""

    provenance = item.provenance
    return (
        -source_priority(item.source_type),
        provenance.source.casefold(),
        (provenance.source_id or "").casefold(),
        item.source_date.isoformat(),
        item.low,
        item.base,
        item.high,
        item.evidence_id or "",
        item.kind.value,
        item.model_dump_json(),
    )


def _context_signature(item: EvidenceItem) -> tuple[str, ...]:
    context = item.context
    return tuple(
        str(value) if value is not None else ""
        for value in (
            context.market,
            context.geography,
            context.segment,
            context.company,
            context.competitor,
            context.product,
            context.facility,
            context.period,
            context.metric,
            context.currency,
            context.scope,
            context.qualifier,
        )
    )


def _validate_comparable(items: Sequence[EvidenceItem]) -> None:
    first = items[0]
    if any(item.kind != first.kind for item in items[1:]):
        raise ValueError("Cannot reconcile different evidence kinds together")
    if any(item.unit.casefold() != first.unit.casefold() for item in items[1:]):
        raise ValueError("Cannot reconcile evidence with different units")
    signature = _context_signature(first)
    if any(_context_signature(item) != signature for item in items[1:]):
        raise ValueError("Cannot reconcile evidence with different contexts")


def reconcile_evidence(observations: Iterable[EvidenceItem]) -> EvidenceConsensus:
    """Reconcile observations by precedence, then coordinate-wise median.

    The highest available source tier governs all three result coordinates.
    Multiple observations at that tier are reconciled using the median of each
    coordinate, with an arithmetic midpoint for an even number of values.  All
    observations, including lower-tier inputs, are retained in deterministic
    contributor order.
    """

    items = tuple(observations)
    if not items:
        raise ValueError("Cannot reconcile empty evidence")
    if any(not isinstance(item, ResearchEvidence) for item in items):
        raise TypeError("Evidence reconciliation requires typed research evidence")
    ordered = tuple(sorted(items, key=_sort_key))
    _validate_comparable(ordered)

    highest_priority = max(source_priority(item.source_type) for item in ordered)
    governing_items = tuple(
        item
        for item in ordered
        if source_priority(item.source_type) == highest_priority
    )
    low = _median(tuple(item.low for item in governing_items))
    base = _median(tuple(item.base for item in governing_items))
    high = _median(tuple(item.high for item in governing_items))
    base_values = tuple(item.base for item in governing_items)
    dispersion = max(base_values) - min(base_values)
    confidence = min(
        (item.confidence for item in governing_items), key=lambda value: value.rank
    )

    contributors = tuple(
        EvidenceContributor(
            evidence=item,
            governing=source_priority(item.source_type) == highest_priority,
        )
        for item in ordered
    )
    return EvidenceConsensus(
        kind=ordered[0].kind,
        unit=ordered[0].unit,
        context=ordered[0].context,
        low=low,
        base=base,
        high=high,
        number_sources=len(contributors),
        dispersion=dispersion,
        confidence=confidence,
        sources=tuple(contributor.source for contributor in contributors),
        contributors=contributors,
        governing_source_type=ordered[0].source_type,
        governing_number_sources=len(governing_items),
    )


def _reconcile_typed(
    observations: Iterable[ResearchEvidence], expected_type: type[ResearchEvidence]
) -> EvidenceConsensus:
    values = tuple(observations)
    if not values:
        raise ValueError("Cannot reconcile empty evidence")
    if any(not isinstance(item, expected_type) for item in values):
        raise TypeError(f"Expected only {expected_type.__name__} evidence")
    return reconcile_evidence(values)  # type: ignore[arg-type]


def reconcile_market_size(
    observations: Iterable[MarketSizeEvidence],
) -> EvidenceConsensus:
    return _reconcile_typed(observations, MarketSizeEvidence)


reconcile_market_size_evidence = reconcile_market_size


def reconcile_market_growth(
    observations: Iterable[MarketGrowthEvidence],
) -> EvidenceConsensus:
    return _reconcile_typed(observations, MarketGrowthEvidence)


reconcile_market_growth_evidence = reconcile_market_growth


def reconcile_market_share(
    observations: Iterable[MarketShareEvidence],
) -> EvidenceConsensus:
    return _reconcile_typed(observations, MarketShareEvidence)


reconcile_market_share_evidence = reconcile_market_share


def reconcile_competitor_observations(
    observations: Iterable[CompetitorObservation],
) -> EvidenceConsensus:
    return _reconcile_typed(observations, CompetitorObservation)


reconcile_competitor_evidence = reconcile_competitor_observations


def reconcile_capacity_constraints(
    observations: Iterable[ProductionCapacityEvidence],
) -> EvidenceConsensus:
    return _reconcile_typed(observations, ProductionCapacityEvidence)


reconcile_production_capacity = reconcile_capacity_constraints


def reconcile_pricing_observations(
    observations: Iterable[PricingObservation],
) -> EvidenceConsensus:
    return _reconcile_typed(observations, PricingObservation)


reconcile_pricing_evidence = reconcile_pricing_observations


class EvidenceReconciler:
    """Small stateless facade for callers that prefer an object API."""

    @staticmethod
    def reconcile(observations: Iterable[EvidenceItem]) -> EvidenceConsensus:
        return reconcile_evidence(observations)


reconcile = reconcile_evidence
MarketSizeConsensus = EvidenceConsensus
MarketGrowthConsensus = EvidenceConsensus
MarketShareConsensus = EvidenceConsensus
CompetitorConsensus = EvidenceConsensus
CapacityConsensus = EvidenceConsensus
PricingConsensus = EvidenceConsensus
ConsensusResult = EvidenceConsensus
EvidenceReconciliationResult = EvidenceConsensus


__all__ = [
    "CapacityConsensus",
    "ConsensusResult",
    "CompetitorConsensus",
    "EvidenceConsensus",
    "EvidenceContributor",
    "EvidenceReconciliationResult",
    "EvidenceReconciler",
    "MarketGrowthConsensus",
    "MarketSizeConsensus",
    "MarketShareConsensus",
    "PricingConsensus",
    "source_priority",
    "reconcile",
    "reconcile_capacity_constraints",
    "reconcile_competitor_observations",
    "reconcile_competitor_evidence",
    "reconcile_evidence",
    "reconcile_market_growth",
    "reconcile_market_growth_evidence",
    "reconcile_market_share",
    "reconcile_market_share_evidence",
    "reconcile_market_size",
    "reconcile_market_size_evidence",
    "reconcile_pricing_evidence",
    "reconcile_pricing_observations",
    "reconcile_production_capacity",
]
