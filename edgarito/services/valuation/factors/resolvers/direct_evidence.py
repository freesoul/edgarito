"""Deterministic point-in-time resolution from supplied evidence."""

from __future__ import annotations

import datetime as dt
from typing import Iterable

from edgarito.services.valuation.factors.contracts import (
    FactorEstimate,
    FactorEvidence,
    FactorRequest,
)
from edgarito.services.valuation.factors.resolvers.base import ResolverResult


def _point(value: dt.date | dt.datetime) -> dt.datetime:
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            return value.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return value
    return dt.datetime.combine(value, dt.time.min)


class DirectEvidenceResolver:
    resolver_id = "direct_evidence"
    priority = 0
    requires_dependencies = False

    def __init__(self, evidence: Iterable[FactorEvidence] = ()):
        self.evidence = tuple(evidence)

    def eligible_evidence(self, request: FactorRequest) -> tuple[FactorEvidence, ...]:
        candidates = [
            item
            for item in self.evidence
            if item.key == request.key
            and not item.superseded
            and item.confidence.rank >= request.min_confidence.rank
            and all(
                _point(available_on) <= _point(request.information_as_of)
                for available_on in item.all_availability_dates
            )
        ]
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    -item.confidence.rank,
                    -_point(item.information_available_on).timestamp(),
                    item.source,
                    item.evidence_id or item.fingerprint,
                ),
            )
        )

    def can_resolve(self, request: FactorRequest, context=None) -> bool:
        return bool(self.eligible_evidence(request))

    def resolve(self, request: FactorRequest, context=None, **kwargs) -> ResolverResult:
        candidates = self.eligible_evidence(request)
        if not candidates:
            return ResolverResult.missing("no eligible exact evidence")
        selected = candidates[0]
        evaluated_at = (
            getattr(context, "evaluated_at", None)
            or (context.get("evaluated_at") if isinstance(context, dict) else None)
            or request.information_as_of
        )
        ref = selected.evidence_id or selected.fingerprint
        estimate = FactorEstimate(
            key=request.key,
            range=selected.range,
            unit=request.key.unit,
            currency=request.key.currency,
            info_as_of=selected.information_available_on,
            target_period=request.key.period,
            confidence=selected.confidence,
            methodology="direct evidence",
            resolver=self.resolver_id,
            evidence_refs=(ref,),
            all_availability_dates=selected.all_availability_dates,
            created_at=evaluated_at,
            immutable=selected.immutable,
            source=selected.source,
            version=selected.version,
            warnings=selected.warnings,
        )
        return ResolverResult.success(estimate)


__all__ = ["DirectEvidenceResolver"]
