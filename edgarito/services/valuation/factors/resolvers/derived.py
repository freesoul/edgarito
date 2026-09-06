"""Conservative deterministic interval arithmetic for derived factors."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping, Optional

from edgarito.services.valuation.factors.contracts import (
    FactorEstimate,
    FactorRange,
    FactorRequest,
)
from edgarito.services.valuation.factors.decomposers.base import FactorDecomposition
from edgarito.services.valuation.factors.resolvers.base import ResolverResult


class DerivedOperation(str, Enum):
    ADD = "ADD"
    SUBTRACT = "SUBTRACT"
    WEIGHTED_SUM = "WEIGHTED_SUM"
    IDENTITY = "IDENTITY"


def _point(value: dt.date | dt.datetime) -> dt.datetime:
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            return value.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return value
    return dt.datetime.combine(value, dt.time.min)


def _child_estimate(children: Mapping[Any, FactorEstimate], key):
    for candidate in (key, key.digest, key.semantic_id):
        try:
            if candidate in children:
                return children[candidate]
        except TypeError:
            continue
    return None


class DerivedFactorResolver:
    resolver_id = "derived"
    requires_dependencies = True

    def can_resolve(self, request, context=None) -> bool:
        return bool(getattr(context, "decomposition", None)) if context else False

    def resolve(
        self,
        request: FactorRequest,
        context=None,
        *,
        children: Optional[Mapping[Any, FactorEstimate]] = None,
        decomposition: Optional[FactorDecomposition] = None,
        operation: Optional[str] = None,
        **kwargs,
    ) -> ResolverResult:
        if isinstance(context, Mapping) and children is None:
            children = context
            context = None
        elif isinstance(context, FactorDecomposition) and decomposition is None:
            decomposition = context
            context = None
        decomposition = decomposition or getattr(context, "decomposition", None)
        children = children or getattr(context, "children", None) or {}
        if decomposition is None:
            return ResolverResult.missing("derived resolver requires a decomposition")
        required = [item for item in decomposition.proposals if item.required]
        resolved = []
        for proposal in required:
            estimate = _child_estimate(children, proposal.child_key)
            if estimate is None:
                return ResolverResult.missing(
                    f"missing required dependency {proposal.child_key.digest}"
                )
            resolved.append((proposal, estimate))
        if not resolved:
            return ResolverResult.missing("derived factor has no dependencies")

        op = (operation or decomposition.operation or "IDENTITY").upper()
        compatibility_error = self._compatibility_error(request, resolved)
        if compatibility_error is not None:
            return ResolverResult.missing(compatibility_error)
        ranges = [estimate.range for _, estimate in resolved]
        try:
            value_range = self._calculate(op, ranges, [proposal.weight for proposal, _ in resolved])
        except (ValueError, KeyError) as exc:
            return ResolverResult.missing(str(exc))

        availability = []
        evidence_refs = []
        dependencies = []
        fingerprints = {}
        confidence = min(
            (estimate.confidence for _, estimate in resolved),
            key=lambda item: item.rank,
        )
        for _, estimate in resolved:
            dependencies.append(estimate.key)
            fingerprints[estimate.key.digest] = estimate.fingerprint
            for available_on in estimate.all_availability_dates:
                if available_on not in availability:
                    availability.append(available_on)
            for ref in estimate.evidence_refs:
                if ref not in evidence_refs:
                    evidence_refs.append(ref)
        availability.sort(key=_point)
        evaluated_at = getattr(context, "evaluated_at", None) or request.information_as_of
        info_as_of = max(
            (estimate.info_as_of for _, estimate in resolved), key=_point
        )
        estimate = FactorEstimate(
            key=request.key,
            range=value_range,
            unit=request.key.unit,
            currency=request.key.currency,
            info_as_of=info_as_of,
            target_period=request.key.period,
            confidence=confidence,
            methodology=f"derived {op.lower()}",
            resolver=self.resolver_id,
            evidence_refs=tuple(sorted(evidence_refs)),
            dependencies=tuple(dependencies),
            dependency_fingerprints=fingerprints,
            all_availability_dates=tuple(availability),
            created_at=evaluated_at,
            source=None,
        )
        return ResolverResult.success(estimate)

    @staticmethod
    def _compatibility_error(request, resolved) -> str | None:
        """Reject arithmetic without an explicit v1 conversion relationship."""

        expected_unit = request.key.unit
        expected_currency = request.key.currency
        expected_period = request.key.period
        for proposal, estimate in resolved:
            child_key = estimate.key
            if child_key != proposal.child_key:
                return (
                    f"dependency {proposal.child_key.digest} returned estimate for "
                    f"{child_key.digest}"
                )
            if child_key.unit != expected_unit or estimate.unit != expected_unit:
                return (
                    f"dependency {child_key.digest} unit {child_key.unit!r} "
                    f"does not match parent unit {expected_unit!r}"
                )
            if (
                child_key.currency != expected_currency
                or estimate.currency != expected_currency
            ):
                return (
                    f"dependency {child_key.digest} currency {child_key.currency!r} "
                    f"does not match parent currency {expected_currency!r}"
                )
            if (
                child_key.period != expected_period
                or estimate.target_period != expected_period
            ):
                return (
                    f"dependency {child_key.digest} target period does not match "
                    "parent target period"
                )
        return None

    @staticmethod
    def _calculate(operation: str, ranges, weights) -> FactorRange:
        if operation == DerivedOperation.IDENTITY.value:
            if len(ranges) != 1:
                raise ValueError("IDENTITY requires exactly one dependency")
            return ranges[0]
        if operation == DerivedOperation.ADD.value:
            if not ranges:
                raise ValueError("ADD requires dependencies")
            return FactorRange(
                low=sum((item.low for item in ranges), Decimal("0")),
                base=sum((item.base for item in ranges), Decimal("0")),
                high=sum((item.high for item in ranges), Decimal("0")),
            )
        if operation == DerivedOperation.SUBTRACT.value:
            if len(ranges) < 2:
                raise ValueError("SUBTRACT requires at least two dependencies")
            return FactorRange(
                low=ranges[0].low - sum((item.high for item in ranges[1:]), Decimal("0")),
                base=ranges[0].base - sum((item.base for item in ranges[1:]), Decimal("0")),
                high=ranges[0].high - sum((item.low for item in ranges[1:]), Decimal("0")),
            )
        if operation == DerivedOperation.WEIGHTED_SUM.value:
            if not ranges or len(ranges) != len(weights):
                raise ValueError("WEIGHTED_SUM requires one weight per dependency")
            lows = []
            bases = []
            highs = []
            for item, raw_weight in zip(ranges, weights, strict=True):
                weight = Decimal(str(raw_weight))
                lows.append(min(weight * item.low, weight * item.high))
                highs.append(max(weight * item.low, weight * item.high))
                bases.append(weight * item.base)
            return FactorRange(
                low=sum(lows, Decimal("0")),
                base=sum(bases, Decimal("0")),
                high=sum(highs, Decimal("0")),
            )
        raise ValueError(f"unsupported derived operation {operation}")


__all__ = ["DerivedFactorResolver", "DerivedOperation"]
