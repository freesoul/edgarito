"""Conservative adapters for typed market-intelligence evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from edgarito.services.research.consensus import EvidenceConsensus
from edgarito.services.research.contracts import (
    EvidenceContext,
    EvidenceKind,
    ResearchEvidence,
)
from edgarito.services.valuation.factors.contracts import (
    FactorConfidence,
    FactorDomain,
    FactorEvidence,
    FactorKey,
    FactorPeriod,
    FactorPeriodType,
    FactorProvenance,
)
from edgarito.services.valuation.factors.identity import (
    canonicalize_geography,
    stable_digest,
)

_PERIOD_YEAR = re.compile(r"^(?:FY\s*)?(\d{4})$", re.IGNORECASE)
_QUARTER = re.compile(
    r"^(?:FY\s*)?(\d{4})\s*[-/]?\s*Q([1-4])$|^Q([1-4])\s*(\d{4})$",
    re.IGNORECASE,
)


def _required(value: str | None, label: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"research evidence requires an unambiguous {label}")
    normalized = str(value).strip()
    if normalized.casefold() in {"unknown", "unspecified", "n/a", "na", "none"}:
        raise ValueError(f"research evidence requires an unambiguous {label}")
    return normalized


def _period(value: str | None) -> FactorPeriod:
    raw = _required(value, "period")
    token = raw.strip().casefold().replace("-", "_").replace(" ", "_")
    if token in {"current_spot", "spot"}:
        return FactorPeriod(period_type=FactorPeriodType.CURRENT_SPOT, period_key="current_spot")
    if token in {"long_term", "longterm"}:
        return FactorPeriod(period_type=FactorPeriodType.LONG_TERM, period_key="long_term")
    match = _QUARTER.match(raw)
    if match:
        year = match.group(1) or match.group(4)
        quarter = match.group(2) or match.group(3)
        return FactorPeriod(
            target_year=int(year),
            period_type=FactorPeriodType.FQ,
            period_key=f"Q{quarter} {year}",
        )
    match = _PERIOD_YEAR.match(raw)
    if match:
        year = int(match.group(1))
        return FactorPeriod(
            target_year=year,
            period_type=FactorPeriodType.FY,
            period_key=f"FY {year}",
        )
    raise ValueError(f"research evidence period is not an exact FY/FQ/spot period: {raw!r}")


def _confidence(value: Any) -> FactorConfidence:
    rank = getattr(value, "rank", None)
    if rank is None:
        rank = {"low": 1, "medium": 2, "high": 3}.get(
            str(getattr(value, "value", value)).casefold(), 1
        )
    return {
        1: FactorConfidence.LOW,
        2: FactorConfidence.MEDIUM,
        3: FactorConfidence.HIGH,
    }.get(rank, FactorConfidence.UNKNOWN)


def _source_name(provenance: Any) -> str:
    return _required(getattr(provenance, "source", provenance), "source")


def _source_type(value: Any) -> str:
    return str(getattr(value, "value", value)).strip().casefold()


def _context_key(
    *,
    kind: EvidenceKind,
    context: EvidenceContext,
    unit: str,
    currency: str | None,
    target: FactorKey | None,
) -> FactorKey:
    market = context.market
    geography = (
        canonicalize_geography(context.geography) if context.geography is not None else None
    )
    segment = context.segment
    product = context.product
    for value, label in (
        (geography, "geography"),
        (segment, "segment"),
        (product, "product"),
        (context.scope, "scope"),
    ):
        if value is not None and str(value).strip().casefold() in {
            "unknown",
            "unspecified",
            "n/a",
            "na",
            "none",
        }:
            raise ValueError(f"research evidence requires an unambiguous {label}")
    unit = _required(unit, "unit")
    if currency is not None:
        currency = _required(currency, "currency")

    if kind in {EvidenceKind.MARKET_SIZE, EvidenceKind.MARKET_GROWTH}:
        domain = FactorDomain.MARKET
        subject_type = "market"
        subject_id = _required(market, "market")
    elif kind is EvidenceKind.MARKET_SHARE:
        domain = FactorDomain.MARKET
        subject_type = "company"
        subject_id = _required(context.company, "company")
        market = _required(market, "market")
    elif kind is EvidenceKind.PRICING:
        domain = FactorDomain.MARKET
        subject_type = "product"
        subject_id = _required(product, "product")
    elif kind is EvidenceKind.PRODUCTION_CAPACITY:
        domain = FactorDomain.OPERATING
        subject_type = "company"
        subject_id = _required(context.company, "company")
    elif kind is EvidenceKind.COMPETITOR:
        domain = FactorDomain.COMPETITOR
        subject_type = "competitor"
        subject_id = _required(context.competitor, "competitor")
    else:  # pragma: no cover - protected by EvidenceKind
        raise ValueError(f"unsupported research evidence kind: {kind!r}")

    candidate = FactorKey(
        domain=domain,
        subject_type=subject_type,
        subject_id=subject_id,
        metric=context.metric or kind.value,
        geography=geography,
        market=market,
        product=product,
        business=segment,
        period=_period(context.period),
        unit=unit,
        currency=currency,
        # EvidenceContext uses ``scope`` for category-specific dimensions such
        # as TAM or observed-price scope.  Keep that dimension in FactorKey
        # rather than silently discarding it.
        qualifier=context.qualifier or context.scope,
    )
    if target is None:
        return candidate
    # Explicit requests own identity.  Optional target dimensions omitted by a
    # caller are treated as unknown, while supplied dimensions must match
    # exactly; period, unit, currency, and the subject coordinates are always
    # checked because they cannot be safely inferred away.
    for name in (
        "domain",
        "subject_type",
        "subject_id",
        "metric",
        "period",
        "unit",
        "currency",
        "geography",
        "market",
        "product",
        "business",
        "qualifier",
    ):
        expected = getattr(target, name)
        if expected is not None and expected != getattr(candidate, name):
            raise ValueError(
                "research evidence coordinates are not exactly compatible with target FactorKey"
            )
    return target


def _factor_provenance(provenance: Any, *, notes: str | None = None) -> FactorProvenance:
    return FactorProvenance(
        source=_source_name(provenance),
        source_id=getattr(provenance, "source_id", None),
        publisher=getattr(provenance, "publisher", None),
        reference=getattr(provenance, "reference", None),
        locator=getattr(provenance, "locator", None),
        url=getattr(provenance, "url", None),
        notes=notes or getattr(provenance, "notes", None),
    )


def _range_values(item: Any) -> tuple[Any, Any, Any]:
    base = item.base
    return item.low, base, item.high


class ResearchFactorAdapter:
    """Convert compatible research observations or consensus to raw evidence."""

    def to_factor_evidence(
        self,
        item: ResearchEvidence | EvidenceConsensus,
        *,
        target_key: FactorKey | None = None,
        target: FactorKey | None = None,
    ) -> FactorEvidence:
        target_key = target_key or target
        if target_key is not None and not isinstance(target_key, FactorKey):
            target_key = FactorKey.model_validate(target_key)
        if isinstance(item, EvidenceConsensus):
            return self._consensus(item, target_key=target_key)
        if not isinstance(item, ResearchEvidence):
            raise TypeError("research adapter accepts ResearchEvidence or EvidenceConsensus")
        return self._observation(item, target_key=target_key)

    adapt = to_factor_evidence

    def _observation(
        self, item: ResearchEvidence, *, target_key: FactorKey | None
    ) -> FactorEvidence:
        key = _context_key(
            kind=item.kind,
            context=item.context,
            unit=item.unit,
            currency=item.currency,
            target=target_key,
        )
        low, base, high = _range_values(item)
        evidence_id = item.evidence_id or stable_digest(item.model_dump(mode="python"))
        context_notes = ";".join(
            f"{name}={value}"
            for name, value in (
                ("scope", item.context.scope),
                ("segment", item.context.segment),
                ("company", item.context.company),
                ("competitor", item.context.competitor),
                ("facility", item.context.facility),
            )
            if value is not None
        )
        provenance = _factor_provenance(
            item.provenance,
            notes=";".join(
                value
                for value in (getattr(item.provenance, "notes", None), context_notes)
                if value
            )
            or None,
        )
        source_type = _source_type(item.source_type)
        return FactorEvidence(
            key=key,
            low=low,
            base=base,
            high=high,
            information_available_on=item.source_date,
            observed_on=item.source_date,
            source=item.source,
            evidence_id=evidence_id,
            all_availability_dates=(item.source_date,),
            provenance=provenance,
            confidence=_confidence(item.confidence),
            source_type=source_type,
            source_types=(source_type,),
            evidence_refs=(evidence_id,),
            warnings=(item.notes,) if item.notes else (),
        )

    def _consensus(
        self, item: EvidenceConsensus, *, target_key: FactorKey | None
    ) -> FactorEvidence:
        key = _context_key(
            kind=item.kind,
            context=item.context,
            unit=item.unit,
            currency=item.context.currency,
            target=target_key,
        )
        if not item.contributors:
            raise ValueError("research consensus requires contributors")
        dates = tuple(contributor.source_date for contributor in item.contributors)
        available_on = max(dates)
        raw_refs = tuple(
            contributor.evidence_id
            or stable_digest(contributor.evidence.model_dump(mode="python"))
            for contributor in item.contributors
        )
        refs_list: list[str] = []
        for index, ref in enumerate(raw_refs):
            candidate_ref = ref
            if candidate_ref in refs_list:
                candidate_ref = stable_digest({"ref": ref, "contributor": index})
            refs_list.append(candidate_ref)
        refs = tuple(refs_list)
        consensus_id = stable_digest(
            {
                "key": key.canonical,
                "range": (item.low, item.base, item.high),
                "refs": refs,
            }
        )
        source_types = tuple(
            dict.fromkeys(_source_type(contributor.source_type) for contributor in item.contributors)
        )
        provenance = FactorProvenance(
            source="research_consensus",
            source_id=consensus_id,
            publisher=_source_type(item.governing_source_type),
            reference=",".join(item.sources),
            notes=(
                f"governing_source_type={_source_type(item.governing_source_type)};"
                f"dispersion={item.dispersion};"
                f"source_dates={','.join(date.isoformat() for date in dates)}"
            ),
        )
        return FactorEvidence(
            key=key,
            low=item.low,
            base=item.base,
            high=item.high,
            information_available_on=available_on,
            observed_on=available_on,
            source="research_consensus",
            evidence_id=consensus_id,
            all_availability_dates=(available_on,),
            provenance=provenance,
            confidence=_confidence(item.confidence),
            source_type=_source_type(item.governing_source_type),
            source_types=source_types,
            evidence_refs=refs,
            dispersion=item.dispersion,
        )

    def to_factor_evidence_many(
        self,
        items: Iterable[ResearchEvidence | EvidenceConsensus],
        *,
        target_key: FactorKey | None = None,
    ) -> tuple[FactorEvidence, ...]:
        values = [self.to_factor_evidence(item, target_key=target_key) for item in items]
        unique = {item.fingerprint: item for item in values}
        return tuple(sorted(unique.values(), key=lambda item: (item.key.digest, item.fingerprint)))

    adapt_many = to_factor_evidence_many


# Names used by callers that want to make the boundary explicit.
ResearchEvidenceFactorAdapter = ResearchFactorAdapter
ExistingResearchEvidenceAdapter = ResearchFactorAdapter
ResearchEvidenceAdapter = ResearchFactorAdapter
ExistingResearchFactorAdapter = ResearchFactorAdapter


__all__ = [
    "ExistingResearchEvidenceAdapter",
    "ExistingResearchFactorAdapter",
    "ResearchEvidenceAdapter",
    "ResearchEvidenceFactorAdapter",
    "ResearchFactorAdapter",
]
