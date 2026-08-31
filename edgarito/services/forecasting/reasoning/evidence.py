"""Canonical evidence catalog construction for ForecastReasoner v1."""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from edgarito.services.forecasting.reasoning.contracts import (
    ForecastReasoningInput,
)
from edgarito.services.research.consensus import EvidenceConsensus

_VALID_EXISTING_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_DROP_KEYS = {
    "supporting_text",
    "source_text",
    "filing_text",
    "raw_text",
    "document_text",
    "content",
    "url",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value


def compact_structured(value: Any) -> Any:
    """Return deterministic structured data with source text removed."""

    value = _jsonable(value)
    if isinstance(value, dict):
        return {
            key: compact_structured(item)
            for key, item in sorted(value.items())
            if key.casefold() not in _DROP_KEYS
        }
    if isinstance(value, list):
        normalized = [compact_structured(item) for item in value]
        # Collections of structured records are sets for identity purposes;
        # scalar arrays (years and forecast paths) retain their semantic order.
        if normalized and all(isinstance(item, dict) for item in normalized):
            normalized.sort(
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return normalized
    if isinstance(value, str) and len(value) > 500:
        return value[:500]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        compact_structured(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _identity_structured(value: Any) -> Any:
    """Canonical identity form retaining provenance/source changes for hashes."""

    value = _jsonable(value)
    if isinstance(value, dict):
        return {key: _identity_structured(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        normalized = [_identity_structured(item) for item in value]
        if normalized and all(isinstance(item, dict) for item in normalized):
            normalized.sort(
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return normalized
    return value


def content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _identity_structured(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


class EvidenceCatalogProvenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_type: str | None = None
    source: str | None = None
    source_id: str | None = None
    publisher: str | None = None
    reference: str | None = None
    source_date: datetime.date | None = None
    payload_type: str
    payload_hash: str


class EvidenceCatalogItem(BaseModel):
    """One compact, citable evidence record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    category: str
    scope: str | None = None
    scope_id: str | None = None
    context: tuple[tuple[str, str], ...] = ()
    fiscal_year: int | None = None
    period: str | None = None
    metric: str | None = None
    driver_id: str | None = None
    unit: str | None = None
    value: Decimal | None = None
    low: Decimal | None = None
    base: Decimal | None = None
    high: Decimal | None = None
    dispersion: Decimal | None = None
    currency: str | None = None
    is_total: bool = False
    is_component: bool = False
    exhaustive: bool = False
    provenance: EvidenceCatalogProvenance
    payload_identity: str

    @field_validator("evidence_id", "category", "payload_identity")
    @classmethod
    def normalize_required(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Evidence catalog identifiers cannot be blank")
        return normalized

    @field_validator("context", mode="before")
    @classmethod
    def normalize_context(cls, value: Any) -> tuple[tuple[str, str], ...]:
        if isinstance(value, Mapping):
            values = value.items()
        else:
            values = value or ()
        return tuple(
            sorted(
                (str(key).strip(), str(item).strip())
                for key, item in values
                if str(key).strip() and str(item).strip()
            )
        )

    @field_validator("value", "low", "base", "high", "dispersion")
    @classmethod
    def validate_numbers(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and not value.is_finite():
            raise ValueError("Evidence catalog numbers must be finite")
        return value

    @property
    def id(self) -> str:
        return self.evidence_id

    @property
    def context_map(self) -> dict[str, str]:
        return dict(self.context)

    @property
    def source_date(self) -> datetime.date | None:
        return self.provenance.source_date

    @property
    def payload_type(self) -> str:
        return self.provenance.payload_type


class EvidenceCatalogExclusion(BaseModel):
    """Structured audit record for evidence withheld by the as-of boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    category: str
    reason: str
    payload_identity: str


class EvidenceCatalog(BaseModel):
    """Order-independent immutable catalog passed to the model and validator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[EvidenceCatalogItem, ...] = ()
    exclusions: tuple[EvidenceCatalogExclusion, ...] = ()
    duplicate_explicit_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_unique_ids(self) -> "EvidenceCatalog":
        ids = tuple(item.evidence_id for item in self.items)
        if len(ids) != len(set(ids)):
            raise ValueError("Evidence catalog IDs must be globally unique")
        return self

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.items)

    @property
    def bundle_hash(self) -> str:
        return content_hash(self.items)

    @property
    def identity(self) -> str:
        return self.bundle_hash

    @property
    def catalog_hash(self) -> str:
        return self.bundle_hash

    @property
    def records(self) -> tuple[EvidenceCatalogItem, ...]:
        return self.items

    def get(self, evidence_id: str) -> EvidenceCatalogItem | None:
        return next(
            (item for item in self.items if item.evidence_id == evidence_id), None
        )

    def exclusion(self, evidence_id: str) -> EvidenceCatalogExclusion | None:
        return next(
            (item for item in self.exclusions if item.evidence_id == evidence_id), None
        )

    def __contains__(self, evidence_id: object) -> bool:
        return any(item.evidence_id == evidence_id for item in self.items)

    def __iter__(self):
        return iter(self.items)

    def as_compact_json(self) -> str:
        return canonical_json(self.items)


_INPUT_COLLECTION_FIELDS = (
    "segments",
    "definitions",
    "observations",
    "management_guidance",
    "management_constraints",
    "investment_programs",
    "historical_facts",
    "research_evidence",
    "evidence_consensus",
    "manual_overrides",
    "manual_forward_driver_observations",
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _provenance(
    value: Any, *, payload_type: str, payload_hash: str
) -> EvidenceCatalogProvenance:
    provenance = _field(value, "provenance") or _field(value, "evidence")
    source_type = _field(value, "source_type")
    source = _field(provenance, "source") or _field(value, "source")
    source_id = (
        _field(provenance, "source_id")
        or _field(value, "source_id")
        or _field(value, "accession_number")
    )
    source_date = _source_date_for(value, provenance)
    return EvidenceCatalogProvenance(
        source_type=str(_enum_value(source_type)) if source_type is not None else None,
        source=str(source) if source is not None else None,
        source_id=str(source_id) if source_id is not None else None,
        publisher=_field(provenance, "publisher"),
        reference=_field(provenance, "reference") or _field(provenance, "locator"),
        source_date=source_date,
        payload_type=payload_type,
        payload_hash=payload_hash,
    )


def _source_date_for(value: Any, provenance: Any = None) -> datetime.date | None:
    provenance = provenance or _field(value, "provenance") or _field(value, "evidence")
    direct = (
        _field(value, "source_date")
        or _field(value, "filing_date")
        or _field(value, "filed")
        or _field(value, "observed_on")
        or _field(value, "as_of")
        or _field(provenance, "filing_date")
        or _field(provenance, "observed_on")
    )
    if direct is not None:
        return direct
    contributors = _field(value, "contributors") or ()
    dates = tuple(
        date
        for contributor in contributors
        if (
            date := _source_date_for(
                _field(contributor, "evidence"),
                _field(contributor, "provenance"),
            )
        )
        is not None
    )
    return max(dates) if dates else None


def _descriptor(
    value: Any,
    *,
    category: str,
    scope: str | None = None,
    scope_id: str | None = None,
    metric: str | None = None,
    driver_id: str | None = None,
) -> dict[str, Any]:
    payload_hash = content_hash(value)
    existing = _field(value, "evidence_id")
    if existing is not None:
        existing = str(existing).strip()
        if not _VALID_EXISTING_ID.fullmatch(existing):
            existing = None
    fiscal_year = _field(value, "fiscal_year")
    period = _field(value, "fiscal_period") or _field(value, "period")
    unit = _field(value, "unit")
    currency = _field(value, "currency")
    context: dict[str, str] = {}
    raw_context = _field(value, "context")
    if raw_context is not None:
        raw_context = _jsonable(raw_context)
        if isinstance(raw_context, Mapping):
            context.update(
                {
                    str(key): str(item)
                    for key, item in raw_context.items()
                    if item is not None and str(item).strip()
                }
            )
    for name in (
        "market",
        "geography",
        "segment",
        "segment_name",
        "company",
        "product",
        "facility",
        "scope",
    ):
        item = _field(value, name)
        if item is not None:
            context.setdefault(name, str(_enum_value(item)))
    for name in ("archetype", "formula_id"):
        item = _field(value, name)
        if item is not None:
            context.setdefault(name, str(_enum_value(item)))
    for name in ("input_metrics", "required_inputs", "optional_inputs"):
        item = _field(value, name)
        if item is not None:
            context.setdefault(
                name, ",".join(str(_enum_value(entry)) for entry in item)
            )
    if scope is not None:
        context.setdefault("scope", scope)
    if scope_id is not None:
        context.setdefault("scope_id", scope_id)
    if metric is None:
        metric = _field(value, "metric")
    if driver_id is None:
        driver_id = _field(value, "driver_id")
    if metric is not None:
        metric = str(_enum_value(metric))
    if driver_id is not None:
        driver_id = str(_enum_value(driver_id))
    if scope is None:
        scope = _field(value, "scope")
    if scope is not None:
        scope = str(_enum_value(scope))
    if scope_id is None:
        scope_id = (
            _field(value, "segment_id")
            or _field(value, "segment_name")
            or _field(value, "company_id")
        )
    if scope_id is not None:
        scope_id = str(_enum_value(scope_id))
    return {
        "category": category,
        "scope": scope,
        "scope_id": scope_id,
        "context": context,
        "fiscal_year": int(fiscal_year) if fiscal_year is not None else None,
        "period": str(_enum_value(period)) if period is not None else None,
        "metric": metric,
        "driver_id": driver_id,
        "unit": str(_enum_value(unit)) if unit is not None else None,
        "currency": str(_enum_value(currency)).upper()
        if currency is not None
        else None,
        "value": _field(value, "value"),
        "low": _field(value, "low"),
        "base": _field(value, "base"),
        "high": _field(value, "high"),
        "dispersion": _field(value, "dispersion"),
        "is_total": bool(_field(value, "is_total", False)),
        "is_component": bool(_field(value, "is_component", False)),
        "exhaustive": bool(_field(value, "exhaustive", False)),
        "payload_hash": payload_hash,
        "payload_type": type(value).__name__,
        "existing_id": existing,
        "raw": value,
    }


def _consensus_descriptor(value: EvidenceConsensus) -> dict[str, Any]:
    descriptor = _descriptor(value, category="MARKET")
    descriptor["metric"] = descriptor["metric"] or str(_enum_value(value.kind))
    return descriptor


def _input_descriptors(value: ForecastReasoningInput) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    result.extend(
        _descriptor(item, category="COMP", scope="segment", scope_id=item.segment_id)
        for item in value.segments
    )
    result.extend(
        _descriptor(
            item,
            category="OP",
            scope="segment",
            scope_id=item.segment_id,
            driver_id=item.driver_id,
        )
        for item in value.definitions
    )
    result.extend(
        _descriptor(
            item,
            category="OP",
            scope=_field(item, "scope"),
            scope_id=item.segment_id,
            driver_id=item.driver_id,
        )
        for item in value.observations
    )
    result.extend(
        _descriptor(
            item,
            category="HIST",
            scope=item.scope.value,
            scope_id=item.scope_id,
            metric=item.metric,
        )
        for item in value.historical_facts
    )
    result.extend(
        _descriptor(
            item,
            category="MGMT",
            scope=_field(item, "scope"),
            scope_id=_field(item, "segment_id"),
            metric=_field(item, "metric") or _field(item, "driver_id"),
        )
        for item in value.management_guidance
    )
    result.extend(
        _descriptor(
            item,
            category="MGMT",
            scope=_field(item, "scope"),
            scope_id=_field(item, "segment_id"),
            metric=_field(item, "metric") or _field(item, "driver_id"),
        )
        for item in value.management_constraints
    )
    result.extend(
        _descriptor(
            item,
            category="MGMT",
            scope="segment" if item.segment_id else "company",
            scope_id=item.segment_id or "company",
        )
        for item in value.investment_programs
    )
    result.extend(
        _descriptor(item, category="MARKET") for item in value.research_evidence
    )
    result.extend(_consensus_descriptor(item) for item in value.evidence_consensus)
    # Manual inputs are evidence for identity and precedence, but they are
    # deliberately categorized separately so a model cannot call them
    # first-party facts.
    result.extend(
        _descriptor(
            item,
            category="MANUAL",
            scope=_field(item, "scope"),
            scope_id=item.segment_id,
            driver_id=item.driver_id,
        )
        for item in value.manual_forward_driver_observations
    )
    result.extend(
        _descriptor(
            item,
            category="MANUAL",
            scope=_field(item, "scope"),
            scope_id=_field(item, "scope_id"),
            metric=_field(item, "metric"),
        )
        for item in value.manual_overrides
    )
    return result


def build_evidence_catalog(value: ForecastReasoningInput | Any) -> EvidenceCatalog:
    """Create stable IDs from content, independent of input collection order."""

    input_value = (
        value
        if isinstance(value, ForecastReasoningInput)
        else ForecastReasoningInput.model_validate(value)
    )

    descriptors = _input_descriptors(input_value)
    records: dict[tuple[str, str], dict[str, Any]] = {}
    assigned_ids: set[str] = set()
    explicit_ids = {item["existing_id"] for item in descriptors if item["existing_id"]}
    explicit_counts = Counter(
        item["existing_id"] for item in descriptors if item["existing_id"]
    )
    exclusions: list[EvidenceCatalogExclusion] = []
    ordered_descriptors = sorted(
        descriptors,
        key=lambda item: (
            item["category"],
            item["payload_hash"],
            item["existing_id"] or "",
        ),
    )
    for item in ordered_descriptors:
        payload_hash = item["payload_hash"]
        existing = (
            item["existing_id"]
            if item["existing_id"] and explicit_counts[item["existing_id"]] == 1
            else None
        )
        evidence_id = existing or f"{item['category']}-{payload_hash[:16]}"
        if evidence_id in assigned_ids or (
            not existing and evidence_id in explicit_ids
        ):
            evidence_id = f"{item['category']}-{payload_hash[:32]}"
            suffix = 1
            while evidence_id in assigned_ids or evidence_id in explicit_ids:
                evidence_id = f"{item['category']}-{payload_hash[:32]}-{suffix}"
                suffix += 1
        assigned_ids.add(evidence_id)
        identity_key = (evidence_id, payload_hash)
        records.setdefault(identity_key, {**item, "evidence_id": evidence_id})
        source_date = _source_date_for(item["raw"])
        if source_date is not None and source_date > input_value.as_of:
            records.pop(identity_key, None)
            exclusions.append(
                EvidenceCatalogExclusion(
                    evidence_id=evidence_id,
                    category=item["category"],
                    reason=f"unavailable after as_of={input_value.as_of.isoformat()}",
                    payload_identity=payload_hash,
                )
            )

    catalog_items: list[EvidenceCatalogItem] = []
    for item in records.values():
        raw = item.pop("raw")
        payload_hash = item.pop("payload_hash")
        payload_type = item.pop("payload_type")
        item.pop("existing_id", None)
        catalog_items.append(
            EvidenceCatalogItem(
                **item,
                provenance=_provenance(
                    raw, payload_type=payload_type, payload_hash=payload_hash
                ),
                payload_identity=payload_hash,
            )
        )
    return EvidenceCatalog(
        items=tuple(
            sorted(
                catalog_items,
                key=lambda item: (item.evidence_id, item.payload_identity),
            )
        ),
        exclusions=tuple(
            sorted(
                exclusions, key=lambda item: (item.evidence_id, item.payload_identity)
            )
        ),
        duplicate_explicit_ids=tuple(
            sorted(item for item, count in explicit_counts.items() if count > 1)
        ),
    )


def compact_reasoning_input(value: ForecastReasoningInput) -> dict[str, Any]:
    """Serialize only as-of-available structured input for the model request."""

    result = compact_structured(value)
    for field_name in _INPUT_COLLECTION_FIELDS:
        records = getattr(value, field_name, ())
        result[field_name] = [
            compact_structured(item)
            for item in records
            if (
                (source_date := _source_date_for(item)) is None
                or source_date <= value.as_of
            )
        ]
    return result


def research_hash(value: ForecastReasoningInput) -> str:
    values = tuple(
        sorted(
            content_hash(item)
            for item in (*value.research_evidence, *value.evidence_consensus)
            if (source_date := _source_date_for(item)) is None
            or source_date <= value.as_of
        )
    )
    return content_hash(values)


def manual_inputs_hash(value: ForecastReasoningInput) -> str:
    values = tuple(
        sorted(
            content_hash(item)
            for item in (
                *value.manual_overrides,
                *value.manual_forward_driver_observations,
            )
            if (source_date := _source_date_for(item)) is None
            or source_date <= value.as_of
        )
    )
    return content_hash(values)


EvidenceCatalogBuilder = type(
    "EvidenceCatalogBuilder",
    (),
    {"build": staticmethod(build_evidence_catalog)},
)
ForecastReasoningEvidenceCatalog = EvidenceCatalog
EvidenceRecord = EvidenceCatalogItem
build_evidence_bundle = build_evidence_catalog


__all__ = [
    "EvidenceCatalogProvenance",
    "EvidenceCatalogItem",
    "EvidenceCatalogExclusion",
    "EvidenceCatalog",
    "build_evidence_catalog",
    "canonical_json",
    "compact_structured",
    "content_hash",
    "research_hash",
    "manual_inputs_hash",
    "EvidenceCatalogBuilder",
    "ForecastReasoningEvidenceCatalog",
    "EvidenceRecord",
    "build_evidence_bundle",
    "compact_reasoning_input",
]
