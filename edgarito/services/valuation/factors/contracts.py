"""Frozen, provider-neutral contracts for semantic valuation factors."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from edgarito.services.valuation.factors.identity import (
    canonical_json,
    canonicalize_currency,
    canonicalize_geography,
    canonicalize_token,
    canonicalize_unit,
    stable_digest,
)


class _FactorEnum(str, Enum):
    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            token = value.strip().casefold().replace("-", "_").replace(" ", "_")
            for member in cls:
                if (
                    member.name.casefold() == token
                    or str(member.value).casefold() == token
                ):
                    return member
        return None


class FactorDomain(_FactorEnum):
    COMPANY = "company"
    BUSINESS = "business"
    OPERATING = "operating"
    MARKET = "market"
    COMPETITOR = "competitor"
    COMMODITY = "commodity"
    MACRO = "macro"
    REGULATORY = "regulatory"
    GEOPOLITICAL = "geopolitical"
    FINANCING = "financing"


class FactorPeriodType(_FactorEnum):
    FY = "FY"
    FQ = "FQ"
    CURRENT_SPOT = "current_spot"
    LONG_TERM = "long_term"


class FactorConfidence(_FactorEnum):
    UNKNOWN = "unknown"
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

    @property
    def rank(self) -> int:
        return {
            FactorConfidence.UNKNOWN: -1,
            FactorConfidence.VERY_LOW: 0,
            FactorConfidence.LOW: 1,
            FactorConfidence.MEDIUM: 2,
            FactorConfidence.HIGH: 3,
            FactorConfidence.VERY_HIGH: 4,
        }[self]


class FactorMateriality(_FactorEnum):
    UNKNOWN = "unknown"
    IMMATERIAL = "immaterial"
    MATERIAL = "material"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FactorPriority(_FactorEnum):
    LOW = "low"
    MEDIUM = "normal"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class FactorResolutionStatus(_FactorEnum):
    CACHE_HIT = "cache_hit"
    DIRECTLY_RESOLVED = "directly_resolved"
    DERIVED = "derived"
    STOPPED = "stopped"
    FAILED = "failed"
    UNRESOLVED = "unresolved"


class CacheState(_FactorEnum):
    HIT = "hit"
    MISS = "miss"
    STALE = "stale"

    # Names useful to callers that want to mirror the resolution status.
    CACHE_HIT = "hit"
    CACHE_MISS = "miss"


class StopReason(_FactorEnum):
    FRESH_CACHE = "fresh_cache"
    SUFFICIENT_CONFIDENCE = "sufficient_confidence"
    BELOW_MATERIALITY = "below_materiality"
    MAX_DEPTH = "max_depth"
    MAX_NODES = "max_nodes"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_RESOLUTION_COST = "max_resolution_cost"
    MAX_EXTERNAL_CALLS = "max_external_calls"
    NO_RESOLVER = "no_resolver"
    NO_DECOMPOSITION = "no_decomposition"
    UNRESOLVED_DEPENDENCIES = "unresolved_dependencies"
    FAILED = "failed"


class FactorFreshnessMode(_FactorEnum):
    AUTO = "auto"
    IMMUTABLE = "immutable"
    TTL = "ttl"
    EXPIRES = "expires"
    NONE = "none"


class FreshnessReason(_FactorEnum):
    ELIGIBLE = "eligible"
    NO_ENTRY = "no_entry"
    KEY_MISMATCH = "key_mismatch"
    TARGET_PERIOD_MISMATCH = "target_period_mismatch"
    FUTURE_INFORMATION = "future_information"
    AVAILABILITY_AFTER_AS_OF = "availability_after_information_as_of"
    CONFIDENCE_BELOW_MINIMUM = "confidence_below_minimum"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"
    TTL_EXPIRED = "ttl_expired"
    IMMUTABLE_REQUIRED = "immutable_required"
    DEPENDENCY_FINGERPRINT_MISMATCH = "dependency_fingerprint_mismatch"


DateLike = dt.date | dt.datetime


class FactorPeriod(BaseModel):
    """The target period is part of identity; FY, FQ, spot, and long-term differ."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    target_year: Optional[int] = Field(default=None, ge=1, le=9999)
    period_type: FactorPeriodType
    period_key: str
    start: Optional[dt.date] = None
    end: Optional[dt.date] = None

    @field_validator("period_type", mode="before")
    @classmethod
    def normalize_period_type(cls, value: object) -> FactorPeriodType:
        return FactorPeriodType(value)

    @field_validator("period_key")
    @classmethod
    def normalize_period_key(cls, value: str) -> str:
        return canonicalize_token(value, field="period_key")

    @model_validator(mode="after")
    def validate_dates(self) -> "FactorPeriod":
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("FactorPeriod start must not be after end")
        return self

    @property
    def canonical(self) -> dict[str, Any]:
        return {
            "target_year": self.target_year,
            "period_type": self.period_type.value,
            "period_key": self.period_key,
            "start": self.start,
            "end": self.end,
        }


class FactorKey(BaseModel):
    """Semantic factor identity, intentionally independent of requester context."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    domain: FactorDomain
    subject_type: str
    subject_id: str
    metric: str
    geography: Optional[str] = None
    market: Optional[str] = None
    industry: Optional[str] = None
    product: Optional[str] = None
    business: Optional[str] = None
    period: FactorPeriod = Field(
        alias="factor_period",
        validation_alias=AliasChoices("factor_period", "period", "target_period"),
    )
    unit: str
    currency: Optional[str] = None
    basis: Optional[str] = None
    qualifier: Optional[str] = None

    @field_validator("domain", mode="before")
    @classmethod
    def normalize_domain(cls, value: object) -> FactorDomain:
        return FactorDomain(value)

    @field_validator(
        "subject_type",
        "metric",
        "market",
        "industry",
        "product",
        "business",
        "basis",
        "qualifier",
    )
    @classmethod
    def normalize_tokens(cls, value: Optional[str], info) -> Optional[str]:
        if value is None:
            return None
        return canonicalize_token(value, field=info.field_name)

    @field_validator("subject_id")
    @classmethod
    def normalize_subject_id(cls, value: str) -> str:
        # Composite company/business identities intentionally retain the
        # delimiter so ``company:segment`` cannot be confused with a single
        # globally scoped token.
        parts = value.split(":")
        if any(not part.strip() for part in parts):
            raise ValueError("subject_id composite components cannot be blank")
        return ":".join(canonicalize_token(part, field="subject_id") for part in parts)

    @field_validator("geography")
    @classmethod
    def normalize_geography(cls, value: Optional[str]) -> Optional[str]:
        return canonicalize_geography(value) if value is not None else None

    @field_validator("unit")
    @classmethod
    def normalize_unit(cls, value: str) -> str:
        return canonicalize_unit(value)

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: Optional[str]) -> Optional[str]:
        return canonicalize_currency(value) if value is not None else None

    @property
    def factor_period(self) -> FactorPeriod:
        return self.period

    @property
    def canonical(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "metric": self.metric,
            "geography": self.geography,
            "market": self.market,
            "industry": self.industry,
            "product": self.product,
            "business": self.business,
            "period": self.period.canonical,
            "unit": self.unit,
            "currency": self.currency,
            "basis": self.basis,
            "qualifier": self.qualifier,
        }

    @property
    def semantic_id(self) -> str:
        return canonical_json(self.canonical)

    @property
    def digest(self) -> str:
        return stable_digest(self.canonical)


class FactorRange(BaseModel):
    low: Decimal
    base: Decimal
    high: Decimal

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("low", "base", "high")
    @classmethod
    def finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Factor ranges must contain finite values")
        return value

    @model_validator(mode="after")
    def ordered(self) -> "FactorRange":
        if not self.low <= self.base <= self.high:
            raise ValueError("Factor range must satisfy low <= base <= high")
        return self

    @classmethod
    def from_point(cls, point: Decimal) -> "FactorRange":
        return cls(low=point, base=point, high=point)

    @property
    def point(self) -> Optional[Decimal]:
        return self.base if self.low == self.base == self.high else None


class FactorProvenance(BaseModel):
    """Optional structured citation carried by factor evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    source: str = Field(validation_alias=AliasChoices("source", "name"))
    source_id: Optional[str] = None
    publisher: Optional[str] = None
    reference: Optional[str] = None
    locator: Optional[str] = None
    url: Optional[str] = None
    notes: Optional[str] = None

    @field_validator(
        "source",
        "source_id",
        "publisher",
        "reference",
        "locator",
        "url",
        "notes",
    )
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Factor provenance text cannot be blank")
        return value


class FactorEvidence(BaseModel):
    """Raw evidence, represented as an ordered range even when supplied as a point."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    key: FactorKey
    low: Optional[Decimal] = None
    base: Optional[Decimal] = None
    high: Optional[Decimal] = None
    point: Optional[Decimal] = None
    information_available_on: DateLike
    observed_on: DateLike
    source: str
    evidence_id: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("evidence_id", "id")
    )
    all_availability_dates: tuple[DateLike, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "all_availability_dates", "availability_dates"
        ),
    )
    provenance: Optional[FactorProvenance | str] = None
    confidence: FactorConfidence
    immutable: bool = False
    superseded: bool = False
    version: int = Field(default=1, ge=1)
    warnings: tuple[str, ...] = ()
    # These additive fields let adapters retain source-tier and consensus
    # metadata without making it part of FactorKey identity.  Existing direct
    # evidence callers can continue to omit them.
    source_type: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("source_type", "source_tier"),
    )
    source_types: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    dispersion: Optional[Decimal] = Field(default=None, ge=0)

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        return canonicalize_token(value, field="source")

    @field_validator("evidence_id")
    @classmethod
    def normalize_evidence_id(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if value is not None and value.strip() else None

    @field_validator("provenance")
    @classmethod
    def normalize_provenance(
        cls, value: Optional[FactorProvenance | str]
    ) -> Optional[FactorProvenance | str]:
        if value is None:
            return None
        if isinstance(value, FactorProvenance):
            return value
        value = value.strip()
        if not value:
            raise ValueError("provenance cannot be blank")
        return value

    @field_validator("warnings")
    @classmethod
    def normalize_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(str(item).strip() for item in value if str(item).strip())

    @field_validator("source_type")
    @classmethod
    def normalize_source_type(cls, value: Optional[str]) -> Optional[str]:
        return canonicalize_token(value, field="source_type") if value is not None else None

    @field_validator("source_types", mode="before")
    @classmethod
    def normalize_source_types(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = (value,)
        normalized = tuple(
            canonicalize_token(str(item), field="source_type")
            for item in value
            if str(item).strip()
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError("source_types must be unique")
        return normalized

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def normalize_evidence_refs(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            value = (value,)
        normalized = tuple(str(item).strip() for item in value if str(item).strip())
        if len(normalized) != len(set(normalized)):
            raise ValueError("evidence_refs must be unique")
        return normalized

    @field_validator("dispersion")
    @classmethod
    def normalize_dispersion(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("evidence dispersion must be finite")
        return value

    @field_validator("all_availability_dates", mode="before")
    @classmethod
    def normalize_availability_dates(cls, value: Any) -> tuple[DateLike, ...]:
        if value is None:
            return ()
        if isinstance(value, (dt.date, dt.datetime)):
            value = (value,)
        return tuple(dict.fromkeys(value))

    @model_validator(mode="after")
    def normalize_range(self) -> "FactorEvidence":
        supplied = (self.low, self.base, self.high)
        if self.point is not None:
            if any(value is not None for value in supplied):
                raise ValueError("Provide either point or low/base/high evidence")
            if not self.point.is_finite():
                raise ValueError("Factor evidence must be finite")
            object.__setattr__(self, "low", self.point)
            object.__setattr__(self, "base", self.point)
            object.__setattr__(self, "high", self.point)
        elif any(value is None for value in supplied):
            raise ValueError("Evidence requires point or low, base, and high")
        FactorRange(low=self.low, base=self.base, high=self.high)
        if not self.all_availability_dates:
            object.__setattr__(
                self, "all_availability_dates", (self.information_available_on,)
            )
        if self.evidence_id is not None and not self.evidence_refs:
            object.__setattr__(self, "evidence_refs", (self.evidence_id,))
        if self.source_type is not None and not self.source_types:
            object.__setattr__(self, "source_types", (self.source_type,))
        return self

    @property
    def range(self) -> FactorRange:
        return FactorRange(low=self.low, base=self.base, high=self.high)

    @property
    def normalized_range(self) -> FactorRange:
        return self.range

    @property
    def fingerprint(self) -> str:
        return stable_digest(
            {
                "key": self.key.canonical,
                "range": self.range.model_dump(mode="python"),
                "information_available_on": self.information_available_on,
                "observed_on": self.observed_on,
                "source": self.source,
                "evidence_id": self.evidence_id,
                "all_availability_dates": self.all_availability_dates,
                "provenance": self.provenance,
                "confidence": self.confidence,
                "immutable": self.immutable,
                "superseded": self.superseded,
                "version": self.version,
                "warnings": self.warnings,
                "source_type": self.source_type,
                "source_types": self.source_types,
                "evidence_refs": self.evidence_refs,
                "dispersion": self.dispersion,
            }
        )

    @property
    def id(self) -> Optional[str]:
        return self.evidence_id

    @property
    def availability_dates(self) -> tuple[DateLike, ...]:
        return self.all_availability_dates

    @property
    def source_tier(self) -> Optional[str]:
        return self.source_type

    @property
    def source_date(self) -> DateLike:
        return self.observed_on


class FactorDependency(BaseModel):
    key: FactorKey
    fingerprint: str

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("fingerprint")
    @classmethod
    def normalize_fingerprint(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("dependency fingerprint cannot be blank")
        return value


class FactorEstimate(BaseModel):
    """A computed factor version, separate from its raw evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    key: FactorKey
    range: FactorRange = Field(
        alias="factor_range",
        validation_alias=AliasChoices(
            "factor_range", "range", "estimate_range", "value_range"
        ),
    )
    unit: str
    currency: Optional[str] = None
    info_as_of: DateLike
    target_period: FactorPeriod
    confidence: FactorConfidence
    methodology: str
    resolver: str
    evidence_refs: tuple[str, ...] = ()
    dependencies: tuple[FactorKey, ...] = ()
    dependency_fingerprints: tuple[tuple[str, str], ...] = ()
    all_availability_dates: tuple[DateLike, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "all_availability_dates", "availability_dates"
        ),
    )
    created_at: DateLike
    expires_at: Optional[DateLike] = None
    immutable: bool = False
    superseded: bool = False
    source: Optional[str] = None
    version: int = Field(default=1, ge=1)
    warnings: tuple[str, ...] = ()

    @field_validator("unit")
    @classmethod
    def normalize_unit(cls, value: str) -> str:
        return canonicalize_unit(value)

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: Optional[str]) -> Optional[str]:
        return canonicalize_currency(value) if value is not None else None

    @field_validator("methodology", "resolver")
    @classmethod
    def normalize_required_text(cls, value: str, info) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name} cannot be blank")
        return value

    @field_validator("source")
    @classmethod
    def normalize_source(cls, value: Optional[str]) -> Optional[str]:
        return canonicalize_token(value, field="source") if value is not None else None

    @field_validator("evidence_refs")
    @classmethod
    def normalize_evidence_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(str(item).strip() for item in value if str(item).strip())
        if len(normalized) != len(set(normalized)):
            raise ValueError("evidence_refs must be unique")
        return normalized

    @field_validator("warnings")
    @classmethod
    def normalize_estimate_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(str(item).strip() for item in value if str(item).strip())

    @field_validator("all_availability_dates", mode="before")
    @classmethod
    def normalize_availability_dates(cls, value: Any) -> tuple[DateLike, ...]:
        if value is None:
            return ()
        if isinstance(value, (dt.date, dt.datetime)):
            value = (value,)
        unique = []
        for item in value:
            if item not in unique:
                unique.append(item)
        return tuple(unique)

    @field_validator("dependency_fingerprints", mode="before")
    @classmethod
    def normalize_dependency_fingerprints(
        cls, value: Any
    ) -> tuple[tuple[str, str], ...]:
        if value is None:
            return ()
        if isinstance(value, Mapping):
            items = value.items()
        else:
            items = value
        result = []
        for item in items:
            if isinstance(item, Mapping):
                key = item.get("key") or item.get("digest")
                fingerprint = item.get("fingerprint")
            else:
                try:
                    key, fingerprint = item
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "dependency_fingerprints must be key/fingerprint pairs"
                    ) from exc
            if hasattr(key, "digest"):
                key = key.digest
            if key is None or fingerprint is None:
                raise ValueError("dependency fingerprints require key and fingerprint")
            result.append((str(key).strip(), str(fingerprint).strip().lower()))
        if len({key for key, _ in result}) != len(result):
            raise ValueError("dependency_fingerprints must have unique keys")
        return tuple(sorted(result))

    @model_validator(mode="before")
    @classmethod
    def accept_dependency_refs(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        refs = values.pop("dependency_refs", None)
        if refs is not None and "dependencies" not in values:
            dependencies = []
            fingerprints = dict(values.get("dependency_fingerprints") or {})
            for ref in refs:
                if isinstance(ref, FactorDependency):
                    dependencies.append(ref.key)
                    fingerprints[ref.key.digest] = ref.fingerprint
                elif isinstance(ref, Mapping):
                    dependency = FactorDependency.model_validate(ref)
                    dependencies.append(dependency.key)
                    fingerprints[dependency.key.digest] = dependency.fingerprint
                else:
                    raise ValueError(
                        "dependency_refs must contain FactorDependency values"
                    )
            values["dependencies"] = dependencies
            values["dependency_fingerprints"] = fingerprints
        elif "dependencies" in values:
            dependencies = []
            fingerprints = dict(values.get("dependency_fingerprints") or {})
            raw_dependencies = values["dependencies"]
            dependencies_are_mapping = isinstance(raw_dependencies, Mapping)
            if dependencies_are_mapping:
                raw_dependencies = raw_dependencies.items()
            for raw_dependency in raw_dependencies:
                raw_fingerprint = None
                if dependencies_are_mapping:
                    raw_dependency, raw_fingerprint = raw_dependency
                if isinstance(raw_dependency, FactorDependency):
                    dependencies.append(raw_dependency.key)
                    fingerprints[raw_dependency.key.digest] = raw_dependency.fingerprint
                elif (
                    isinstance(raw_dependency, FactorKey)
                    and raw_fingerprint is not None
                ):
                    dependencies.append(raw_dependency)
                    fingerprints[raw_dependency.digest] = str(raw_fingerprint)
                elif (
                    isinstance(raw_dependency, Mapping)
                    and "fingerprint" in raw_dependency
                ):
                    dependency = FactorDependency.model_validate(raw_dependency)
                    dependencies.append(dependency.key)
                    fingerprints[dependency.key.digest] = dependency.fingerprint
                else:
                    dependencies.append(raw_dependency)
            values["dependencies"] = dependencies
            if fingerprints:
                values["dependency_fingerprints"] = fingerprints
        if "factor_range" in values and "range" not in values:
            values["range"] = values.pop("factor_range")
        return values

    @model_validator(mode="after")
    def validate_semantics(self) -> "FactorEstimate":
        if self.target_period != self.key.period:
            raise ValueError("target_period must exactly match key.factor_period")
        if self.unit != self.key.unit:
            raise ValueError("estimate unit must match key unit")
        if self.currency != self.key.currency:
            raise ValueError("estimate currency must match key currency")
        if not self.all_availability_dates:
            methodology = self.methodology.casefold()
            resolver = self.resolver.casefold()
            evidence_backed = bool(self.evidence_refs) or "evidence" in resolver
            configured = any(
                token in methodology
                for token in ("manual", "configured")
            )
            if evidence_backed or not configured:
                raise ValueError(
                    "all_availability_dates is required for non-configured estimates"
                )
            object.__setattr__(self, "all_availability_dates", (self.info_as_of,))
        dependency_keys = [dependency.digest for dependency in self.dependencies]
        if len(dependency_keys) != len(set(dependency_keys)):
            raise ValueError("estimate dependencies must be unique")
        dependency_by_identity = {
            identity: dependency
            for dependency in self.dependencies
            for identity in (dependency.digest, dependency.semantic_id)
        }
        normalized_fingerprints = []
        for key, fingerprint in self.dependency_fingerprints:
            dependency = dependency_by_identity.get(key)
            if dependency is None:
                raise ValueError(
                    "dependency fingerprint keys must match estimate dependencies"
                )
            normalized_fingerprints.append((dependency.digest, fingerprint))
        if len({key for key, _ in normalized_fingerprints}) != len(
            normalized_fingerprints
        ):
            raise ValueError("dependency fingerprints must be unique")
        if len(normalized_fingerprints) != len(self.dependencies):
            raise ValueError("every estimate dependency requires a fingerprint")
        normalized_tuple = tuple(sorted(normalized_fingerprints))
        if normalized_tuple != self.dependency_fingerprints:
            object.__setattr__(self, "dependency_fingerprints", normalized_tuple)
        return self

    @property
    def dependency_fingerprint_map(self) -> dict[str, str]:
        return dict(self.dependency_fingerprints)

    @property
    def dependency_refs(self) -> tuple[FactorDependency, ...]:
        fingerprints = self.dependency_fingerprint_map
        return tuple(
            FactorDependency(key=key, fingerprint=fingerprints.get(key.digest, ""))
            for key in self.dependencies
            if key.digest in fingerprints
        )

    @property
    def factor_range(self) -> FactorRange:
        return self.range

    @property
    def value_range(self) -> FactorRange:
        return self.range

    @property
    def availability_dates(self) -> tuple[DateLike, ...]:
        return self.all_availability_dates

    @property
    def fingerprint(self) -> str:
        """Deterministic computation fingerprint, excluding cache timing."""

        return stable_digest(
            {
                "key": self.key.canonical,
                "range": self.range.model_dump(mode="python"),
                "unit": self.unit,
                "currency": self.currency,
                "info_as_of": self.info_as_of,
                "target_period": self.target_period.canonical,
                "confidence": self.confidence,
                "methodology": self.methodology,
                "resolver": self.resolver,
                "evidence_refs": self.evidence_refs,
                "dependencies": [key.canonical for key in self.dependencies],
                "dependency_fingerprints": self.dependency_fingerprints,
                "all_availability_dates": self.all_availability_dates,
                "immutable": self.immutable,
                "superseded": self.superseded,
                "source": self.source,
                "version": self.version,
                "warnings": self.warnings,
            }
        )


class FactorCost(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    amount: Decimal = Field(default=Decimal("0"), ge=0)
    currency: str = "USD"
    units: int = Field(default=1, ge=0)
    provider: Optional[str] = None

    @field_validator("amount")
    @classmethod
    def finite_amount(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("cost must be finite")
        return value

    @field_validator("currency")
    @classmethod
    def normalize_cost_currency(cls, value: str) -> str:
        return canonicalize_currency(value)

    @field_validator("provider")
    @classmethod
    def normalize_cost_provider(cls, value: Optional[str]) -> Optional[str]:
        return (
            canonicalize_token(value, field="provider") if value is not None else None
        )


class FactorBudget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_cost: Optional[Decimal] = Field(default=None, ge=0)
    currency: str = "USD"
    max_attempts: Optional[int] = Field(default=None, ge=0)
    max_depth: Optional[int] = Field(default=None, ge=0)
    max_nodes: Optional[int] = Field(default=None, ge=0)
    max_external_calls: Optional[int] = Field(default=None, ge=0)
    max_model_calls: Optional[int] = Field(default=None, ge=0)
    max_resolution_cost: Optional[Decimal] = Field(default=None, ge=0)

    @field_validator("max_cost")
    @classmethod
    def finite_budget(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("budget max_cost must be finite")
        return value

    @field_validator("max_resolution_cost")
    @classmethod
    def finite_resolution_budget(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and not value.is_finite():
            raise ValueError("budget max_resolution_cost must be finite")
        return value

    @field_validator("currency")
    @classmethod
    def normalize_budget_currency(cls, value: str) -> str:
        return canonicalize_currency(value)


class FactorRequest(BaseModel):
    """A resolution request; requester/audit data is never factor identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: FactorKey
    information_as_of: DateLike
    priority: FactorPriority = FactorPriority.NORMAL
    materiality: FactorMateriality = FactorMateriality.MEDIUM
    min_confidence: FactorConfidence = Field(
        default=FactorConfidence.LOW,
        alias="minimum_confidence",
        validation_alias=AliasChoices("minimum_confidence", "min_confidence"),
    )
    max_depth: int = Field(default=4, ge=0)
    remaining_depth: Optional[int] = Field(default=None, ge=0)
    budget: Optional[FactorBudget] = None
    requester: Optional[str] = None
    audit_context: Mapping[str, str] = Field(default_factory=dict)

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: object) -> FactorPriority:
        return FactorPriority(value)

    @field_validator("materiality", mode="before")
    @classmethod
    def normalize_materiality(cls, value: object) -> FactorMateriality:
        return FactorMateriality(value)

    @field_validator("min_confidence", mode="before")
    @classmethod
    def normalize_min_confidence(cls, value: object) -> FactorConfidence:
        return FactorConfidence(value)

    @field_validator("requester")
    @classmethod
    def normalize_requester(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("audit_context")
    @classmethod
    def normalize_audit_context(cls, value: Mapping[str, str]) -> dict[str, str]:
        return {str(key): str(item) for key, item in value.items()}

    @model_validator(mode="after")
    def validate_depth(self) -> "FactorRequest":
        if self.remaining_depth is not None and self.remaining_depth > self.max_depth:
            raise ValueError("remaining_depth cannot exceed max_depth")
        return self


class FactorAttempt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: FactorResolutionStatus
    cost: FactorCost = FactorCost()
    started_at: DateLike
    completed_at: Optional[DateLike] = None
    stop_reason: Optional[StopReason] = None
    warnings: tuple[str, ...] = ()


class FactorDependencyProposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    key: FactorKey
    reason: str = Field(
        default="",
        validation_alias=AliasChoices("reason", "rationale"),
    )
    required: bool = True
    relationship_role: str = Field(
        default="dependency",
        validation_alias=AliasChoices("relationship_role", "role"),
    )
    materiality: FactorMateriality = FactorMateriality.MEDIUM
    priority: FactorPriority = FactorPriority.NORMAL
    weight: Decimal = Decimal("1")
    cost: Any = None

    @field_validator("materiality", mode="before")
    @classmethod
    def normalize_materiality(cls, value):
        return FactorMateriality(value)

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value):
        return FactorPriority(value)

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        value = value.strip()
        return value

    @property
    def rationale(self) -> str:
        return self.reason


# Short aliases retain a compact public vocabulary for callers while the
# descriptive names above remain the canonical class names.
FactorStatus = FactorResolutionStatus
FactorCacheState = CacheState
CacheLookupState = CacheState
Confidence = FactorConfidence
Materiality = FactorMateriality
Priority = FactorPriority


__all__ = [
    "CacheState",
    "FactorAttempt",
    "FactorBudget",
    "FactorCacheState",
    "FactorConfidence",
    "FactorCost",
    "FactorDependency",
    "FactorDependencyProposal",
    "FactorDomain",
    "FactorEstimate",
    "FactorEvidence",
    "FactorFreshnessMode",
    "FactorKey",
    "FactorMateriality",
    "FactorPeriod",
    "FactorPeriodType",
    "FactorProvenance",
    "FactorPriority",
    "FactorRange",
    "FactorRequest",
    "FactorResolutionStatus",
    "FactorStatus",
    "CacheLookupState",
    "Confidence",
    "Materiality",
    "Priority",
    "FreshnessReason",
    "StopReason",
]
