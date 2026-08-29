"""Immutable, provider-neutral contracts for market-intelligence evidence.

The contracts in this module describe observations only.  They deliberately do
not contain a forecast horizon, a model, or a provider-specific payload.  A
range is an ordered ``low``/``base``/``high`` observation; reconciliation of
several such observations lives in :mod:`edgarito.services.research.consensus`.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal, TypeAlias, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class EvidenceSourceType(str, Enum):
    """Evidence tiers, ordered from strongest to weakest."""

    REPORTED_FIRST_PARTY_FACT = "reported_first_party_fact"
    INDEPENDENT_INDUSTRY_ESTIMATE = "independent_industry_estimate"
    ANALYST_ESTIMATE = "analyst_estimate"
    DERIVED_OBSERVATION = "derived_observation"

    # Readable compatibility aliases do not add additional precedence tiers.
    FIRST_PARTY_REPORTED_FACT = REPORTED_FIRST_PARTY_FACT
    FIRST_PARTY = REPORTED_FIRST_PARTY_FACT
    INDUSTRY_ESTIMATE = INDEPENDENT_INDUSTRY_ESTIMATE
    INDEPENDENT = INDEPENDENT_INDUSTRY_ESTIMATE
    ANALYST = ANALYST_ESTIMATE
    DERIVED = DERIVED_OBSERVATION


SOURCE_PRECEDENCE: tuple[EvidenceSourceType, ...] = (
    EvidenceSourceType.REPORTED_FIRST_PARTY_FACT,
    EvidenceSourceType.INDEPENDENT_INDUSTRY_ESTIMATE,
    EvidenceSourceType.ANALYST_ESTIMATE,
    EvidenceSourceType.DERIVED_OBSERVATION,
)


_CONTEXT_FIELD_MAP = {
    "market": "market",
    "geography": "geography",
    "segment": "segment",
    "company": "company",
    "competitor": "competitor",
    "product": "product",
    "facility": "facility",
    "period": "period",
    "metric": "metric",
    "currency": "currency",
    "market_scope": "scope",
    "share_basis": "qualifier",
    "growth_basis": "qualifier",
    "constraint_type": "qualifier",
    "price_type": "qualifier",
    "observation_type": "qualifier",
}


class EvidenceConfidence(str, Enum):
    """Confidence labels retained on observations and consensus results."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return {self.LOW: 1, self.MEDIUM: 2, self.HIGH: 3}[self]


class EvidenceKind(str, Enum):
    """Supported market-intelligence evidence categories."""

    MARKET_SIZE = "market_size"
    MARKET_GROWTH = "market_growth"
    MARKET_SHARE = "market_share"
    COMPETITOR = "competitor"
    PRODUCTION_CAPACITY = "production_capacity"
    PRICING = "pricing"


class EvidenceContext(BaseModel):
    """Provider-neutral dimensions that make observations comparable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    market: str | None = None
    geography: str | None = None
    segment: str | None = None
    company: str | None = None
    competitor: str | None = None
    product: str | None = None
    facility: str | None = None
    period: str | None = None
    metric: str | None = None
    currency: str | None = None
    scope: str | None = None
    qualifier: str | None = None

    @field_validator(
        "market",
        "geography",
        "segment",
        "company",
        "competitor",
        "product",
        "facility",
        "period",
        "metric",
        "currency",
        "scope",
        "qualifier",
    )
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Evidence context text cannot be blank")
        return normalized

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        return value.upper() if value is not None else None

    @field_validator("scope", "qualifier")
    @classmethod
    def normalize_labels(cls, value: str | None) -> str | None:
        return value.casefold() if value is not None else None


class EvidenceProvenance(BaseModel):
    """Typed source identity and optional citation details."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(validation_alias=AliasChoices("source", "name"))
    source_id: str | None = None
    publisher: str | None = None
    reference: str | None = None
    locator: str | None = None
    url: str | None = None
    notes: str | None = None

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
    def normalize_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Evidence provenance text cannot be blank")
        return normalized

    @property
    def source_name(self) -> str:
        """Return the human-readable source label."""

        return self.source


class EstimateRange(BaseModel):
    """A finite, ordered low/base/high estimate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    low: Decimal
    base: Decimal
    high: Decimal

    @field_validator("low", "base", "high")
    @classmethod
    def require_finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Evidence estimates must be finite")
        return value

    @model_validator(mode="after")
    def validate_order(self) -> "EstimateRange":
        if not self.low <= self.base <= self.high:
            raise ValueError("Evidence estimate must satisfy low <= base <= high")
        return self


class ResearchEvidence(BaseModel):
    """Common immutable payload shared by every evidence kind."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EvidenceKind
    evidence_id: str | None = None
    source_date: datetime.date = Field(
        validation_alias=AliasChoices(
            "source_date", "observed_on", "date", "as_of", "published_on"
        )
    )
    source_type: EvidenceSourceType = Field(
        validation_alias=AliasChoices("source_type", "source_category")
    )
    low: Decimal
    base: Decimal
    high: Decimal
    provenance: EvidenceProvenance | str
    confidence: EvidenceConfidence = EvidenceConfidence.MEDIUM
    unit: str = Field(validation_alias=AliasChoices("unit", "units"))
    currency: str | None = None
    context: EvidenceContext = Field(default_factory=EvidenceContext)
    metric: str | None = None
    notes: str | None = None

    @field_validator("evidence_id", "unit", "metric", "currency", "notes")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Evidence text fields cannot be blank")
        return normalized

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        return value.upper() if value is not None else None

    @field_validator(
        "market",
        "geography",
        "segment",
        "company",
        "competitor",
        "product",
        "facility",
        "period",
        "market_scope",
        "share_basis",
        "growth_basis",
        "constraint_type",
        "price_type",
        "observation_type",
        "observation",
        "constraint",
        check_fields=False,
    )
    @classmethod
    def normalize_context_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Evidence context fields cannot be blank")
        return normalized

    @field_validator(
        "market_scope",
        "share_basis",
        "growth_basis",
        "constraint_type",
        "price_type",
        "observation_type",
        check_fields=False,
    )
    @classmethod
    def normalize_category_labels(cls, value: str | None) -> str | None:
        return value.casefold() if value is not None else None

    @field_validator("source_type", mode="before")
    @classmethod
    def normalize_source_type(
        cls, value: EvidenceSourceType | str
    ) -> EvidenceSourceType | str:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(
        cls, value: EvidenceConfidence | str
    ) -> EvidenceConfidence | str:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("provenance", mode="before")
    @classmethod
    def normalize_provenance(
        cls, value: EvidenceProvenance | str
    ) -> EvidenceProvenance | str:
        if isinstance(value, str):
            return EvidenceProvenance(source=value)
        return value

    @field_validator("low", "base", "high")
    @classmethod
    def require_finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Evidence estimates must be finite")
        return value

    @model_validator(mode="after")
    def validate_estimate(self) -> "ResearchEvidence":
        EstimateRange(low=self.low, base=self.base, high=self.high)
        return self

    @model_validator(mode="before")
    @classmethod
    def expand_estimate_range(cls, value: object) -> object:
        """Accept a nested range while retaining flat low/base/high storage."""

        if not isinstance(value, dict) or "estimate" not in value:
            return value
        payload = dict(value)
        raw_estimate = payload.pop("estimate")
        if isinstance(raw_estimate, EstimateRange):
            estimate_values = raw_estimate.model_dump()
        elif isinstance(raw_estimate, dict):
            estimate_values = raw_estimate
        else:
            return payload
        for name in ("low", "base", "high"):
            nested_value = estimate_values.get(name)
            if name in payload and nested_value is not None:
                if payload[name] != nested_value:
                    raise ValueError(f"Evidence {name} conflicts with estimate")
            elif name not in payload:
                payload[name] = nested_value
        return payload

    @model_validator(mode="before")
    @classmethod
    def merge_context_fields(cls, value: object) -> object:
        """Make direct category fields visible through the common context."""

        if not isinstance(value, dict):
            return value
        payload = dict(value)
        raw_context = payload.get("context", {})
        if isinstance(raw_context, EvidenceContext):
            context_values = raw_context.model_dump()
        elif isinstance(raw_context, dict):
            context_values = dict(raw_context)
        else:
            return payload
        for attribute, context_name in _CONTEXT_FIELD_MAP.items():
            direct_value = payload.get(attribute)
            if direct_value is None:
                continue
            if isinstance(direct_value, Enum):
                direct_value = direct_value.value
            if isinstance(direct_value, str):
                direct_value = direct_value.strip()
                if context_name == "currency":
                    direct_value = direct_value.upper()
            current = context_values.get(context_name)
            if isinstance(current, str):
                current = current.strip()
                if context_name == "currency":
                    current = current.upper()
            if current is not None and current != direct_value:
                raise ValueError(
                    f"Evidence context {context_name!r} conflicts with {attribute!r}"
                )
            context_values[context_name] = direct_value
        payload["context"] = context_values
        return payload

    def model_post_init(self, __context: object) -> None:
        """Include subclass defaults in the common immutable context."""

        context_values = self.context.model_dump()
        for attribute, context_name in _CONTEXT_FIELD_MAP.items():
            direct_value = getattr(self, attribute, None)
            if direct_value is None:
                continue
            if isinstance(direct_value, Enum):
                direct_value = direct_value.value
            if context_name == "currency":
                direct_value = direct_value.upper()
            current = context_values.get(context_name)
            if current is not None and current != direct_value:
                raise ValueError(
                    f"Evidence context {context_name!r} conflicts with {attribute!r}"
                )
            context_values[context_name] = direct_value
        object.__setattr__(self, "context", EvidenceContext(**context_values))

    @property
    def estimate(self) -> EstimateRange:
        """Return the ordered range as a typed value object."""

        return EstimateRange(low=self.low, base=self.base, high=self.high)

    @property
    def source(self) -> str:
        """Return the stable human-readable provenance source label."""

        provenance = self.provenance
        return (
            provenance.source
            if isinstance(provenance, EvidenceProvenance)
            else provenance
        )


class MarketSizeEvidence(ResearchEvidence):
    """Observed market or TAM size."""

    kind: Literal[EvidenceKind.MARKET_SIZE] = EvidenceKind.MARKET_SIZE
    market: str = Field(validation_alias=AliasChoices("market", "market_name", "name"))
    geography: str | None = None
    segment: str | None = None
    market_scope: str = "tam"

    @model_validator(mode="after")
    def validate_nonnegative_size(self) -> "MarketSizeEvidence":
        if self.low < 0:
            raise ValueError("Market-size estimates cannot be negative")
        return self


class MarketGrowthEvidence(ResearchEvidence):
    """Observed or published market growth estimate, not a forecast plan."""

    kind: Literal[EvidenceKind.MARKET_GROWTH] = EvidenceKind.MARKET_GROWTH
    market: str = Field(validation_alias=AliasChoices("market", "market_name", "name"))
    geography: str | None = None
    segment: str | None = None
    growth_basis: str = "growth_rate"


class MarketShareEvidence(ResearchEvidence):
    """Observed market-share range for a company or competitor."""

    kind: Literal[EvidenceKind.MARKET_SHARE] = EvidenceKind.MARKET_SHARE
    company: str = Field(
        validation_alias=AliasChoices("company", "company_name", "name")
    )
    market: str = Field(
        validation_alias=AliasChoices("market", "market_name", "market_id")
    )
    geography: str | None = None
    segment: str | None = None
    share_basis: str = "market_share"

    @model_validator(mode="after")
    def validate_share(self) -> "MarketShareEvidence":
        if self.low < 0:
            raise ValueError("Market-share estimates cannot be negative")
        if self.unit.casefold() in {"%", "percent", "percentage", "percentage_points"}:
            if self.high > 100:
                raise ValueError("Percentage market share cannot exceed 100")
        return self


class CompetitorObservation(ResearchEvidence):
    """Quantified observation about a competitor, with optional description."""

    kind: Literal[EvidenceKind.COMPETITOR] = EvidenceKind.COMPETITOR
    competitor: str = Field(
        validation_alias=AliasChoices("competitor", "competitor_name", "name")
    )
    market: str | None = None
    geography: str | None = None
    segment: str | None = None
    observation_type: str = "competitor_metric"
    observation: str | None = None


class ProductionCapacityEvidence(ResearchEvidence):
    """Observed production, capacity, or supply constraint."""

    kind: Literal[EvidenceKind.PRODUCTION_CAPACITY] = EvidenceKind.PRODUCTION_CAPACITY
    company: str = Field(
        validation_alias=AliasChoices("company", "company_name", "name")
    )
    market: str | None = None
    facility: str | None = None
    geography: str | None = None
    constraint_type: str = "capacity"
    constraint: str | None = None

    @model_validator(mode="after")
    def validate_nonnegative_capacity(self) -> "ProductionCapacityEvidence":
        if self.low < 0:
            raise ValueError("Production-capacity estimates cannot be negative")
        return self


class PricingObservation(ResearchEvidence):
    """Observed price or pricing range for a market offering."""

    kind: Literal[EvidenceKind.PRICING] = EvidenceKind.PRICING
    market: str | None = None
    product: str = Field(
        validation_alias=AliasChoices("product", "product_name", "name")
    )
    competitor: str | None = None
    geography: str | None = None
    price_type: str = "observed_price"
    currency: str | None = None

    @model_validator(mode="after")
    def validate_nonnegative_price(self) -> "PricingObservation":
        if self.low < 0:
            raise ValueError("Pricing observations cannot be negative")
        return self


EvidenceItem: TypeAlias = Annotated[
    Union[
        MarketSizeEvidence,
        MarketGrowthEvidence,
        MarketShareEvidence,
        CompetitorObservation,
        ProductionCapacityEvidence,
        PricingObservation,
    ],
    Field(discriminator="kind"),
]

# Short names are useful to callers without creating separate contracts.
SourceType = EvidenceSourceType
SourceCategory = EvidenceSourceType
Confidence = EvidenceConfidence
ConfidenceLevel = EvidenceConfidence
Provenance = EvidenceProvenance
OrderedEstimate = EstimateRange
EvidenceEstimate = EstimateRange
EvidenceRange = EstimateRange
MarketSizeObservation = MarketSizeEvidence
MarketTAMEvidence = MarketSizeEvidence
MarketSizeTAMEvidence = MarketSizeEvidence
TAMEvidence = MarketSizeEvidence
MarketGrowthObservation = MarketGrowthEvidence
MarketShareObservation = MarketShareEvidence
CompetitorEvidence = CompetitorObservation
ProductionCapacityConstraint = ProductionCapacityEvidence
ProductionCapacityConstraintEvidence = ProductionCapacityEvidence
ProductionCapacityConstraintObservation = ProductionCapacityEvidence
CapacityConstraintObservation = ProductionCapacityEvidence
CapacityConstraintEvidence = ProductionCapacityEvidence
PricingEvidence = PricingObservation


__all__ = [
    "Confidence",
    "ConfidenceLevel",
    "CompetitorEvidence",
    "CompetitorObservation",
    "EvidenceConfidence",
    "EvidenceContext",
    "EvidenceEstimate",
    "EvidenceItem",
    "EvidenceKind",
    "EvidenceProvenance",
    "EvidenceRange",
    "EvidenceSourceType",
    "EstimateRange",
    "MarketGrowthEvidence",
    "MarketGrowthObservation",
    "MarketShareEvidence",
    "MarketShareObservation",
    "MarketSizeEvidence",
    "MarketSizeObservation",
    "MarketSizeTAMEvidence",
    "MarketTAMEvidence",
    "OrderedEstimate",
    "PricingEvidence",
    "PricingObservation",
    "ProductionCapacityConstraint",
    "ProductionCapacityConstraintEvidence",
    "ProductionCapacityConstraintObservation",
    "ProductionCapacityEvidence",
    "CapacityConstraintObservation",
    "CapacityConstraintEvidence",
    "Provenance",
    "ResearchEvidence",
    "SOURCE_PRECEDENCE",
    "SourceType",
    "SourceCategory",
    "TAMEvidence",
]
