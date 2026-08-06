import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SpecializedInputType(str, Enum):
    REIT = "reit"
    RESOURCE = "resource"
    BIOTECH = "biotech"
    SOTP = "sotp"

    @property
    def label(self) -> str:
        return {
            SpecializedInputType.REIT: "REIT / property",
            SpecializedInputType.RESOURCE: "Natural resource",
            SpecializedInputType.BIOTECH: "Biotech pipeline",
            SpecializedInputType.SOTP: "Sum-of-the-parts",
        }[self]


class ExtractionReadiness(str, Enum):
    READY = "ready"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class ExtractedFieldOrigin(str, Enum):
    REPORTED = "reported"
    DERIVED_PROXY = "derived_proxy"
    SUPPLIED = "supplied"


class ExtractionPeriodKind(str, Enum):
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    YEAR_TO_DATE = "year_to_date"
    INSTANT = "instant"


class ExtractedValuationField(BaseModel):
    """One reported or derived specialized valuation input with provenance."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: Decimal
    unit: str
    fiscal_year: int
    fiscal_period: str = "FY"
    period_kind: ExtractionPeriodKind = ExtractionPeriodKind.ANNUAL
    period_end: datetime.date
    origin: ExtractedFieldOrigin
    source_concepts: tuple[str, ...]
    accession_numbers: tuple[str, ...] = ()
    derivation: Optional[str] = None
    dimensions: dict[str, str] = Field(default_factory=dict)

    @field_validator("name", "unit", "derivation")
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Extracted field text cannot be blank")
        return normalized

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Extracted values must be finite")
        return value

    @model_validator(mode="after")
    def validate_origin(self) -> "ExtractedValuationField":
        if self.origin == ExtractedFieldOrigin.DERIVED_PROXY and not self.derivation:
            raise ValueError("Derived proxy fields require a derivation")
        if self.origin != ExtractedFieldOrigin.SUPPLIED and not self.source_concepts:
            raise ValueError("Extracted fields require source concepts")
        return self


class SpecializedValuationExtraction(BaseModel):
    """Provider-neutral readiness report for one specialized valuation profile."""

    model_config = ConfigDict(frozen=True)

    provider: str
    company_id: str
    company_name: str
    ticker: Optional[str] = None
    input_type: SpecializedInputType
    readiness: ExtractionReadiness
    source_scope: str = "SEC Company Facts standard taxonomies"
    fields: tuple[ExtractedValuationField, ...] = ()
    missing_inputs: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unique_fields(self) -> "SpecializedValuationExtraction":
        keys = [
            (
                field.name,
                field.fiscal_year,
                field.fiscal_period,
                tuple(sorted(field.dimensions.items())),
            )
            for field in self.fields
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Specialized extracted fields must be unique")
        if self.readiness == ExtractionReadiness.READY and self.missing_inputs:
            raise ValueError("A ready extraction cannot retain missing inputs")
        return self

    def latest(self, name: str) -> Optional[ExtractedValuationField]:
        return max(
            (field for field in self.fields if field.name == name),
            key=lambda field: (field.fiscal_year, field.period_end),
            default=None,
        )


__all__ = [
    "ExtractedFieldOrigin",
    "ExtractedValuationField",
    "ExtractionPeriodKind",
    "ExtractionReadiness",
    "SpecializedInputType",
    "SpecializedValuationExtraction",
]
