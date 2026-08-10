from __future__ import annotations

import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GuidanceMetric(str, Enum):
    REVENUE = "revenue"
    REVENUE_GROWTH = "revenue_growth"
    OPERATING_INCOME = "operating_income"
    OPERATING_MARGIN = "operating_margin"
    EBIT = "ebit"
    EBIT_MARGIN = "ebit_margin"
    EBITDA = "ebitda"
    EBITDA_MARGIN = "ebitda_margin"
    GROSS_MARGIN = "gross_margin"
    CAPEX = "capex"
    OPERATING_CASH_FLOW = "operating_cash_flow"
    FREE_CASH_FLOW = "free_cash_flow"
    EPS = "eps"
    TAX_RATE = "tax_rate"
    BOOKINGS = "bookings"
    BACKLOG = "backlog"
    OTHER = "other"


class GuidancePeriodType(str, Enum):
    QUARTER = "quarter"
    FISCAL_YEAR = "fiscal_year"
    MULTI_YEAR_TARGET = "multi_year_target"
    LONG_TERM_TARGET = "long_term_target"


class GuidanceValueKind(str, Enum):
    MONETARY = "monetary"
    PERCENTAGE = "percentage"
    PER_SHARE = "per_share"
    COUNT = "count"


class GuidanceUnit(str, Enum):
    ACTUAL = "actual"
    THOUSANDS = "thousands"
    MILLIONS = "millions"
    BILLIONS = "billions"
    PERCENT = "percent"
    PER_SHARE = "per_share"


class GuidanceBasis(str, Enum):
    GAAP = "gaap"
    NON_GAAP = "non_gaap"
    REPORTED = "reported"
    CONSTANT_CURRENCY = "constant_currency"
    UNKNOWN = "unknown"


class GuidanceScope(str, Enum):
    CONSOLIDATED = "consolidated"
    SEGMENT = "segment"
    UNKNOWN = "unknown"


class GuidanceQualifier(str, Enum):
    POINT = "point"
    RANGE = "range"
    APPROXIMATELY = "approximately"
    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    MORE_THAN = "more_than"
    LESS_THAN = "less_than"
    UNKNOWN = "unknown"


class MonetaryForecastConstraint(BaseModel):
    """An absolute monetary forecast target or bound with its provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    point: Decimal | None = None
    minimum: Decimal | None = None
    maximum: Decimal | None = None
    source: str = "management_guidance"

    @field_validator("point", "minimum", "maximum")
    @classmethod
    def require_finite_value(cls, value: Decimal | None) -> Decimal | None:
        if value is not None:
            if not value.is_finite():
                raise ValueError("Monetary forecast constraints must be finite")
            if value < 0:
                raise ValueError("Monetary forecast constraints cannot be negative")
        return value

    @field_validator("source")
    @classmethod
    def require_source(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Monetary forecast constraint source cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_bounds(self) -> "MonetaryForecastConstraint":
        if self.point is None and self.minimum is None and self.maximum is None:
            raise ValueError("Monetary forecast constraint requires point or bounds")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("Monetary forecast minimum cannot exceed maximum")
        if self.point is not None:
            if self.minimum is not None and self.point < self.minimum:
                raise ValueError("Monetary forecast point cannot be below minimum")
            if self.maximum is not None and self.point > self.maximum:
                raise ValueError("Monetary forecast point cannot exceed maximum")
        return self

    @property
    def methodology(self) -> str:
        if self.minimum is not None and self.maximum is not None:
            return "range"
        if self.minimum is not None:
            return "floor"
        if self.maximum is not None:
            return "ceiling"
        return "point"


class GuidanceStatus(str, Enum):
    ISSUED = "issued"
    RAISED = "raised"
    LOWERED = "lowered"
    REAFFIRMED = "reaffirmed"
    WITHDRAWN = "withdrawn"


class ExtractedGuidanceItem(BaseModel):
    """Strict OpenAI output before deterministic scaling and validation."""

    model_config = ConfigDict(extra="forbid")

    metric: GuidanceMetric
    metric_name: str | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = Field(default=None, ge=1, le=4)
    period_type: GuidancePeriodType
    # Structured Outputs supports JSON numbers directly. Decimal's generated
    # JSON schema contains regex lookarounds unsupported by the API, so exact
    # Decimal conversion happens immediately in deterministic validation.
    point: float | None = None
    low: float | None = None
    high: float | None = None
    value_kind: GuidanceValueKind
    currency: str | None = None
    unit: GuidanceUnit
    basis: GuidanceBasis = GuidanceBasis.UNKNOWN
    scope: GuidanceScope = GuidanceScope.UNKNOWN
    segment_name: str | None = None
    qualifier: GuidanceQualifier = GuidanceQualifier.UNKNOWN
    status: GuidanceStatus = GuidanceStatus.ISSUED
    supporting_text: str
    extraction_confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_a_number(self) -> "ExtractedGuidanceItem":
        if self.point is None and self.low is None and self.high is None:
            raise ValueError("Numerical guidance requires point or range values")
        return self


class ExtractedGuidanceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guidance: list[ExtractedGuidanceItem] = Field(default_factory=list)


class ManagementGuidance(BaseModel):
    """Provider-neutral, scaled guidance tied to matched SEC evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: GuidanceMetric
    metric_name: str | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = Field(default=None, ge=1, le=4)
    period_type: GuidancePeriodType
    point: Decimal | None = None
    low: Decimal | None = None
    high: Decimal | None = None
    value_kind: GuidanceValueKind
    currency: str | None = None
    unit: str
    basis: GuidanceBasis = GuidanceBasis.UNKNOWN
    scope: GuidanceScope = GuidanceScope.UNKNOWN
    segment_name: str | None = None
    qualifier: GuidanceQualifier = GuidanceQualifier.UNKNOWN
    status: GuidanceStatus = GuidanceStatus.ISSUED
    filing_date: datetime.date
    accession_number: str
    filing_form: str
    source_document: str
    source_document_type: str
    supporting_text: str
    evidence_verified: bool
    extraction_model: str
    extraction_confidence: Decimal | None = Field(default=None, ge=0, le=1)

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        return value.strip().upper() if value else None

    @model_validator(mode="after")
    def validate_values(self) -> "ManagementGuidance":
        values = [
            value for value in (self.point, self.low, self.high) if value is not None
        ]
        if not values:
            raise ValueError("Numerical guidance requires point or range values")
        if any(not value.is_finite() for value in values):
            raise ValueError("Guidance values must be finite")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("Guidance low cannot exceed high")
        if self.period_type == GuidancePeriodType.QUARTER:
            if self.fiscal_year is None or self.fiscal_quarter is None:
                raise ValueError("Quarter guidance requires fiscal year and quarter")
        elif self.fiscal_quarter is not None:
            raise ValueError("Only quarter guidance may set fiscal_quarter")
        if self.scope == GuidanceScope.SEGMENT and not self.segment_name:
            raise ValueError("Segment guidance requires segment_name")
        return self

    @property
    def midpoint(self) -> Decimal | None:
        if self.point is not None:
            return self.point
        if self.low is not None and self.high is not None:
            return (self.low + self.high) / Decimal(2)
        return self.low if self.low is not None else self.high

    @property
    def period_label(self) -> str:
        if self.fiscal_year is None:
            return self.period_type.value.replace("_", " ")
        if self.fiscal_quarter is not None:
            return f"FY{self.fiscal_year} Q{self.fiscal_quarter}"
        return f"FY{self.fiscal_year}"


class GuidanceRejection(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: str
    item: ExtractedGuidanceItem | None = None


class GuidanceExtractionCacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    extracted_at: datetime.datetime
    model: str
    reasoning_effort: str
    prompt_version: str
    schema_version: str
    content_hash: str
    accession_number: str
    document_filename: str
    accepted: tuple[ManagementGuidance, ...] = ()
    rejected: tuple[GuidanceRejection, ...] = ()


class GuidanceApplication(BaseModel):
    model_config = ConfigDict(frozen=True)

    driver: str
    fiscal_year: int
    value: Decimal
    guidance: ManagementGuidance
    methodology: str
    source: str = "management_guidance"


class GuidanceOverlayResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    applications: tuple[GuidanceApplication, ...] = ()
    evidence_only: tuple[ManagementGuidance, ...] = ()
    rejected_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    cache_hits: int = 0
    cache_misses: int = 0
    filings_inspected: int = 0
    documents_inspected: int = 0
    extracted_guidance_records: int = 0
    rejected_records: int = 0
    filings_inspected: int = 0
    documents_inspected: int = 0
    extracted_guidance_records: int = 0
    rejected_records: int = 0
