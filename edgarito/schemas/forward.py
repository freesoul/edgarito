"""Provider-neutral forward revenue evidence models.

Forward estimates are deliberately kept out of
``NormalizedCompanyFinancials``.  The latter contains reported or
period-reconstructed facts, while the models in this module describe an
external expectation about a future fiscal year.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from enum import Enum

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class ForwardEstimateProviderStatus(str, Enum):
    """Outcome of one provider attempt in the forward-estimate resolver."""

    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    NOT_NEEDED = "not_needed"


class ForwardRevenueEstimate(BaseModel):
    """One annual analyst/forward revenue estimate.

    ``fiscal_year`` is the issuer fiscal year after provider-specific period
    labels have been normalized.  It is not a calendar-year label unless the
    issuer's fiscal year actually ends in December.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fiscal_year: int = Field(ge=1900, le=2200)
    average: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "average",
            "avg",
            "value",
            "revenue",
            "estimated_revenue",
        ),
    )
    low: Decimal | None = None
    high: Decimal | None = None
    analyst_count: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "analyst_count",
            "number_of_analysts",
            "numberOfAnalysts",
        ),
    )
    source: str = Field(
        default="unknown", validation_alias=AliasChoices("source", "provider")
    )
    currency: str | None = None
    observed_at: datetime.datetime | datetime.date | None = None
    period_end: datetime.date | None = None
    source_period: str | None = None
    mapping_method: str | None = None
    confidence: str | None = None

    @field_validator("average", "low", "high")
    @classmethod
    def validate_amount(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and (not value.is_finite() or value < 0):
            raise ValueError("Forward revenue estimates must be finite and non-negative")
        return value

    @field_validator("source", "mapping_method", "source_period")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        return value.strip().upper() if value else None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().casefold()
        if normalized not in {"high", "medium", "low"}:
            raise ValueError("Forward estimate confidence must be high, medium, or low")
        return normalized

    @classmethod
    def from_value(
        cls,
        fiscal_year: int,
        value: Decimal,
        *,
        source: str,
        **kwargs,
    ) -> "ForwardRevenueEstimate":
        """Construct a point estimate while retaining the common contract."""

        return cls(fiscal_year=fiscal_year, average=value, source=source, **kwargs)

    @property
    def midpoint(self) -> Decimal | None:
        """Return the point estimate or the midpoint of an available range."""

        if self.average is not None:
            return self.average
        if self.low is not None and self.high is not None:
            return (self.low + self.high) / Decimal(2)
        return self.low if self.low is not None else self.high

    @property
    def value(self) -> Decimal | None:
        """Compatibility alias for consumers that use a scalar estimate."""

        return self.midpoint

    @property
    def estimate(self) -> Decimal | None:
        return self.midpoint

    @property
    def revenue(self) -> Decimal | None:
        return self.midpoint

    @property
    def provider(self) -> str:
        """Return the source provider without duplicating it in the schema."""

        return self.source


class ForwardEstimateProviderDiagnostic(BaseModel):
    """Content-free diagnostics for one forward-estimate provider attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    status: ForwardEstimateProviderStatus | str
    reason: str | None = None
    estimate_count: int = Field(default=0, ge=0)
    years: tuple[int, ...] = ()
    credentials_available: bool | None = None
    attempted: bool = True

    @field_validator("provider", "reason")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @property
    def label(self) -> str:
        return self.provider.replace("_", " ").title()


class ForwardRevenueEstimateResult(BaseModel):
    """Resolved provider result plus fallback diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    estimates: tuple[ForwardRevenueEstimate, ...] = ()
    selected_provider: str | None = None
    diagnostics: tuple[ForwardEstimateProviderDiagnostic, ...] = ()
    warnings: tuple[str, ...] = ()
    fallback_reason: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.estimates)

    @property
    def provider(self) -> str | None:
        return self.selected_provider

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(item.fiscal_year for item in self.estimates)


__all__ = [
    "ForwardEstimateProviderDiagnostic",
    "ForwardEstimateProviderStatus",
    "ForwardRevenueEstimate",
    "ForwardRevenueEstimateResult",
]
