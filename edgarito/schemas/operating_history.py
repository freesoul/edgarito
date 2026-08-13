"""Normalized multi-period operating history contracts."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from edgarito.schemas.operating import (
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
)


class OperatingHistoryAudit(BaseModel):
    """Content-free diagnostics for one assembled first-party time series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted_periods: tuple[str, ...] = ()
    accepted_metrics: tuple[str, ...] = ()
    accepted_pairs: tuple[tuple[str, str], ...] = ()
    missing_pairs: tuple[str, ...] = ()
    period_failures: tuple[str, ...] = ()
    unit_failures: tuple[str, ...] = ()
    input_observations: int = Field(default=0, ge=0)
    accepted_observations: int = Field(default=0, ge=0)
    deduplicated_observations: int = Field(default=0, ge=0)
    derived_observations: int = Field(default=0, ge=0)
    historical_revenue_pairs: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @field_validator(
        "accepted_periods",
        "accepted_metrics",
        "missing_pairs",
        "period_failures",
        "unit_failures",
        "historical_revenue_pairs",
        "warnings",
    )
    @classmethod
    def normalize_texts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(str(item).strip() for item in value if str(item).strip())
        )


class OperatingTimeSeries(BaseModel):
    """Company -> segment -> metric observations in compatible periods.

    Values are normalized to base units in ``observations`` while each
    observation retains ``original_unit`` and ``original_scale`` provenance.
    ``historical_revenue`` is deliberately shaped for the existing deterministic
    operating engine and contains only one selected annual/LTM value per segment
    and fiscal year.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    company_id: str = "company"
    segments: tuple[OperatingSegment, ...] = ()
    definitions: tuple[OperatingDriverDefinition, ...] = ()
    observations: tuple[OperatingDriverObservation, ...] = ()
    company_revenue: dict[int, Decimal] = Field(default_factory=dict)
    historical_revenue: dict[str, dict[int, Decimal]] = Field(default_factory=dict)
    audit: OperatingHistoryAudit = Field(default_factory=OperatingHistoryAudit)

    @field_validator("company_id")
    @classmethod
    def normalize_company_id(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Operating history company_id cannot be blank")
        return normalized

    @field_validator("historical_revenue")
    @classmethod
    def normalize_revenue(
        cls, value: dict[str, dict[int, Decimal]]
    ) -> dict[str, dict[int, Decimal]]:
        result: dict[str, dict[int, Decimal]] = {}
        for segment_id, values in value.items():
            segment = str(segment_id).strip()
            if not segment:
                raise ValueError("Operating history segment cannot be blank")
            result[segment] = {
                int(year): amount
                for year, amount in sorted(
                    values.items(), key=lambda item: int(item[0])
                )
                if amount >= 0 and amount.is_finite()
            }
        return result

    @property
    def historical_revenue_by_segment(self) -> dict[str, dict[int, Decimal]]:
        """Descriptive alias for the engine-facing nested history mapping."""

        return self.historical_revenue

    @property
    def engine_historical_revenue(self) -> dict[object, object]:
        """Return the mixed shape accepted by the existing forecast engine."""

        return {
            **self.company_revenue,
            **self.historical_revenue,
        }


# Descriptive aliases keep the service discoverable under both time-series and
# history terminology without creating duplicate contracts.
NormalizedOperatingTimeSeries = OperatingTimeSeries
OperatingHistory = OperatingTimeSeries


__all__ = [
    "NormalizedOperatingTimeSeries",
    "OperatingHistory",
    "OperatingHistoryAudit",
    "OperatingTimeSeries",
]
