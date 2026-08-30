"""Read-only adapters for driver-economics validation."""

from __future__ import annotations

from typing import Any

from .contracts import ForecastValidationContext


def driver_economics_to_validation_context(artifact: Any) -> ForecastValidationContext:
    """Expose operating economics/canonical FCFF rows without mutating them."""

    if isinstance(artifact, ForecastValidationContext):
        return artifact
    if hasattr(artifact, "consolidated_revenue") and hasattr(artifact, "years"):
        rows = tuple(
            item.model_dump(mode="python") if hasattr(item, "model_dump") else item
            for item in artifact.years
        )
        return ForecastValidationContext(
            rows=rows,
            rows_supplied=True,
            methodology="driver_based_operating_economics",
            unit=getattr(artifact, "unit", None),
        )
    if hasattr(artifact, "observations"):
        # The generic adapter also retains composite terminal/valuation data.
        return ForecastValidationContext.from_artifact(artifact)
    return ForecastValidationContext.from_artifact(artifact)


adapt_driver_economics = driver_economics_to_validation_context
adapt_operating_economics = driver_economics_to_validation_context
operating_economics_to_validation_context = driver_economics_to_validation_context

__all__ = [
    "adapt_driver_economics",
    "adapt_operating_economics",
    "driver_economics_to_validation_context",
    "operating_economics_to_validation_context",
]
