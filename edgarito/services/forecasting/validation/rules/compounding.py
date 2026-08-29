"""Visibility checks for total forecast compounding."""

from __future__ import annotations

from ..config import ForecastValidationConfig
from ..contracts import (
    HUNDRED,
    ONE,
    ZERO,
    ForecastValidationContext,
    Severity,
    ValidationCategory,
)
from ._utils import finding


class CompoundingScaleRule:
    name = "compounding_scale"

    def evaluate(
        self, context: ForecastValidationContext, config: ForecastValidationConfig
    ):
        findings = []
        findings.extend(
            _metric_scale(
                context,
                config,
                metric="revenue",
                threshold=config.compounding_revenue_multiple,
            )
        )
        findings.extend(
            _metric_scale(
                context,
                config,
                metric="fcff",
                threshold=config.compounding_fcff_multiple,
            )
        )
        return tuple(findings)


def _metric_scale(context, config, *, metric: str, threshold):
    rows = [row for row in context.rows if getattr(row, metric) is not None]
    if len(rows) < 2:
        return ()
    first = getattr(rows[0], metric)
    final = getattr(rows[-1], metric)
    if first == ZERO or (
        metric == "fcff" and abs(first) <= config.near_zero_denominator
    ):
        return ()
    if first * final < 0:
        return ()
    scale = abs(final) / abs(first)
    total_change_pct = (scale - ONE) * HUNDRED
    result = [
        finding(
            "COMPOUNDING_SCALE",
            Severity.INFO,
            ValidationCategory.COMPOUNDING,
            f"{metric.upper()} changes to {scale}x its first forecast value over the explicit horizon",
            fiscal_year=rows[-1].fiscal_year,
            metric=metric,
            observed_value=scale,
            threshold=threshold,
            reference_value=first,
            explanation=f"Total explicit-horizon compounding is {total_change_pct}%.",
        )
    ]
    if scale >= threshold:
        result.append(
            finding(
                "EXTREME_COMPOUNDING_SCALE",
                Severity.WARNING,
                ValidationCategory.COMPOUNDING,
                f"{metric.upper()} compounds to {scale}x its first forecast value",
                fiscal_year=rows[-1].fiscal_year,
                metric=metric,
                observed_value=scale,
                threshold=threshold,
                reference_value=first,
                explanation="Large absolute scale expansion can result from mechanically repeated assumptions.",
            )
        )
    return tuple(result)


__all__ = ["CompoundingScaleRule"]
