"""Shared pure helpers for validation rules."""

from __future__ import annotations

from decimal import Decimal

from ..config import ForecastValidationConfig
from ..contracts import (
    HUNDRED,
    ONE,
    ZERO,
    ForecastValidationContext,
    ForecastValidationFinding,
    TerminalMetrics,
    ValidationCategory,
    ValidationSeverity,
)


def finding(
    code: str,
    severity: ValidationSeverity,
    category: ValidationCategory,
    message: str,
    *,
    fiscal_year: int | None = None,
    metric: str | None = None,
    observed_value: Decimal | None = None,
    threshold: Decimal | None = None,
    reference_value: Decimal | None = None,
    explanation: str | None = None,
) -> ForecastValidationFinding:
    return ForecastValidationFinding(
        code=code,
        severity=severity,
        category=category,
        message=message,
        fiscal_year=fiscal_year,
        metric=metric,
        observed_value=observed_value,
        threshold=threshold,
        reference_value=reference_value,
        explanation=explanation,
    )


def percent_ratio(
    numerator: Decimal | None,
    denominator: Decimal | None,
    near_zero: Decimal,
) -> Decimal | None:
    if numerator is None or denominator is None or abs(denominator) <= near_zero:
        return None
    return numerator / denominator * HUNDRED


def relative_change(
    current: Decimal | None,
    previous: Decimal | None,
    near_zero: Decimal,
) -> Decimal | None:
    if current is None or previous is None or abs(previous) <= near_zero:
        return None
    return (current - previous) / abs(previous) * HUNDRED


def close_enough(
    actual: Decimal,
    expected: Decimal,
    config: ForecastValidationConfig,
) -> bool:
    scale = max(abs(actual), abs(expected), ONE)
    return abs(actual - expected) <= max(
        config.absolute_identity_tolerance,
        scale * config.fcff_identity_tolerance_pct / HUNDRED,
    )


def operating_margin(row, config: ForecastValidationConfig) -> Decimal | None:
    if row.operating_margin is not None:
        return row.operating_margin
    return percent_ratio(row.ebit, row.revenue, config.near_zero_denominator)


def fcff_margin(row, config: ForecastValidationConfig) -> Decimal | None:
    return percent_ratio(row.fcff, row.revenue, config.near_zero_denominator)


def terminal_metrics(context: ForecastValidationContext) -> TerminalMetrics | None:
    return context.terminal


def positive_sign(value: Decimal) -> int:
    if value > ZERO:
        return 1
    if value < ZERO:
        return -1
    return 0


def severity(value: str) -> ValidationSeverity:
    return ValidationSeverity(value)


def has_consecutive_true(values: list[bool], length: int) -> bool:
    run = 0
    for value in values:
        run = run + 1 if value else 0
        if run >= length:
            return True
    return False


__all__ = [
    "HUNDRED",
    "ZERO",
    "close_enough",
    "fcff_margin",
    "finding",
    "has_consecutive_true",
    "operating_margin",
    "percent_ratio",
    "positive_sign",
    "relative_change",
    "severity",
    "terminal_metrics",
]
