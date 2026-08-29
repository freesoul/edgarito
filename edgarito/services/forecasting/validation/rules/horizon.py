"""Forecast-year ordering and horizon checks."""

from __future__ import annotations

from ..config import ForecastValidationConfig
from ..contracts import ForecastValidationContext, Severity, ValidationCategory
from ._utils import finding


class HorizonIntegrityRule:
    name = "horizon_integrity"

    def evaluate(
        self, context: ForecastValidationContext, config: ForecastValidationConfig
    ):
        rows = context.rows
        if not rows:
            if not context.rows_supplied:
                return ()
            return (
                finding(
                    "EMPTY_FORECAST",
                    Severity.HIGH,
                    ValidationCategory.HORIZON,
                    "The explicit forecast contains no year rows",
                    explanation="Horizon-dependent validation cannot establish a forecast path.",
                ),
            )

        findings = []
        years = [row.fiscal_year for row in rows]
        duplicates = sorted({year for year in years if years.count(year) > 1})
        for year in duplicates:
            findings.append(
                finding(
                    "DUPLICATE_FORECAST_YEAR",
                    Severity.HIGH,
                    ValidationCategory.HORIZON,
                    f"Forecast year FY{year} appears more than once",
                    fiscal_year=year,
                    metric="fiscal_year",
                    observed_value=year,
                    explanation="Each explicit fiscal year should have one canonical row.",
                )
            )

        non_monotonic_at = next(
            (
                index
                for index, (previous, current) in enumerate(
                    zip(years, years[1:], strict=False)
                )
                if current <= previous
            ),
            None,
        )
        if non_monotonic_at is not None:
            previous = years[non_monotonic_at]
            current = years[non_monotonic_at + 1]
            findings.append(
                finding(
                    "NON_MONOTONIC_FORECAST_YEARS",
                    Severity.HIGH,
                    ValidationCategory.HORIZON,
                    f"Forecast years are not strictly increasing: FY{previous} then FY{current}",
                    fiscal_year=current,
                    metric="fiscal_year",
                    observed_value=current,
                    reference_value=previous,
                    explanation="Year-over-year rules require an ordered explicit path.",
                )
            )

        missing = sorted(
            {
                year
                for previous, current in zip(years, years[1:], strict=False)
                if current > previous + 1
                for year in range(previous + 1, current)
            }
        )
        for year in missing:
            findings.append(
                finding(
                    "MISSING_FORECAST_YEAR",
                    Severity(config.missing_year_severity),
                    ValidationCategory.HORIZON,
                    f"Forecast path skips FY{year}",
                    fiscal_year=year,
                    metric="fiscal_year",
                    observed_value=year,
                    explanation="A skipped year can make period-to-period changes misleading.",
                )
            )

        if len(rows) < config.minimum_explicit_years:
            findings.append(
                finding(
                    "SHORT_FORECAST_HORIZON",
                    Severity.WARNING,
                    ValidationCategory.HORIZON,
                    f"Explicit forecast horizon has only {len(rows)} year(s)",
                    metric="forecast_horizon",
                    observed_value=len(rows),
                    threshold=config.minimum_explicit_years,
                    explanation="A short path limits trend and discontinuity diagnostics.",
                )
            )
        if len(rows) > config.maximum_explicit_years:
            findings.append(
                finding(
                    "LONG_FORECAST_HORIZON",
                    Severity.WARNING,
                    ValidationCategory.HORIZON,
                    f"Explicit forecast horizon has {len(rows)} years",
                    metric="forecast_horizon",
                    observed_value=len(rows),
                    threshold=config.maximum_explicit_years,
                    explanation="Long explicit paths are especially sensitive to mechanical assumptions.",
                )
            )
        return tuple(findings)


__all__ = ["HorizonIntegrityRule"]
