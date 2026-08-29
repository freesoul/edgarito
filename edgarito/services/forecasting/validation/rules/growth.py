"""FCFF year-over-year growth and sign stability checks."""

from __future__ import annotations

from ..config import ForecastValidationConfig
from ..contracts import HUNDRED, ForecastValidationContext, Severity, ValidationCategory
from ._utils import finding, positive_sign


class FcffGrowthRule:
    name = "fcff_growth"

    def evaluate(
        self, context: ForecastValidationContext, config: ForecastValidationConfig
    ):
        findings = []
        hypergrowth_run = 0
        repeated_reported = False
        previous_row = None
        for row in context.rows:
            current = row.fcff
            if previous_row is None or current is None or previous_row.fcff is None:
                previous_row = row
                hypergrowth_run = 0
                continue
            if row.fiscal_year != previous_row.fiscal_year + 1:
                previous_row = row
                hypergrowth_run = 0
                continue

            previous = previous_row.fcff
            previous_sign = positive_sign(previous)
            current_sign = positive_sign(current)
            if previous_sign * current_sign == -1:
                findings.append(
                    finding(
                        "SIGN_TRANSITION",
                        Severity.WARNING,
                        ValidationCategory.GROWTH,
                        f"FCFF changes sign from FY{previous_row.fiscal_year} to FY{row.fiscal_year}",
                        fiscal_year=row.fiscal_year,
                        metric="fcff",
                        observed_value=current,
                        reference_value=previous,
                        explanation="Percentage growth is not meaningful across a sign transition.",
                    )
                )
                hypergrowth_run = 0
                previous_row = row
                continue

            if abs(previous) <= config.near_zero_denominator:
                findings.append(
                    finding(
                        "FCFF_GROWTH_NEAR_ZERO_DENOMINATOR",
                        Severity.WARNING,
                        ValidationCategory.GROWTH,
                        f"FCFF growth denominator is near zero in FY{previous_row.fiscal_year}",
                        fiscal_year=row.fiscal_year,
                        metric="fcff",
                        observed_value=previous,
                        threshold=config.near_zero_denominator,
                        explanation="The validator reports denominator instability instead of a misleading percentage.",
                    )
                )
                hypergrowth_run = 0
                previous_row = row
                continue

            growth_pct = (current - previous) / abs(previous) * HUNDRED
            magnitude_change_pct = abs(current - previous) / abs(previous) * HUNDRED
            is_hypergrowth = growth_pct >= config.max_fcff_growth_pct
            hypergrowth_run = hypergrowth_run + 1 if is_hypergrowth else 0
            if is_hypergrowth:
                findings.append(
                    finding(
                        "FCFF_GROWTH_EXTREME",
                        Severity.WARNING,
                        ValidationCategory.GROWTH,
                        f"FCFF grows {growth_pct}% from FY{previous_row.fiscal_year} to FY{row.fiscal_year}",
                        fiscal_year=row.fiscal_year,
                        metric="fcff_growth",
                        observed_value=growth_pct,
                        threshold=config.max_fcff_growth_pct,
                        explanation="Large year-over-year FCFF growth can indicate mechanical extrapolation.",
                    )
                )
            if magnitude_change_pct >= config.max_fcff_jump_pct:
                findings.append(
                    finding(
                        "FCFF_GROWTH_ABRUPT_JUMP",
                        Severity.WARNING,
                        ValidationCategory.GROWTH,
                        f"FCFF changes by {magnitude_change_pct}% between consecutive forecast years",
                        fiscal_year=row.fiscal_year,
                        metric="fcff",
                        observed_value=magnitude_change_pct,
                        threshold=config.max_fcff_jump_pct,
                        explanation="The size of the jump warrants an explicit driver or bridge review.",
                    )
                )
            if (
                hypergrowth_run >= config.repeated_hypergrowth_years
                and not repeated_reported
            ):
                findings.append(
                    finding(
                        "FCFF_REPEATED_HYPERGROWTH",
                        Severity.WARNING,
                        ValidationCategory.GROWTH,
                        f"FCFF exceeds the hypergrowth threshold for {hypergrowth_run} consecutive changes",
                        fiscal_year=row.fiscal_year,
                        metric="fcff_growth",
                        observed_value=growth_pct,
                        threshold=config.max_fcff_growth_pct,
                        explanation="Repeated high growth compounds quickly and should be visible to valuation reviewers.",
                    )
                )
                repeated_reported = True
            previous_row = row
        return tuple(findings)


__all__ = ["FcffGrowthRule"]
