"""Revenue/FCFF and operating-margin sanity checks."""

from __future__ import annotations

from decimal import Decimal

from ..config import ForecastValidationConfig
from ..contracts import (
    HUNDRED,
    ForecastValidationContext,
    Severity,
    ValidationCategory,
)
from ._utils import fcff_margin, finding, operating_margin, percent_ratio


class RevenueFcffMarginRule:
    name = "revenue_fcff_margin"

    def evaluate(
        self, context: ForecastValidationContext, config: ForecastValidationConfig
    ):
        findings = []
        margins: list[tuple[int, Decimal]] = []
        previous: tuple[int, Decimal] | None = None
        for row in context.rows:
            margin = fcff_margin(row, config)
            if margin is None:
                previous = None
                continue
            margins.append((row.fiscal_year, margin))
            if (
                margin > config.max_fcff_margin_pct
                or margin < config.min_fcff_margin_pct
            ):
                findings.append(
                    finding(
                        "EXTREME_FCFF_MARGIN",
                        Severity.HIGH,
                        ValidationCategory.MARGIN,
                        f"FCFF margin is {margin}% in FY{row.fiscal_year}",
                        fiscal_year=row.fiscal_year,
                        metric="fcff_margin",
                        observed_value=margin,
                        threshold=(
                            config.max_fcff_margin_pct
                            if margin > config.max_fcff_margin_pct
                            else config.min_fcff_margin_pct
                        ),
                        explanation="FCFF is outside the broad configured relationship with revenue.",
                    )
                )
            if previous is not None:
                change = margin - previous[1]
                if abs(change) > config.max_fcff_margin_change_points:
                    findings.append(
                        finding(
                            "FCFF_MARGIN_ABRUPT_CHANGE",
                            Severity.WARNING,
                            ValidationCategory.MARGIN,
                            f"FCFF margin changes by {change} percentage points from FY{previous[0]} to FY{row.fiscal_year}",
                            fiscal_year=row.fiscal_year,
                            metric="fcff_margin",
                            observed_value=change,
                            threshold=config.max_fcff_margin_change_points,
                            reference_value=previous[1],
                            explanation="A sharp margin movement can be an artifact of inconsistent FCFF or revenue assumptions.",
                        )
                    )
            if margin > HUNDRED:
                findings.append(
                    finding(
                        "FCFF_EXCEEDS_REVENUE",
                        Severity.HIGH,
                        ValidationCategory.MARGIN,
                        f"FCFF exceeds revenue in FY{row.fiscal_year}",
                        fiscal_year=row.fiscal_year,
                        metric="fcff_to_revenue",
                        observed_value=margin,
                        threshold=HUNDRED,
                        explanation="FCFF materially above revenue is an independent cross-check on the forecast bridge.",
                    )
                )
            previous = (row.fiscal_year, margin)

        range_margins = [
            margin
            for row in (*context.historical_rows, *context.rows)
            if (margin := fcff_margin(row, config)) is not None
        ]
        terminal = context.terminal
        if terminal is not None:
            terminal_margin = terminal.terminal_fcff_margin
            if terminal_margin is None:
                terminal_margin = percent_ratio(
                    terminal.terminal_fcff,
                    terminal.terminal_revenue,
                    config.near_zero_denominator,
                )
            if terminal_margin is not None:
                if terminal_margin > config.max_terminal_fcff_margin_pct:
                    findings.append(
                        finding(
                            "EXTREME_TERMINAL_FCFF_MARGIN",
                            Severity.HIGH,
                            ValidationCategory.MARGIN,
                            f"Terminal FCFF margin is {terminal_margin}%",
                            metric="terminal_fcff_margin",
                            observed_value=terminal_margin,
                            threshold=config.max_terminal_fcff_margin_pct,
                            explanation="The terminal FCFF/revenue relationship exceeds the configured sanity range.",
                        )
                    )
                if (
                    range_margins
                    and terminal_margin - max(range_margins)
                    > config.max_fcff_margin_change_points
                ):
                    findings.append(
                        finding(
                            "TERMINAL_FCFF_MARGIN_DISCONTINUITY",
                            Severity.WARNING,
                            ValidationCategory.DISCONTINUITY,
                            "Terminal FCFF margin is materially above the explicit forecast range",
                            metric="terminal_fcff_margin",
                            observed_value=terminal_margin,
                            threshold=config.max_fcff_margin_change_points,
                            reference_value=max(range_margins),
                            explanation="A terminal margin step-up should have an explicit economic rationale.",
                        )
                    )
        return tuple(findings)


class OperatingMarginRule:
    name = "operating_margin"

    def evaluate(
        self, context: ForecastValidationContext, config: ForecastValidationConfig
    ):
        findings = []
        margins: list[tuple[int, Decimal]] = []
        previous: tuple[int, Decimal] | None = None
        previous_change: Decimal | None = None
        expansion_run = 0
        expansion_reported = False
        for row in context.rows:
            margin = operating_margin(row, config)
            if margin is None:
                previous = None
                previous_change = None
                expansion_run = 0
                continue
            margins.append((row.fiscal_year, margin))
            if (
                margin > config.max_operating_margin_pct
                or margin < config.min_operating_margin_pct
            ):
                findings.append(
                    finding(
                        "IMPOSSIBLE_OPERATING_MARGIN",
                        Severity.HIGH,
                        ValidationCategory.OPERATING_MARGIN,
                        f"Operating margin is {margin}% in FY{row.fiscal_year}",
                        fiscal_year=row.fiscal_year,
                        metric="operating_margin",
                        observed_value=margin,
                        threshold=(
                            config.max_operating_margin_pct
                            if margin > config.max_operating_margin_pct
                            else config.min_operating_margin_pct
                        ),
                        explanation="Operating margin is outside the configured broad economic bounds.",
                    )
                )
            if previous is not None:
                change = margin - previous[1]
                if abs(change) > config.max_operating_margin_change_points:
                    findings.append(
                        finding(
                            "OPERATING_MARGIN_ABRUPT_CHANGE",
                            Severity.WARNING,
                            ValidationCategory.OPERATING_MARGIN,
                            f"Operating margin changes by {change} percentage points from FY{previous[0]} to FY{row.fiscal_year}",
                            fiscal_year=row.fiscal_year,
                            metric="operating_margin",
                            observed_value=change,
                            threshold=config.max_operating_margin_change_points,
                            reference_value=previous[1],
                            explanation="An abrupt operating-margin movement warrants an input and bridge review.",
                        )
                    )
                expanding = change >= config.mechanical_expansion_min_points and (
                    previous_change is None
                    or abs(change - previous_change)
                    <= config.mechanical_expansion_tolerance_points
                )
                expansion_run = expansion_run + 1 if expanding else 0
                if (
                    expansion_run >= config.mechanical_expansion_years
                    and not expansion_reported
                ):
                    findings.append(
                        finding(
                            "MECHANICAL_OPERATING_MARGIN_EXPANSION",
                            Severity.WARNING,
                            ValidationCategory.OPERATING_MARGIN,
                            f"Operating margin expands mechanically for {expansion_run} consecutive forecast changes",
                            fiscal_year=row.fiscal_year,
                            metric="operating_margin",
                            observed_value=margin,
                            threshold=config.mechanical_expansion_min_points,
                            explanation="Repeated similarly sized expansions may be an extrapolation artifact.",
                        )
                    )
                    expansion_reported = True
                previous_change = change
            previous = (row.fiscal_year, margin)

        terminal = context.terminal
        if terminal is not None and terminal.terminal_operating_margin is not None:
            terminal_margin = terminal.terminal_operating_margin
            if terminal_margin > config.max_terminal_operating_margin_pct:
                findings.append(
                    finding(
                        "EXTREME_TERMINAL_OPERATING_MARGIN",
                        Severity.HIGH,
                        ValidationCategory.OPERATING_MARGIN,
                        f"Terminal operating margin is {terminal_margin}%",
                        metric="terminal_operating_margin",
                        observed_value=terminal_margin,
                        threshold=config.max_terminal_operating_margin_pct,
                        explanation="The terminal operating margin exceeds the broad configured sanity range.",
                    )
                )
            range_margins = [
                margin
                for row in (*context.historical_rows, *context.rows)
                if (margin := operating_margin(row, config)) is not None
            ]
            if (
                range_margins
                and terminal_margin - max(range_margins)
                > config.terminal_margin_discontinuity_points
            ):
                findings.append(
                    finding(
                        "TERMINAL_OPERATING_MARGIN_DISCONTINUITY",
                        Severity.WARNING,
                        ValidationCategory.DISCONTINUITY,
                        "Terminal operating margin is materially above the explicit forecast range",
                        metric="terminal_operating_margin",
                        observed_value=terminal_margin,
                        threshold=config.terminal_margin_discontinuity_points,
                        reference_value=max(range_margins),
                        explanation="A terminal margin step-up should be supported by explicit economics.",
                    )
                )
        return tuple(findings)


__all__ = ["OperatingMarginRule", "RevenueFcffMarginRule"]
