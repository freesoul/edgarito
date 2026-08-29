"""Terminal-rate, terminal-dependence, and explicit-to-terminal checks."""

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
from ._utils import finding, operating_margin, relative_change


class TerminalValueRule:
    name = "terminal_value"

    def evaluate(
        self, context: ForecastValidationContext, config: ForecastValidationConfig
    ):
        terminal = context.terminal
        if terminal is None:
            return ()
        findings = []
        growth = terminal.terminal_growth_rate
        if growth is not None and terminal.wacc is not None and growth >= terminal.wacc:
            findings.append(
                finding(
                    "TERMINAL_GROWTH_NOT_BELOW_WACC",
                    Severity.CRITICAL,
                    ValidationCategory.TERMINAL,
                    f"Terminal growth {growth}% is not below WACC {terminal.wacc}%",
                    metric="terminal_growth_rate",
                    observed_value=growth,
                    threshold=terminal.wacc,
                    reference_value=terminal.wacc,
                    explanation="A perpetual-growth denominator of WACC minus g must remain positive.",
                )
            )

        enterprise_value = terminal.enterprise_value
        terminal_pv = terminal.terminal_value_pv
        share = terminal.terminal_value_share_pct
        if share is None and terminal_pv is not None:
            if enterprise_value is None and terminal.explicit_forecast_pv is not None:
                enterprise_value = terminal_pv + terminal.explicit_forecast_pv
            if enterprise_value is not None and enterprise_value != ZERO:
                share = terminal_pv / enterprise_value * HUNDRED
        if share is not None and share >= config.terminal_value_share_warning_pct:
            findings.append(
                finding(
                    "HIGH_TERMINAL_VALUE_SHARE",
                    Severity.WARNING,
                    ValidationCategory.TERMINAL,
                    f"Terminal value contributes {share}% of enterprise value",
                    metric="terminal_value_share",
                    observed_value=share,
                    threshold=config.terminal_value_share_warning_pct,
                    explanation="A high terminal-value share is not automatically wrong, but makes valuation highly terminal-sensitive.",
                )
            )

        return tuple(findings)


class TerminalEconomicConsistencyRule:
    name = "terminal_economic_consistency"

    def evaluate(
        self, context: ForecastValidationContext, config: ForecastValidationConfig
    ):
        if context.terminal is None:
            return ()
        return _terminal_economic_findings(context.terminal, config)


class TerminalDiscontinuityRule:
    name = "terminal_discontinuity"

    def evaluate(
        self, context: ForecastValidationContext, config: ForecastValidationConfig
    ):
        return _explicit_terminal_findings(context, config)


def _terminal_economic_findings(terminal, config: ForecastValidationConfig):
    findings = []
    growth = terminal.terminal_growth_rate
    roic = terminal.terminal_roic
    if growth is None or roic is None or roic <= ZERO:
        return tuple(findings)

    expected_rate_pct = growth / roic * HUNDRED
    actual_rate_pct = terminal.terminal_reinvestment_rate
    reinvestment_supplied = (
        terminal.terminal_reinvestment_rate is not None
        or terminal.terminal_reinvestment is not None
    )
    if actual_rate_pct is None and terminal.terminal_reinvestment is not None:
        if (
            terminal.terminal_nopat is not None
            and abs(terminal.terminal_nopat) > config.near_zero_denominator
        ):
            actual_rate_pct = (
                terminal.terminal_reinvestment / abs(terminal.terminal_nopat) * HUNDRED
            )
    if actual_rate_pct is not None:
        if (
            abs(actual_rate_pct - expected_rate_pct)
            > config.terminal_identity_tolerance_pct
        ):
            findings.append(
                finding(
                    "TERMINAL_REINVESTMENT_IDENTITY_INCONSISTENT",
                    Severity.WARNING,
                    ValidationCategory.TERMINAL_ECONOMICS,
                    f"Terminal reinvestment rate {actual_rate_pct}% does not support g {growth}% at ROIC {roic}%",
                    metric="terminal_reinvestment_rate",
                    observed_value=actual_rate_pct,
                    threshold=config.terminal_identity_tolerance_pct,
                    reference_value=expected_rate_pct,
                    explanation="Terminal economics imply g = ROIC × reinvestment rate.",
                )
            )
        if (
            growth > config.minimum_terminal_reinvestment_pct
            and actual_rate_pct < config.minimum_terminal_reinvestment_pct
        ):
            findings.append(
                finding(
                    "TERMINAL_HIGH_GROWTH_LOW_REINVESTMENT",
                    Severity.WARNING,
                    ValidationCategory.TERMINAL_ECONOMICS,
                    "Terminal growth is positive while terminal reinvestment is nearly zero",
                    metric="terminal_reinvestment_rate",
                    observed_value=actual_rate_pct,
                    threshold=config.minimum_terminal_reinvestment_pct,
                    reference_value=expected_rate_pct,
                    explanation="Perpetual growth requires reinvestment when terminal ROIC is finite.",
                )
            )

    if terminal.terminal_nopat is not None and terminal.terminal_fcff is not None:
        expected_fcff = terminal.terminal_nopat * (ONE - expected_rate_pct / HUNDRED)
        scale = max(abs(terminal.terminal_fcff), abs(expected_fcff), ONE)
        tolerance = max(
            config.absolute_identity_tolerance,
            scale * config.terminal_identity_tolerance_pct / HUNDRED,
        )
        if abs(terminal.terminal_fcff - expected_fcff) > tolerance:
            findings.append(
                finding(
                    "TERMINAL_FCFF_REINVESTMENT_IDENTITY_INCONSISTENT",
                    Severity.WARNING,
                    ValidationCategory.TERMINAL_ECONOMICS,
                    "Terminal FCFF does not match NOPAT after implied reinvestment",
                    metric="terminal_fcff",
                    observed_value=terminal.terminal_fcff,
                    threshold=tolerance,
                    reference_value=expected_fcff,
                    explanation="The supplied terminal values do not satisfy FCFF = NOPAT × (1 - g / ROIC).",
                )
            )
        implied_rate = None
        if abs(terminal.terminal_nopat) > config.near_zero_denominator:
            implied_rate = (
                (terminal.terminal_nopat - terminal.terminal_fcff)
                / abs(terminal.terminal_nopat)
                * HUNDRED
            )
        if (
            implied_rate is not None
            and not reinvestment_supplied
            and growth > config.minimum_terminal_reinvestment_pct
            and implied_rate < config.minimum_terminal_reinvestment_pct
        ):
            findings.append(
                finding(
                    "TERMINAL_HIGH_GROWTH_LOW_REINVESTMENT",
                    Severity.WARNING,
                    ValidationCategory.TERMINAL_ECONOMICS,
                    "Terminal FCFF implies almost no reinvestment despite positive perpetual growth",
                    metric="terminal_reinvestment_rate",
                    observed_value=implied_rate,
                    threshold=config.minimum_terminal_reinvestment_pct,
                    reference_value=expected_rate_pct,
                    explanation="The implied terminal cash conversion is inconsistent with the supplied growth and ROIC.",
                )
            )
    return tuple(findings)


def _explicit_terminal_findings(
    context: ForecastValidationContext, config: ForecastValidationConfig
):
    terminal = context.terminal
    if terminal is None or not context.rows:
        return ()
    findings = []
    if terminal.terminal_growth_rate is not None:
        fcff_rows = [row for row in context.rows if row.fcff is not None]
        if len(fcff_rows) >= 2:
            previous, current = fcff_rows[-2], fcff_rows[-1]
            growth = (
                relative_change(
                    current.fcff,
                    previous.fcff,
                    config.near_zero_denominator,
                )
                if current.fiscal_year == previous.fiscal_year + 1
                else None
            )
            if (
                growth is not None
                and growth - terminal.terminal_growth_rate
                > config.terminal_growth_stepdown_points
            ):
                findings.append(
                    finding(
                        "TERMINAL_FCFF_GROWTH_DISCONTINUITY",
                        Severity.WARNING,
                        ValidationCategory.DISCONTINUITY,
                        "Terminal FCFF growth steps down sharply from the final explicit period",
                        fiscal_year=current.fiscal_year,
                        metric="fcff_growth",
                        observed_value=growth,
                        threshold=config.terminal_growth_stepdown_points,
                        reference_value=terminal.terminal_growth_rate,
                        explanation="A sharp transition may be valid, but should be explicit rather than accidental.",
                    )
                )

    margin_rows = [
        row
        for row in reversed(context.rows)
        if operating_margin(row, config) is not None
    ]
    explicit_margin = operating_margin(margin_rows[0], config) if margin_rows else None
    if (
        explicit_margin is not None
        and terminal.terminal_operating_margin is not None
        and abs(terminal.terminal_operating_margin - explicit_margin)
        > config.terminal_margin_discontinuity_points
    ):
        findings.append(
            finding(
                "TERMINAL_OPERATING_MARGIN_STEP",
                Severity.WARNING,
                ValidationCategory.DISCONTINUITY,
                "Terminal operating margin changes sharply from the final explicit year",
                metric="terminal_operating_margin",
                observed_value=terminal.terminal_operating_margin,
                threshold=config.terminal_margin_discontinuity_points,
                reference_value=explicit_margin,
                explanation="Terminal profitability should transition transparently from the explicit forecast.",
            )
        )

    reinvestment_rows = [
        row for row in reversed(context.rows) if row.reinvestment_rate is not None
    ]
    if terminal.terminal_reinvestment_rate is not None and reinvestment_rows:
        explicit_reinvestment_rate = reinvestment_rows[0].reinvestment_rate
        terminal_rate = terminal.terminal_reinvestment_rate
        if (
            abs(terminal_rate - explicit_reinvestment_rate)
            > config.terminal_reinvestment_discontinuity_points
        ):
            findings.append(
                finding(
                    "TERMINAL_REINVESTMENT_STEP",
                    Severity.WARNING,
                    ValidationCategory.DISCONTINUITY,
                    "Terminal reinvestment rate changes sharply from the final explicit year",
                    metric="terminal_reinvestment_rate",
                    observed_value=terminal_rate,
                    threshold=config.terminal_reinvestment_discontinuity_points,
                    reference_value=explicit_reinvestment_rate,
                    explanation="A terminal investment change should be visible in the valuation assumptions.",
                )
            )
    return tuple(findings)


__all__ = [
    "TerminalDiscontinuityRule",
    "TerminalEconomicConsistencyRule",
    "TerminalValueRule",
]
