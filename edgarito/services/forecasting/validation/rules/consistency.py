"""Cross-metric economic and FCFF accounting identity checks."""

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
from ._utils import close_enough, finding


class CrossMetricConsistencyRule:
    name = "cross_metric_consistency"

    def evaluate(
        self, context: ForecastValidationContext, config: ForecastValidationConfig
    ):
        findings = []
        for row in context.rows:
            if (
                row.ebit is not None
                and row.gross_profit is not None
                and row.ebit > row.gross_profit
                and not (
                    row.other_operating_income is not None
                    and row.other_operating_income > ZERO
                )
            ):
                findings.append(
                    finding(
                        "EBIT_EXCEEDS_GROSS_PROFIT",
                        Severity.HIGH,
                        ValidationCategory.CONSISTENCY,
                        f"EBIT exceeds gross profit in FY{row.fiscal_year}",
                        fiscal_year=row.fiscal_year,
                        metric="ebit",
                        observed_value=row.ebit,
                        reference_value=row.gross_profit,
                        explanation="Operating profit normally cannot exceed the gross profit it is derived from.",
                    )
                )
            findings.extend(_nopat_findings(row, config))
            findings.extend(_fcff_identity_findings(row, config))

        terminal = context.terminal
        if terminal is not None:
            findings.extend(_terminal_fcff_identity_findings(terminal, config))
        return tuple(findings)


def _nopat_findings(row, config: ForecastValidationConfig):
    if row.ebit is None or row.nopat is None:
        return ()
    findings = []
    if row.tax_rate is not None:
        expected = row.ebit * (ONE - row.tax_rate / HUNDRED)
        if not close_enough(row.nopat, expected, config):
            findings.append(
                finding(
                    "NOPAT_TAX_IDENTITY_INCONSISTENT",
                    Severity.HIGH,
                    ValidationCategory.CONSISTENCY,
                    f"NOPAT does not match EBIT after the supplied tax rate in FY{row.fiscal_year}",
                    fiscal_year=row.fiscal_year,
                    metric="nopat",
                    observed_value=row.nopat,
                    threshold=config.fcff_identity_tolerance_pct,
                    reference_value=expected,
                    explanation="NOPAT is checked independently as EBIT × (1 - tax rate / 100).",
                )
            )
        if row.tax_rate >= ZERO and row.nopat > row.ebit and row.ebit >= ZERO:
            findings.append(
                finding(
                    "NOPAT_EXCEEDS_EBIT",
                    Severity.WARNING,
                    ValidationCategory.CONSISTENCY,
                    f"NOPAT exceeds EBIT despite a non-negative tax rate in FY{row.fiscal_year}",
                    fiscal_year=row.fiscal_year,
                    metric="nopat",
                    observed_value=row.nopat,
                    reference_value=row.ebit,
                    explanation="Positive tax rates should not increase positive EBIT into NOPAT.",
                )
            )
    elif row.tax is not None:
        expected = row.ebit - row.tax
        if not close_enough(row.nopat, expected, config):
            findings.append(
                finding(
                    "NOPAT_TAX_EXPENSE_INCONSISTENT",
                    Severity.WARNING,
                    ValidationCategory.CONSISTENCY,
                    f"NOPAT does not match EBIT less tax expense in FY{row.fiscal_year}",
                    fiscal_year=row.fiscal_year,
                    metric="nopat",
                    observed_value=row.nopat,
                    threshold=config.fcff_identity_tolerance_pct,
                    reference_value=expected,
                    explanation="With no tax rate, the supplied tax expense is used as the directional check.",
                )
            )
    return tuple(findings)


def _fcff_identity_findings(row, config: ForecastValidationConfig):
    required = (
        row.fcff,
        row.nopat,
        row.depreciation_and_amortization,
        row.capex,
        row.delta_nwc,
    )
    if any(value is None for value in required):
        return ()
    expected = row.nopat + row.depreciation_and_amortization - row.capex - row.delta_nwc
    if close_enough(row.fcff, expected, config):
        return ()
    return (
        finding(
            "FCFF_ACCOUNTING_IDENTITY_INCONSISTENT",
            Severity.HIGH,
            ValidationCategory.CONSISTENCY,
            f"FCFF does not match its accounting identity in FY{row.fiscal_year}",
            fiscal_year=row.fiscal_year,
            metric="fcff",
            observed_value=row.fcff,
            threshold=config.fcff_identity_tolerance_pct,
            reference_value=expected,
            explanation="Checked independently as NOPAT + D&A - CAPEX - ΔNWC.",
        ),
    )


def _terminal_fcff_identity_findings(terminal, config: ForecastValidationConfig):
    required = (
        terminal.terminal_fcff,
        terminal.terminal_nopat,
        terminal.terminal_da,
        terminal.terminal_capex,
        terminal.terminal_delta_nwc,
    )
    if any(value is None for value in required):
        return ()
    expected = (
        terminal.terminal_nopat
        + terminal.terminal_da
        - terminal.terminal_capex
        - terminal.terminal_delta_nwc
    )
    if close_enough(terminal.terminal_fcff, expected, config):
        return ()
    return (
        finding(
            "TERMINAL_FCFF_ACCOUNTING_IDENTITY_INCONSISTENT",
            Severity.HIGH,
            ValidationCategory.CONSISTENCY,
            "Terminal FCFF does not match its supplied accounting identity",
            metric="terminal_fcff",
            observed_value=terminal.terminal_fcff,
            threshold=config.fcff_identity_tolerance_pct,
            reference_value=expected,
            explanation="Checked independently as terminal NOPAT + D&A - CAPEX - ΔNWC.",
        ),
    )


__all__ = ["CrossMetricConsistencyRule"]
