"""Working-capital ratio and denominator-stability checks."""

from __future__ import annotations

from decimal import Decimal

from ..config import ForecastValidationConfig
from ..contracts import (
    HUNDRED,
    ZERO,
    ForecastValidationContext,
    Severity,
    ValidationCategory,
)
from ._utils import finding


class WorkingCapitalRule:
    name = "working_capital"

    def evaluate(
        self, context: ForecastValidationContext, config: ForecastValidationConfig
    ):
        findings = []
        previous_revenue: Decimal | None = None
        source_run = 0
        source_reported = False
        for row in context.rows:
            if row.delta_nwc is None:
                previous_revenue = None
                source_run = 0
                continue

            delta_nwc_ratio = None
            if (
                row.revenue is not None
                and abs(row.revenue) > config.near_zero_denominator
            ):
                delta_nwc_ratio = row.delta_nwc / abs(row.revenue) * HUNDRED
                if abs(delta_nwc_ratio) > config.max_delta_nwc_to_revenue_pct:
                    findings.append(
                        finding(
                            "EXTREME_DELTA_NWC_TO_REVENUE",
                            Severity.WARNING,
                            ValidationCategory.WORKING_CAPITAL,
                            f"ΔNWC is {delta_nwc_ratio}% of revenue in FY{row.fiscal_year}",
                            fiscal_year=row.fiscal_year,
                            metric="delta_nwc_to_revenue",
                            observed_value=delta_nwc_ratio,
                            threshold=config.max_delta_nwc_to_revenue_pct,
                            explanation="The working-capital flow is outside the broad configured revenue relationship.",
                        )
                    )

                large_source = (
                    row.delta_nwc < ZERO
                    and -delta_nwc_ratio >= config.max_working_capital_source_pct
                )
                source_run = source_run + 1 if large_source else 0
                if (
                    source_run >= config.persistent_working_capital_years
                    and not source_reported
                ):
                    findings.append(
                        finding(
                            "PERSISTENT_WORKING_CAPITAL_SOURCE",
                            Severity.WARNING,
                            ValidationCategory.WORKING_CAPITAL,
                            "Forecast assumes a persistently large working-capital source",
                            fiscal_year=row.fiscal_year,
                            metric="delta_nwc_to_revenue",
                            observed_value=delta_nwc_ratio,
                            threshold=config.max_working_capital_source_pct,
                            explanation="Large negative ΔNWC boosts FCFF and should have an explicit operating explanation.",
                        )
                    )
                    source_reported = True
            else:
                source_run = 0

            if previous_revenue is not None and row.revenue is not None:
                delta_revenue = row.revenue - previous_revenue
                if abs(delta_revenue) <= config.near_zero_denominator:
                    findings.append(
                        finding(
                            "WORKING_CAPITAL_NEAR_ZERO_DELTA_REVENUE",
                            Severity.WARNING,
                            ValidationCategory.WORKING_CAPITAL,
                            f"ΔNWC/ΔRevenue is unstable in FY{row.fiscal_year}",
                            fiscal_year=row.fiscal_year,
                            metric="delta_nwc_to_delta_revenue",
                            observed_value=delta_revenue,
                            threshold=config.near_zero_denominator,
                            explanation="The validator does not report a percentage with a near-zero revenue change.",
                        )
                    )
                else:
                    delta_ratio = row.delta_nwc / delta_revenue * HUNDRED
                    if abs(delta_ratio) > config.max_delta_nwc_to_delta_revenue_pct:
                        findings.append(
                            finding(
                                "EXTREME_DELTA_NWC_TO_DELTA_REVENUE",
                                Severity.WARNING,
                                ValidationCategory.WORKING_CAPITAL,
                                f"ΔNWC/ΔRevenue is {delta_ratio}% in FY{row.fiscal_year}",
                                fiscal_year=row.fiscal_year,
                                metric="delta_nwc_to_delta_revenue",
                                observed_value=delta_ratio,
                                threshold=config.max_delta_nwc_to_delta_revenue_pct,
                                explanation="This ratio is highly sensitive to small changes in revenue.",
                            )
                        )
            previous_revenue = row.revenue
        return tuple(findings)


__all__ = ["WorkingCapitalRule"]
