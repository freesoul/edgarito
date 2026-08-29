"""CAPEX, D&A, and generic reinvestment relationship checks."""

from __future__ import annotations

from decimal import Decimal

from ..config import ForecastValidationConfig
from ..contracts import HUNDRED, ForecastValidationContext, Severity, ValidationCategory
from ._utils import finding, percent_ratio


class ReinvestmentRule:
    name = "reinvestment"

    def evaluate(
        self, context: ForecastValidationContext, config: ForecastValidationConfig
    ):
        findings = []
        previous_revenue: Decimal | None = None
        previous_revenue_year: int | None = None
        previous_capex_ratio: tuple[int, Decimal] | None = None
        negligible_high_growth_run = 0
        reported_negligible_run = False
        for row in context.rows:
            capex_ratio = percent_ratio(
                row.capex, row.revenue, config.near_zero_denominator
            )
            da_ratio = percent_ratio(
                row.depreciation_and_amortization,
                row.revenue,
                config.near_zero_denominator,
            )
            if (
                capex_ratio is not None
                and capex_ratio > config.max_capex_to_revenue_pct
            ):
                findings.append(
                    finding(
                        "EXTREME_CAPEX_TO_REVENUE",
                        Severity.WARNING,
                        ValidationCategory.REINVESTMENT,
                        f"CAPEX is {capex_ratio}% of revenue in FY{row.fiscal_year}",
                        fiscal_year=row.fiscal_year,
                        metric="capex_to_revenue",
                        observed_value=capex_ratio,
                        threshold=config.max_capex_to_revenue_pct,
                        explanation="CAPEX is outside the broad configured revenue relationship.",
                    )
                )
            if da_ratio is not None and da_ratio > config.max_da_to_revenue_pct:
                findings.append(
                    finding(
                        "EXTREME_DA_TO_REVENUE",
                        Severity.WARNING,
                        ValidationCategory.REINVESTMENT,
                        f"D&A is {da_ratio}% of revenue in FY{row.fiscal_year}",
                        fiscal_year=row.fiscal_year,
                        metric="da_to_revenue",
                        observed_value=da_ratio,
                        threshold=config.max_da_to_revenue_pct,
                        explanation="D&A is outside the broad configured revenue relationship.",
                    )
                )

            if row.capex is not None and row.depreciation_and_amortization is not None:
                if (
                    abs(row.depreciation_and_amortization)
                    <= config.near_zero_denominator
                ):
                    capex_to_da = None
                else:
                    capex_to_da = row.capex / row.depreciation_and_amortization
                if capex_to_da is not None and (
                    capex_to_da > config.max_capex_to_da_ratio
                    or capex_to_da < config.min_capex_to_da_ratio
                ):
                    findings.append(
                        finding(
                            "EXTREME_CAPEX_TO_DA",
                            Severity.WARNING,
                            ValidationCategory.REINVESTMENT,
                            f"CAPEX/D&A is {capex_to_da} in FY{row.fiscal_year}",
                            fiscal_year=row.fiscal_year,
                            metric="capex_to_da",
                            observed_value=capex_to_da,
                            threshold=config.max_capex_to_da_ratio,
                            explanation="CAPEX and D&A have an unusually wide relationship.",
                        )
                    )
                elif (
                    abs(row.capex) <= config.near_zero_denominator
                    and row.depreciation_and_amortization > config.near_zero_denominator
                ):
                    findings.append(
                        finding(
                            "DA_WITHOUT_CAPEX",
                            Severity.WARNING,
                            ValidationCategory.REINVESTMENT,
                            f"D&A is positive while CAPEX is near zero in FY{row.fiscal_year}",
                            fiscal_year=row.fiscal_year,
                            metric="capex_to_da",
                            observed_value=row.capex,
                            threshold=config.near_zero_denominator,
                            explanation="Persistently negligible replacement investment can make FCFF mechanically optimistic.",
                        )
                    )
                elif abs(row.capex) > config.near_zero_denominator:
                    da_to_capex = row.depreciation_and_amortization / row.capex
                    if da_to_capex > config.max_da_to_capex_ratio:
                        findings.append(
                            finding(
                                "EXTREME_DA_TO_CAPEX",
                                Severity.WARNING,
                                ValidationCategory.REINVESTMENT,
                                f"D&A/CAPEX is {da_to_capex} in FY{row.fiscal_year}",
                                fiscal_year=row.fiscal_year,
                                metric="da_to_capex",
                                observed_value=da_to_capex,
                                threshold=config.max_da_to_capex_ratio,
                                explanation="D&A materially exceeds current reinvestment for this forecast year.",
                            )
                        )

            growth_pct = None
            if previous_revenue is not None and row.revenue is not None:
                if (
                    previous_revenue_year is not None
                    and row.fiscal_year == previous_revenue_year + 1
                    and abs(previous_revenue) > config.near_zero_denominator
                ):
                    growth_pct = (
                        (row.revenue - previous_revenue)
                        / abs(previous_revenue)
                        * HUNDRED
                    )
            if (
                growth_pct is not None
                and capex_ratio is not None
                and growth_pct >= config.high_growth_for_reinvestment_pct
                and capex_ratio <= config.negligible_capex_to_revenue_pct
            ):
                negligible_high_growth_run += 1
            else:
                negligible_high_growth_run = 0
            if (
                negligible_high_growth_run >= config.persistent_reinvestment_years
                and not reported_negligible_run
            ):
                findings.append(
                    finding(
                        "HIGH_GROWTH_NEGLIGIBLE_CAPEX",
                        Severity.WARNING,
                        ValidationCategory.REINVESTMENT,
                        "Revenue grows rapidly while CAPEX remains persistently negligible",
                        fiscal_year=row.fiscal_year,
                        metric="capex_to_revenue",
                        observed_value=capex_ratio,
                        threshold=config.negligible_capex_to_revenue_pct,
                        explanation="This generic warning does not infer industry capital intensity; it highlights a potentially mechanical path.",
                    )
                )
                reported_negligible_run = True

            if previous_capex_ratio is not None and capex_ratio is not None:
                collapse = previous_capex_ratio[1] - capex_ratio
                if (
                    collapse > config.max_capex_ratio_collapse_points
                    and growth_pct is not None
                    and growth_pct >= config.high_growth_for_reinvestment_pct
                ):
                    findings.append(
                        finding(
                            "CAPEX_RATIO_COLLAPSE_DURING_GROWTH",
                            Severity.WARNING,
                            ValidationCategory.REINVESTMENT,
                            f"CAPEX/revenue falls by {collapse} percentage points during high revenue growth",
                            fiscal_year=row.fiscal_year,
                            metric="capex_to_revenue",
                            observed_value=collapse,
                            threshold=config.max_capex_ratio_collapse_points,
                            reference_value=previous_capex_ratio[1],
                            explanation="A large investment-ratio collapse can make later FCFF look overstated.",
                        )
                    )

            previous_revenue = row.revenue
            previous_revenue_year = row.fiscal_year
            if capex_ratio is not None:
                previous_capex_ratio = (row.fiscal_year, capex_ratio)
            else:
                previous_capex_ratio = None
        return tuple(findings)


__all__ = ["ReinvestmentRule"]
