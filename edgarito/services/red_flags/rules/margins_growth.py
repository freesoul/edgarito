from decimal import Decimal

from edgarito.config.red_flags import MarginsGrowthConfiguration
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
)
from edgarito.schemas.red_flags import (
    RedFlag,
    RedFlagCategory,
    RedFlagWarning,
)
from edgarito.services.red_flags.rules.context import PeriodKey


class _MarginsGrowthRules:
    def _check_margins_growth(
        self,
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]],
        periods: list[PeriodKey],
        config: MarginsGrowthConfiguration,
        flags: list[RedFlag],
        warnings: list[RedFlagWarning],
    ) -> None:
        if not config.enabled:
            return
        growth_count = 0
        margin_count = 0
        margin_decline_count = 0
        for index, period in enumerate(periods):
            current = by_period[period]
            previous = (
                by_period[periods[index - 1]]
                if index and self._consecutive_periods(periods[index - 1], period)
                else None
            )
            revenue = self._single(current, FinancialConcept.REVENUE)
            prior_revenue = (
                self._single(previous, FinancialConcept.REVENUE)
                if previous is not None
                else None
            )
            if (
                config.minimum_revenue_growth_pct is not None
                and revenue is not None
                and prior_revenue is not None
                and revenue.unit == prior_revenue.unit
                and prior_revenue.value > 0
            ):
                growth = (
                    (revenue.value - prior_revenue.value)
                    / prior_revenue.value
                    * Decimal(100)
                )
                growth_count += 1
                if growth < config.minimum_revenue_growth_pct:
                    self._add_flag(
                        flags,
                        code="revenue_growth_low",
                        category=RedFlagCategory.MARGINS_GROWTH,
                        severity=config.severity,
                        message=(
                            f"Revenue grew {self._format(growth)}%, below the configured "
                            f"{self._format(config.minimum_revenue_growth_pct)}% floor."
                        ),
                        evidence=self._evidence(
                            metric="revenue_growth",
                            value=growth,
                            unit="%",
                            threshold=config.minimum_revenue_growth_pct,
                            comparison="<",
                            formula="100 × (revenue - prior revenue) / prior revenue",
                            period=period,
                            values=(revenue, prior_revenue),
                        ),
                    )

            if (
                config.minimum_revenue_growth_pct is not None
                and index
                and (
                    revenue is None
                    or prior_revenue is None
                    or revenue.unit != prior_revenue.unit
                    or prior_revenue.value <= 0
                )
            ):
                self._period_warning(
                    warnings,
                    code="revenue_growth_period_unavailable",
                    category=RedFlagCategory.MARGINS_GROWTH,
                    period=period,
                    message=(
                        "Revenue growth could not be evaluated for this period "
                        "because compatible consecutive revenue periods were not "
                        "reported."
                    ),
                    required=(FinancialConcept.REVENUE,),
                )

            operating_income = self._single(current, FinancialConcept.OPERATING_INCOME)
            margin = self._ratio(operating_income, revenue, percentage=True)
            if margin is not None:
                margin_count += 1
                if (
                    config.minimum_operating_margin_pct is not None
                    and margin.value < config.minimum_operating_margin_pct
                ):
                    self._add_flag(
                        flags,
                        code="operating_margin_low",
                        category=RedFlagCategory.MARGINS_GROWTH,
                        severity=config.severity,
                        message=(
                            f"Operating margin was {self._format(margin.value)}%, below the configured "
                            f"{self._format(config.minimum_operating_margin_pct)}% floor."
                        ),
                        evidence=self._evidence(
                            metric="operating_margin",
                            value=margin.value,
                            unit="%",
                            threshold=config.minimum_operating_margin_pct,
                            comparison="<",
                            formula="100 × operating income / revenue",
                            period=period,
                            values=(margin,),
                        ),
                    )
            else:
                self._period_warning(
                    warnings,
                    code="operating_margin_period_unavailable",
                    category=RedFlagCategory.MARGINS_GROWTH,
                    period=period,
                    message=(
                        "Operating margin could not be evaluated for this period "
                        "because compatible revenue and operating income were not "
                        "reported."
                    ),
                    required=(
                        FinancialConcept.OPERATING_INCOME,
                        FinancialConcept.REVENUE,
                    ),
                )

            if index:
                prior_margin = self._ratio(
                    self._single(previous, FinancialConcept.OPERATING_INCOME),
                    prior_revenue,
                    percentage=True,
                )
                if (
                    previous is not None
                    and margin is not None
                    and prior_margin is not None
                ):
                    margin_decline_count += 1
                    decline = prior_margin.value - margin.value
                    if decline > config.maximum_operating_margin_decline_pp:
                        self._add_flag(
                            flags,
                            code="operating_margin_decline",
                            category=RedFlagCategory.MARGINS_GROWTH,
                            severity=config.severity,
                            message=(
                                f"Operating margin declined {self._format(decline)} percentage points, "
                                f"above the configured {self._format(config.maximum_operating_margin_decline_pp)}-point ceiling."
                            ),
                            evidence=self._evidence(
                                metric="operating_margin_decline",
                                value=decline,
                                unit="percentage_points",
                                threshold=config.maximum_operating_margin_decline_pp,
                                comparison=">",
                                formula="prior operating margin - current operating margin",
                                period=period,
                                values=(margin, prior_margin),
                            ),
                        )
                else:
                    self._period_warning(
                        warnings,
                        code="operating_margin_trend_period_unavailable",
                        category=RedFlagCategory.MARGINS_GROWTH,
                        period=period,
                        message=(
                            "Operating-margin trend could not be evaluated for this "
                            "period because compatible consecutive margins were not "
                            "reported."
                        ),
                        required=(
                            FinancialConcept.OPERATING_INCOME,
                            FinancialConcept.REVENUE,
                        ),
                    )
        if config.minimum_revenue_growth_pct is not None and growth_count == 0:
            self._warning(
                warnings,
                code="revenue_growth_unavailable",
                category=RedFlagCategory.MARGINS_GROWTH,
                message="Revenue growth was unavailable because compatible consecutive revenue periods were not reported.",
                required=(FinancialConcept.REVENUE,),
            )
        if margin_count == 0:
            self._warning(
                warnings,
                code="operating_margin_unavailable",
                category=RedFlagCategory.MARGINS_GROWTH,
                message="Operating margin was unavailable because compatible revenue and operating income were not reported.",
                required=(FinancialConcept.OPERATING_INCOME, FinancialConcept.REVENUE),
            )
        if margin_decline_count == 0:
            self._warning(
                warnings,
                code="operating_margin_trend_unavailable",
                category=RedFlagCategory.MARGINS_GROWTH,
                message="Operating-margin trend was unavailable because compatible consecutive margin periods were not reported.",
                required=(FinancialConcept.OPERATING_INCOME, FinancialConcept.REVENUE),
            )
