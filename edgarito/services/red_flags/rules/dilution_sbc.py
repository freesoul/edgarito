from decimal import Decimal

from edgarito.config.red_flags import DilutionSbcConfiguration
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


class _DilutionSbcRules:
    def _check_dilution_sbc(
        self,
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]],
        periods: list[PeriodKey],
        config: DilutionSbcConfiguration,
        flags: list[RedFlag],
        warnings: list[RedFlagWarning],
    ) -> None:
        if not config.enabled:
            return
        share_growth_count = 0
        dilution_count = 0
        sbc_count = 0
        for index, period in enumerate(periods):
            current = by_period[period]
            previous = (
                by_period[periods[index - 1]]
                if index and self._consecutive_periods(periods[index - 1], period)
                else None
            )
            shares = self._share_count(current)
            previous_shares = (
                self._share_count(previous) if previous is not None else None
            )
            if (
                shares is not None
                and previous_shares is not None
                and shares.unit == previous_shares.unit
                and previous_shares.value > 0
            ):
                growth = (
                    (shares.value - previous_shares.value)
                    / previous_shares.value
                    * Decimal(100)
                )
                share_growth_count += 1
                if growth > config.maximum_share_count_growth_pct:
                    self._add_flag(
                        flags,
                        code="share_count_growth_high",
                        category=RedFlagCategory.DILUTION_SBC,
                        severity=config.severity,
                        message=(
                            f"Share count grew {self._format(growth)}%, above the configured "
                            f"{self._format(config.maximum_share_count_growth_pct)}% ceiling."
                        ),
                        evidence=self._evidence(
                            metric="share_count_growth",
                            value=growth,
                            unit="%",
                            threshold=config.maximum_share_count_growth_pct,
                            comparison=">",
                            formula="100 × (current shares - prior shares) / prior shares",
                            period=period,
                            values=(shares, previous_shares),
                        ),
                    )
            elif index:
                self._period_warning(
                    warnings,
                    code="share_count_growth_period_unavailable",
                    category=RedFlagCategory.DILUTION_SBC,
                    period=period,
                    message=(
                        "Share-count growth could not be evaluated for this period "
                        "because compatible consecutive share counts were not "
                        "reported."
                    ),
                    required=(
                        FinancialConcept.SHARES_OUTSTANDING,
                        FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES,
                    ),
                )

            basic = self._single(
                current, FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES
            )
            diluted = self._single(
                current, FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES
            )
            if (
                basic is not None
                and diluted is not None
                and basic.unit == diluted.unit
                and basic.value > 0
            ):
                premium = (diluted.value - basic.value) / basic.value * Decimal(100)
                dilution_count += 1
                if premium > config.maximum_diluted_share_premium_pct:
                    self._add_flag(
                        flags,
                        code="diluted_share_premium_high",
                        category=RedFlagCategory.DILUTION_SBC,
                        severity=config.severity,
                        message=(
                            f"Diluted shares were {self._format(premium)}% above basic shares, "
                            f"above the configured {self._format(config.maximum_diluted_share_premium_pct)}% ceiling."
                        ),
                        evidence=self._evidence(
                            metric="diluted_share_premium",
                            value=premium,
                            unit="%",
                            threshold=config.maximum_diluted_share_premium_pct,
                            comparison=">",
                            formula="100 × (diluted shares - basic shares) / basic shares",
                            period=period,
                            values=(diluted, basic),
                        ),
                    )
            else:
                self._period_warning(
                    warnings,
                    code="diluted_share_premium_period_unavailable",
                    category=RedFlagCategory.DILUTION_SBC,
                    period=period,
                    message=(
                        "Diluted-share premium could not be evaluated for this "
                        "period because compatible basic and diluted weighted-average "
                        "share counts were not reported."
                    ),
                    required=(
                        FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES,
                        FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES,
                    ),
                )

            sbc = self._single(current, FinancialConcept.STOCK_BASED_COMPENSATION)
            revenue = self._single(current, FinancialConcept.REVENUE)
            if sbc is not None and revenue is not None and sbc.unit == revenue.unit:
                if revenue.value != 0:
                    sbc_ratio = abs(sbc.value) / abs(revenue.value) * Decimal(100)
                    sbc_count += 1
                    if sbc_ratio > config.maximum_sbc_to_revenue_pct:
                        self._add_flag(
                            flags,
                            code="stock_based_compensation_high",
                            category=RedFlagCategory.DILUTION_SBC,
                            severity=config.severity,
                            message=(
                                f"Stock-based compensation was {self._format(sbc_ratio)}% of revenue, "
                                f"above the configured {self._format(config.maximum_sbc_to_revenue_pct)}% ceiling."
                            ),
                            evidence=self._evidence(
                                metric="stock_based_compensation_to_revenue",
                                value=sbc_ratio,
                                unit="%",
                                threshold=config.maximum_sbc_to_revenue_pct,
                                comparison=">",
                                formula="100 × absolute stock-based compensation / revenue",
                                period=period,
                                values=(sbc, revenue),
                            ),
                        )
            else:
                self._period_warning(
                    warnings,
                    code="stock_based_compensation_period_unavailable",
                    category=RedFlagCategory.DILUTION_SBC,
                    period=period,
                    message=(
                        "Stock-based compensation as a share of revenue could not "
                        "be evaluated for this period because compatible SBC and "
                        "revenue were not reported."
                    ),
                    required=(
                        FinancialConcept.STOCK_BASED_COMPENSATION,
                        FinancialConcept.REVENUE,
                    ),
                )
        if share_growth_count == 0:
            self._warning(
                warnings,
                code="share_count_growth_unavailable",
                category=RedFlagCategory.DILUTION_SBC,
                message=(
                    "Share-count growth was unavailable because current and prior "
                    "normalized share counts did not overlap."
                ),
                required=(FinancialConcept.SHARES_OUTSTANDING,),
            )
        if dilution_count == 0:
            self._warning(
                warnings,
                code="diluted_share_premium_unavailable",
                category=RedFlagCategory.DILUTION_SBC,
                message=(
                    "Diluted-share premium was unavailable because compatible basic "
                    "and diluted weighted-average share counts were not reported."
                ),
                required=(
                    FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES,
                    FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES,
                ),
            )
        if sbc_count == 0:
            self._warning(
                warnings,
                code="stock_based_compensation_unavailable",
                category=RedFlagCategory.DILUTION_SBC,
                message=(
                    "Stock-based compensation as a share of revenue was unavailable "
                    "because normalized SBC and revenue did not overlap."
                ),
                required=(
                    FinancialConcept.STOCK_BASED_COMPENSATION,
                    FinancialConcept.REVENUE,
                ),
            )
