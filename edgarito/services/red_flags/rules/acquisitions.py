from decimal import Decimal

from edgarito.config.red_flags import AcquisitionsConfiguration
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


class _AcquisitionsRules:
    def _check_acquisitions(
        self,
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]],
        periods: list[PeriodKey],
        config: AcquisitionsConfiguration,
        flags: list[RedFlag],
        warnings: list[RedFlagWarning],
    ) -> None:
        if not config.enabled:
            return
        spend_count = 0
        acquisition_to_fcf_count = 0
        goodwill_count = 0
        goodwill_trend_count = 0
        for index, period in enumerate(periods):
            current = by_period[period]
            previous = (
                by_period[periods[index - 1]]
                if index and self._consecutive_periods(periods[index - 1], period)
                else None
            )
            acquisition = self._single(current, FinancialConcept.ACQUISITION_CASH_PAID)
            revenue = self._single(current, FinancialConcept.REVENUE)
            fcf = self._free_cash_flow(current)
            if (
                acquisition is not None
                and revenue is not None
                and acquisition.unit == revenue.unit
            ):
                if revenue.value != 0:
                    spend_count += 1
                    acquisition_to_revenue = (
                        abs(acquisition.value) / abs(revenue.value) * Decimal(100)
                    )
                    if (
                        acquisition_to_revenue
                        > config.maximum_acquisition_to_revenue_pct
                    ):
                        self._add_flag(
                            flags,
                            code="acquisition_spend_high",
                            category=RedFlagCategory.ACQUISITIONS,
                            severity=config.severity,
                            message=(
                                f"Acquisition cash spending was {self._format(acquisition_to_revenue)}% of revenue, "
                                f"above the configured {self._format(config.maximum_acquisition_to_revenue_pct)}% ceiling."
                            ),
                            evidence=self._evidence(
                                metric="acquisition_cash_to_revenue",
                                value=acquisition_to_revenue,
                                unit="%",
                                threshold=config.maximum_acquisition_to_revenue_pct,
                                comparison=">",
                                formula="100 × absolute acquisition cash paid / revenue",
                                period=period,
                                values=(acquisition, revenue),
                            ),
                        )
                else:
                    self._period_warning(
                        warnings,
                        code="acquisition_to_revenue_period_unavailable",
                        category=RedFlagCategory.ACQUISITIONS,
                        period=period,
                        message=(
                            "Acquisition spending versus revenue could not be "
                            "evaluated for this period because revenue was zero."
                        ),
                        required=(
                            FinancialConcept.ACQUISITION_CASH_PAID,
                            FinancialConcept.REVENUE,
                        ),
                    )
            else:
                self._period_warning(
                    warnings,
                    code="acquisition_to_revenue_period_unavailable",
                    category=RedFlagCategory.ACQUISITIONS,
                    period=period,
                    message=(
                        "Acquisition spending versus revenue could not be evaluated "
                        "for this period because compatible acquisition cash paid and "
                        "revenue were not reported."
                    ),
                    required=(
                        FinancialConcept.ACQUISITION_CASH_PAID,
                        FinancialConcept.REVENUE,
                    ),
                )
            if (
                acquisition is not None
                and fcf is not None
                and fcf.unit == acquisition.unit
                and fcf.value > 0
            ):
                acquisition_to_fcf_count += 1
                acquisition_to_fcf = abs(acquisition.value) / fcf.value * Decimal(100)
                if acquisition_to_fcf > config.maximum_acquisition_to_fcf_pct:
                    self._add_flag(
                        flags,
                        code="acquisition_spend_exceeds_fcf",
                        category=RedFlagCategory.ACQUISITIONS,
                        severity=config.severity,
                        message=(
                            f"Acquisition cash spending was {self._format(acquisition_to_fcf)}% of FCF, "
                            f"above the configured {self._format(config.maximum_acquisition_to_fcf_pct)}% ceiling."
                        ),
                        evidence=self._evidence(
                            metric="acquisition_cash_to_fcf",
                            value=acquisition_to_fcf,
                            unit="%",
                            threshold=config.maximum_acquisition_to_fcf_pct,
                            comparison=">",
                            formula="100 × absolute acquisition cash paid / free cash flow",
                            period=period,
                            values=(acquisition, fcf),
                        ),
                    )
            else:
                self._period_warning(
                    warnings,
                    code="acquisition_to_fcf_period_unavailable",
                    category=RedFlagCategory.ACQUISITIONS,
                    period=period,
                    message=(
                        "Acquisition spending versus FCF could not be evaluated "
                        "because compatible acquisition cash paid and positive FCF "
                        "were not reported."
                    ),
                    required=self._ACQUISITION_TO_FCF_CONCEPTS,
                )

            goodwill = self._single(current, FinancialConcept.GOODWILL)
            prior_goodwill = (
                self._single(previous, FinancialConcept.GOODWILL)
                if previous is not None
                else None
            )
            if (
                goodwill is not None
                and prior_goodwill is not None
                and goodwill.unit == prior_goodwill.unit
                and prior_goodwill.value > 0
            ):
                goodwill_count += 1
                goodwill_trend_count += 1
                growth = (
                    (goodwill.value - prior_goodwill.value)
                    / prior_goodwill.value
                    * Decimal(100)
                )
                if growth > config.maximum_goodwill_growth_pct:
                    self._add_flag(
                        flags,
                        code="goodwill_growth_high",
                        category=RedFlagCategory.ACQUISITIONS,
                        severity=config.severity,
                        message=(
                            f"Goodwill grew {self._format(growth)}%, above the configured "
                            f"{self._format(config.maximum_goodwill_growth_pct)}% ceiling."
                        ),
                        evidence=self._evidence(
                            metric="goodwill_growth",
                            value=growth,
                            unit="%",
                            threshold=config.maximum_goodwill_growth_pct,
                            comparison=">",
                            formula="100 × (current goodwill - prior goodwill) / prior goodwill",
                            period=period,
                            values=(goodwill, prior_goodwill),
                        ),
                    )
            elif index:
                self._period_warning(
                    warnings,
                    code="goodwill_growth_period_unavailable",
                    category=RedFlagCategory.ACQUISITIONS,
                    period=period,
                    message=(
                        "Goodwill growth could not be evaluated for this period "
                        "because compatible consecutive goodwill balances were not "
                        "reported."
                    ),
                    required=(FinancialConcept.GOODWILL,),
                )
        if spend_count == 0:
            self._warning(
                warnings,
                code="acquisition_cash_unavailable",
                category=RedFlagCategory.ACQUISITIONS,
                message=(
                    "Acquisition spending was unavailable because the normalized "
                    "acquisition-cash concept and revenue did not overlap."
                ),
                required=(
                    FinancialConcept.ACQUISITION_CASH_PAID,
                    FinancialConcept.REVENUE,
                ),
            )
        if goodwill_trend_count == 0:
            self._warning(
                warnings,
                code="goodwill_growth_unavailable",
                category=RedFlagCategory.ACQUISITIONS,
                message=(
                    "Goodwill growth was unavailable because current and prior "
                    "normalized goodwill balances did not overlap."
                ),
                required=(FinancialConcept.GOODWILL,),
            )
        if acquisition_to_fcf_count == 0:
            self._warning(
                warnings,
                code="acquisition_to_fcf_unavailable",
                category=RedFlagCategory.ACQUISITIONS,
                message=(
                    "Acquisition spending versus FCF was unavailable because "
                    "compatible acquisition cash paid, free cash flow, and positive "
                    "FCF were not reported."
                ),
                required=self._ACQUISITION_TO_FCF_CONCEPTS,
            )
