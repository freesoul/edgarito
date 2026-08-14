from edgarito.config.red_flags import RedFlagsConfiguration
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
)
from edgarito.schemas.red_flags import RedFlagCategory, RedFlagWarning
from edgarito.services.red_flags.rules.context import PeriodKey


class _CompletenessRules:
    def _check_latest_period_completeness(
        self,
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]],
        period: PeriodKey,
        configuration: RedFlagsConfiguration,
        warnings: list[RedFlagWarning],
    ) -> None:
        """Prevent older complete periods from masking an incomplete latest one."""
        observations = by_period[period]
        for category in configuration.enabled_categories:
            if any(
                warning.category == category and warning.period == period
                for warning in warnings
            ):
                continue
            requirements: tuple[tuple[FinancialConcept, ...], ...]
            if category == RedFlagCategory.FCF_VS_EARNINGS:
                requirements = (self._FCF_CONCEPTS,)
            elif category == RedFlagCategory.DEBT:
                requirements = (
                    (
                        FinancialConcept.CASH_AND_EQUIVALENTS,
                        FinancialConcept.OPERATING_INCOME,
                        FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
                    ),
                    (
                        FinancialConcept.OPERATING_INCOME,
                        FinancialConcept.INTEREST_EXPENSE,
                    ),
                    (
                        FinancialConcept.SHORT_TERM_DEBT,
                        FinancialConcept.LONG_TERM_DEBT_CURRENT,
                        FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
                    ),
                )
            elif category == RedFlagCategory.DILUTION_SBC:
                requirements = (
                    (
                        FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES,
                        FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES,
                    ),
                    (
                        FinancialConcept.STOCK_BASED_COMPENSATION,
                        FinancialConcept.REVENUE,
                    ),
                )
            elif category == RedFlagCategory.ACQUISITIONS:
                requirements = (
                    (
                        FinancialConcept.ACQUISITION_CASH_PAID,
                        FinancialConcept.REVENUE,
                    ),
                    self._ACQUISITION_TO_FCF_CONCEPTS,
                    (FinancialConcept.GOODWILL,),
                )
            elif category == RedFlagCategory.MARGINS_GROWTH:
                requirements = (
                    (FinancialConcept.OPERATING_INCOME, FinancialConcept.REVENUE),
                )
            elif category == RedFlagCategory.ROIC:
                requirements = (
                    (
                        FinancialConcept.OPERATING_INCOME,
                        FinancialConcept.PRETAX_INCOME,
                        FinancialConcept.INCOME_TAX_EXPENSE,
                        FinancialConcept.STOCKHOLDERS_EQUITY,
                        FinancialConcept.CASH_AND_EQUIVALENTS,
                    ),
                    (
                        FinancialConcept.SHORT_TERM_DEBT,
                        FinancialConcept.LONG_TERM_DEBT_CURRENT,
                        FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
                    ),
                )
            elif category == RedFlagCategory.CASH_CONVERSION:
                requirements = (
                    (
                        FinancialConcept.OPERATING_CASH_FLOW,
                        FinancialConcept.NET_INCOME,
                    ),
                )
            elif category == RedFlagCategory.ACCOUNTING_QUALITY:
                requirements = (
                    (FinancialConcept.GOODWILL, FinancialConcept.TOTAL_ASSETS),
                )
            else:
                requirements = ()

            if not requirements:
                continue
            missing_set = {
                concept
                for requirement in requirements
                if not all(concept in observations for concept in requirement)
                for concept in requirement
            }
            missing = tuple(sorted(missing_set, key=lambda concept: concept.value))
            if not missing:
                continue
            self._warning(
                warnings,
                code="latest_period_incomplete",
                category=category,
                message=(
                    f"The latest evaluated period {self._period_label(period)} is "
                    f"missing normalized inputs for the {category.value} rules; "
                    "a clean result cannot be claimed for this category."
                ),
                required=missing,
                period=period,
            )
