from edgarito.config.red_flags import DebtConfiguration
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
)
from edgarito.schemas.red_flags import (
    RedFlag,
    RedFlagCategory,
    RedFlagWarning,
)
from edgarito.services.red_flags.rules.context import PeriodKey, _Value


class _DebtRules:
    def _check_debt(
        self,
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]],
        periods: list[PeriodKey],
        config: DebtConfiguration,
        flags: list[RedFlag],
        warnings: list[RedFlagWarning],
    ) -> None:
        if not config.enabled:
            return
        net_debt_count = 0
        coverage_count = 0
        for period in periods:
            current = by_period[period]
            debt = self._gross_debt(current)
            cash = self._single(current, FinancialConcept.CASH_AND_EQUIVALENTS)
            ebitda = self._combine(
                current,
                (
                    (FinancialConcept.OPERATING_INCOME, 1),
                    (FinancialConcept.DEPRECIATION_AND_AMORTIZATION, 1),
                ),
            )
            net_debt_is_available = (
                debt is not None
                and cash is not None
                and ebitda is not None
                and len({debt.unit, cash.unit, ebitda.unit}) == 1
                and ebitda.value > 0
            )
            if net_debt_is_available:
                net_debt = _Value(
                    debt.value - cash.value,
                    debt.unit,
                    debt.observations + cash.observations,
                )
                ratio = net_debt.value / ebitda.value
                net_debt_count += 1
                if ratio > config.maximum_net_debt_to_ebitda:
                    self._add_flag(
                        flags,
                        code="net_debt_to_ebitda_high",
                        category=RedFlagCategory.DEBT,
                        severity=config.severity,
                        message=(
                            f"Net debt was {self._format(ratio)}× EBITDA, above "
                            f"the configured {self._format(config.maximum_net_debt_to_ebitda)}× ceiling."
                        ),
                        evidence=self._evidence(
                            metric="net_debt_to_ebitda",
                            value=ratio,
                            unit="x",
                            threshold=config.maximum_net_debt_to_ebitda,
                            comparison=">",
                            formula="(gross debt - cash and equivalents) / EBITDA",
                            period=period,
                            values=(net_debt, ebitda),
                        ),
                    )
            else:
                self._period_warning(
                    warnings,
                    code="net_debt_to_ebitda_period_unavailable",
                    category=RedFlagCategory.DEBT,
                    period=period,
                    message=(
                        "Net-debt-to-EBITDA could not be evaluated for this period "
                        "because compatible debt, cash, and positive EBITDA were "
                        "not reported."
                    ),
                    required=(
                        FinancialConcept.SHORT_TERM_DEBT,
                        FinancialConcept.LONG_TERM_DEBT_CURRENT,
                        FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
                        FinancialConcept.CASH_AND_EQUIVALENTS,
                        FinancialConcept.OPERATING_INCOME,
                        FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
                    ),
                )

            interest = self._single(current, FinancialConcept.INTEREST_EXPENSE)
            operating_income = self._single(current, FinancialConcept.OPERATING_INCOME)
            if (
                interest is not None
                and operating_income is not None
                and interest.unit == operating_income.unit
                and interest.value != 0
            ):
                coverage = operating_income.value / abs(interest.value)
                coverage_count += 1
                if coverage < config.minimum_interest_coverage:
                    self._add_flag(
                        flags,
                        code="interest_coverage_low",
                        category=RedFlagCategory.DEBT,
                        severity=config.severity,
                        message=(
                            f"Operating income covered interest {self._format(coverage)}×, "
                            f"below the configured {self._format(config.minimum_interest_coverage)}× floor."
                        ),
                        evidence=self._evidence(
                            metric="interest_coverage",
                            value=coverage,
                            unit="x",
                            threshold=config.minimum_interest_coverage,
                            comparison="<",
                            formula="operating income / absolute interest expense",
                            period=period,
                            values=(operating_income, interest),
                        ),
                    )
            else:
                self._period_warning(
                    warnings,
                    code="interest_coverage_period_unavailable",
                    category=RedFlagCategory.DEBT,
                    period=period,
                    message=(
                        "Interest coverage could not be evaluated for this period "
                        "because compatible operating income and non-zero interest "
                        "expense were not reported."
                    ),
                    required=(
                        FinancialConcept.OPERATING_INCOME,
                        FinancialConcept.INTEREST_EXPENSE,
                    ),
                )
        if net_debt_count == 0:
            self._warning(
                warnings,
                code="net_debt_to_ebitda_unavailable",
                category=RedFlagCategory.DEBT,
                message=(
                    "Net-debt-to-EBITDA was unavailable because debt, cash, or "
                    "positive EBITDA was not reported with compatible units."
                ),
                required=(
                    FinancialConcept.SHORT_TERM_DEBT,
                    FinancialConcept.LONG_TERM_DEBT_CURRENT,
                    FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
                    FinancialConcept.CASH_AND_EQUIVALENTS,
                    FinancialConcept.OPERATING_INCOME,
                    FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
                ),
            )
        if coverage_count == 0:
            self._warning(
                warnings,
                code="interest_coverage_unavailable",
                category=RedFlagCategory.DEBT,
                message=(
                    "Interest coverage was unavailable because compatible "
                    "operating income and non-zero interest expense were not reported."
                ),
                required=(
                    FinancialConcept.OPERATING_INCOME,
                    FinancialConcept.INTEREST_EXPENSE,
                ),
            )
