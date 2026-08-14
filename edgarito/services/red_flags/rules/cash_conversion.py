from decimal import Decimal

from edgarito.config.red_flags import CashConversionConfiguration
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


class _CashConversionRules:
    def _check_cash_conversion(
        self,
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]],
        periods: list[PeriodKey],
        config: CashConversionConfiguration,
        flags: list[RedFlag],
        warnings: list[RedFlagWarning],
    ) -> None:
        if not config.enabled:
            return
        count = 0
        for period in periods:
            current = by_period[period]
            operating_cash_flow = self._single(
                current, FinancialConcept.OPERATING_CASH_FLOW
            )
            net_income = self._single(current, FinancialConcept.NET_INCOME)
            if (
                operating_cash_flow is None
                or net_income is None
                or operating_cash_flow.unit != net_income.unit
                or net_income.value <= 0
            ):
                self._period_warning(
                    warnings,
                    code="cash_conversion_period_unavailable",
                    category=RedFlagCategory.CASH_CONVERSION,
                    period=period,
                    message=(
                        "Cash conversion could not be evaluated for this period "
                        "because compatible operating cash flow and positive net "
                        "income were not reported."
                    ),
                    required=(
                        FinancialConcept.OPERATING_CASH_FLOW,
                        FinancialConcept.NET_INCOME,
                    ),
                )
                continue
            count += 1
            ratio = operating_cash_flow.value / net_income.value * Decimal(100)
            if ratio < config.minimum_operating_cash_flow_to_net_income_pct:
                self._add_flag(
                    flags,
                    code="cash_conversion_low",
                    category=RedFlagCategory.CASH_CONVERSION,
                    severity=config.severity,
                    message=(
                        f"Operating cash flow converted {self._format(ratio)}% of net income, "
                        f"below the configured {self._format(config.minimum_operating_cash_flow_to_net_income_pct)}% floor."
                    ),
                    evidence=self._evidence(
                        metric="operating_cash_flow_to_net_income",
                        value=ratio,
                        unit="%",
                        threshold=config.minimum_operating_cash_flow_to_net_income_pct,
                        comparison="<",
                        formula="100 × operating cash flow / net income",
                        period=period,
                        values=(operating_cash_flow, net_income),
                    ),
                )
        if count == 0:
            self._warning(
                warnings,
                code="cash_conversion_unavailable",
                category=RedFlagCategory.CASH_CONVERSION,
                message=(
                    "Cash conversion was unavailable because compatible operating "
                    "cash flow and positive net income were not reported."
                ),
                required=(
                    FinancialConcept.OPERATING_CASH_FLOW,
                    FinancialConcept.NET_INCOME,
                ),
            )
