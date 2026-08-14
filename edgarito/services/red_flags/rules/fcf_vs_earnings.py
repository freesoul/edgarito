from decimal import Decimal

from edgarito.config.red_flags import FcfVsEarningsConfiguration
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


class _FcfVsEarningsRules:
    def _check_fcf_vs_earnings(
        self,
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]],
        periods: list[PeriodKey],
        config: FcfVsEarningsConfiguration,
        flags: list[RedFlag],
        warnings: list[RedFlagWarning],
    ) -> None:
        if not config.enabled:
            return
        evidence_count = 0
        positive_earnings_count = 0
        for period in periods:
            fcf = self._free_cash_flow(by_period[period])
            earnings = self._single(by_period[period], FinancialConcept.NET_INCOME)
            if fcf is None or earnings is None or fcf.unit != earnings.unit:
                self._period_warning(
                    warnings,
                    code="fcf_vs_earnings_period_unavailable",
                    category=RedFlagCategory.FCF_VS_EARNINGS,
                    period=period,
                    message=(
                        "FCF versus earnings could not be evaluated because the "
                        "period lacks compatible operating cash flow, capital "
                        "expenditures, or net income."
                    ),
                    required=self._FCF_CONCEPTS,
                )
                continue
            if earnings.value <= 0:
                continue
            positive_earnings_count += 1
            evidence_count += 1
            ratio = fcf.value / earnings.value * Decimal(100)
            evidence = self._evidence(
                metric="fcf_to_net_income",
                value=ratio,
                unit="%",
                threshold=config.minimum_fcf_to_net_income_pct,
                comparison="<",
                formula="100 × (operating cash flow - capital expenditures) / net income",
                period=period,
                values=(fcf, earnings),
            )
            if ratio < config.minimum_fcf_to_net_income_pct:
                self._add_flag(
                    flags,
                    code="fcf_below_earnings",
                    category=RedFlagCategory.FCF_VS_EARNINGS,
                    severity=config.severity,
                    message=(
                        f"Free cash flow converted {self._format(ratio)}% of net "
                        f"income, below the configured {self._format(config.minimum_fcf_to_net_income_pct)}% floor."
                    ),
                    evidence=evidence,
                )
            if config.flag_negative_fcf_with_positive_earnings and fcf.value < 0:
                self._add_flag(
                    flags,
                    code="negative_fcf_with_positive_earnings",
                    category=RedFlagCategory.FCF_VS_EARNINGS,
                    severity=config.severity,
                    message=(
                        "Free cash flow was negative while reported net income "
                        f"was positive ({self._format(earnings.value)} {earnings.unit})."
                    ),
                    evidence=self._evidence(
                        metric="free_cash_flow",
                        value=fcf.value,
                        unit=fcf.unit,
                        threshold=Decimal(0),
                        comparison="<",
                        formula="operating cash flow - capital expenditures",
                        period=period,
                        values=(fcf, earnings),
                    ),
                )
        if evidence_count == 0:
            self._warning(
                warnings,
                code="fcf_vs_earnings_unavailable",
                category=RedFlagCategory.FCF_VS_EARNINGS,
                message=(
                    "FCF versus earnings was unavailable because normalized "
                    "operating cash flow, capital expenditures, and positive net "
                    "income did not overlap."
                ),
                required=(
                    FinancialConcept.OPERATING_CASH_FLOW,
                    FinancialConcept.CAPITAL_EXPENDITURES,
                    FinancialConcept.NET_INCOME,
                ),
            )
