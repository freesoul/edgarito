import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from edgarito.config.red_flags import (
    AccountingQualityConfiguration,
    AcquisitionsConfiguration,
    CashConversionConfiguration,
    DebtConfiguration,
    DilutionSbcConfiguration,
    FcfVsEarningsConfiguration,
    MarginsGrowthConfiguration,
    RedFlagsConfiguration,
    RedFlagsProfileLoader,
    RoicConfiguration,
)
from edgarito.enums.edgar.period import FISCAL_PERIOD_PRIORITY, FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.red_flags import (
    RedFlag,
    RedFlagCategory,
    RedFlagEvidence,
    RedFlagSeverity,
    RedFlagSourceObservation,
    RedFlagsReport,
    RedFlagWarning,
)

PeriodKey = tuple[int, FiscalPeriod]


@dataclass(frozen=True)
class _Value:
    value: Decimal
    unit: str
    observations: tuple[FinancialObservation, ...]


class InvestmentRedFlagsService:
    """Run deterministic investment red-flag rules on normalized financials.

    Rules never infer that a company is clean when their required normalized
    observations are absent.  Instead, the report contains a typed warning for
    each unavailable rule.
    """

    _CATEGORY_ORDER = tuple(RedFlagCategory)

    _FCF_CONCEPTS = (
        FinancialConcept.OPERATING_CASH_FLOW,
        FinancialConcept.CAPITAL_EXPENDITURES,
        FinancialConcept.NET_INCOME,
    )
    _ACQUISITION_TO_FCF_CONCEPTS = (
        FinancialConcept.ACQUISITION_CASH_PAID,
        FinancialConcept.OPERATING_CASH_FLOW,
        FinancialConcept.CAPITAL_EXPENDITURES,
    )
    _ROIC_CONCEPTS = (
        FinancialConcept.OPERATING_INCOME,
        FinancialConcept.PRETAX_INCOME,
        FinancialConcept.INCOME_TAX_EXPENSE,
        FinancialConcept.STOCKHOLDERS_EQUITY,
        FinancialConcept.CASH_AND_EQUIVALENTS,
        FinancialConcept.SHORT_TERM_DEBT,
        FinancialConcept.LONG_TERM_DEBT_CURRENT,
        FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
    )

    def __init__(self, configuration: RedFlagsConfiguration | None = None):
        self.configuration = configuration or RedFlagsProfileLoader.load()

    def analyze(
        self,
        financials: NormalizedCompanyFinancials,
        *,
        granularity: Granularity = Granularity.ANNUAL,
        configuration: RedFlagsConfiguration | None = None,
    ) -> RedFlagsReport:
        selected = configuration or self.configuration
        periods = self._periods(financials, granularity)
        evaluated = periods[-selected.history_periods :]
        by_period = self._observations_by_period(financials, granularity)
        flags: list[RedFlag] = []
        warnings: list[RedFlagWarning] = []

        if not evaluated:
            warnings.append(
                RedFlagWarning(
                    code="financial_periods_unavailable",
                    message=(
                        f"No {granularity.value} normalized financial periods were "
                        "available; red-flag rules were not evaluated."
                    ),
                )
            )
        else:
            self._check_fcf_vs_earnings(
                by_period, evaluated, selected.fcf_vs_earnings, flags, warnings
            )
            self._check_debt(by_period, evaluated, selected.debt, flags, warnings)
            self._check_dilution_sbc(
                by_period, evaluated, selected.dilution_sbc, flags, warnings
            )
            self._check_acquisitions(
                by_period, evaluated, selected.acquisitions, flags, warnings
            )
            self._check_margins_growth(
                by_period, evaluated, selected.margins_growth, flags, warnings
            )
            self._check_roic(by_period, evaluated, selected.roic, flags, warnings)
            self._check_cash_conversion(
                by_period, evaluated, selected.cash_conversion, flags, warnings
            )
            self._check_concentration(selected, warnings)
            self._check_accounting_quality(
                by_period, evaluated, selected.accounting_quality, flags, warnings
            )
            self._check_latest_period_completeness(
                by_period, evaluated[-1], selected, warnings
            )

        flags.sort(key=self._flag_sort_key)
        warnings = self._deduplicate_warnings(warnings)
        return RedFlagsReport(
            provider=financials.provider,
            company_id=financials.company_id,
            company_name=financials.company_name,
            ticker=financials.ticker,
            granularity=granularity,
            configuration_name=selected.name,
            evaluated_periods=tuple(evaluated),
            flags=tuple(flags),
            warnings=tuple(warnings),
        )

    def detect(
        self,
        financials: NormalizedCompanyFinancials,
        *,
        granularity: Granularity = Granularity.ANNUAL,
        configuration: RedFlagsConfiguration | None = None,
    ) -> RedFlagsReport:
        """Compatibility alias for callers that use detection terminology."""
        return self.analyze(
            financials, granularity=granularity, configuration=configuration
        )

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
                    (FinancialConcept.OPERATING_INCOME, FinancialConcept.INTEREST_EXPENSE),
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
                    (FinancialConcept.STOCK_BASED_COMPENSATION, FinancialConcept.REVENUE),
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
                if index
                and self._consecutive_periods(periods[index - 1], period)
                else None
            )
            shares = self._share_count(current)
            previous_shares = self._share_count(previous) if previous is not None else None
            if (
                shares is not None
                and previous_shares is not None
                and shares.unit == previous_shares.unit
                and previous_shares.value > 0
            ):
                growth = (shares.value - previous_shares.value) / previous_shares.value * Decimal(100)
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

            basic = self._single(current, FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES)
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
            if acquisition is not None and revenue is not None and acquisition.unit == revenue.unit:
                if revenue.value != 0:
                    spend_count += 1
                    acquisition_to_revenue = abs(acquisition.value) / abs(revenue.value) * Decimal(100)
                    if acquisition_to_revenue > config.maximum_acquisition_to_revenue_pct:
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
                growth = (goodwill.value - prior_goodwill.value) / prior_goodwill.value * Decimal(100)
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
                if index
                and self._consecutive_periods(periods[index - 1], period)
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
                growth = (revenue.value - prior_revenue.value) / prior_revenue.value * Decimal(100)
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
                if previous is not None and margin is not None and prior_margin is not None:
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

    def _check_roic(
        self,
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]],
        periods: list[PeriodKey],
        config: RoicConfiguration,
        flags: list[RedFlag],
        warnings: list[RedFlagWarning],
    ) -> None:
        if not config.enabled:
            return
        roics: dict[PeriodKey, tuple[_Value, _Value]] = {}
        for period in periods:
            roic = self._roic(by_period[period])
            if roic is None:
                self._period_warning(
                    warnings,
                    code="roic_period_unavailable",
                    category=RedFlagCategory.ROIC,
                    period=period,
                    message=(
                        "ROIC could not be evaluated for this period because "
                        "compatible operating income, tax inputs, equity, debt, and "
                        "cash were not reported."
                    ),
                    required=self._ROIC_CONCEPTS,
                )
                continue
            roics[period] = roic
            value, nopat = roic
            if value.value < config.minimum_roic_pct:
                self._add_flag(
                    flags,
                    code="roic_low",
                    category=RedFlagCategory.ROIC,
                    severity=config.severity,
                    message=(
                        f"ROIC was {self._format(value.value)}%, below the configured "
                        f"{self._format(config.minimum_roic_pct)}% floor."
                    ),
                    evidence=self._evidence(
                        metric="roic",
                        value=value.value,
                        unit="%",
                        threshold=config.minimum_roic_pct,
                        comparison="<",
                        formula="100 × NOPAT / (stockholders' equity + gross debt - cash)",
                        period=period,
                        values=(value, nopat),
                    ),
                )
        trend_count = 0
        for index, period in enumerate(periods):
            if not index:
                continue
            previous_period = periods[index - 1]
            if (
                not self._consecutive_periods(previous_period, period)
                or period not in roics
                or previous_period not in roics
            ):
                self._period_warning(
                    warnings,
                    code="roic_trend_period_unavailable",
                    category=RedFlagCategory.ROIC,
                    period=period,
                    message=(
                        "ROIC trend could not be evaluated for this period because "
                        "complete ROIC observations for consecutive periods were not "
                        "reported."
                    ),
                    required=self._ROIC_CONCEPTS,
                )
                continue
            current, current_nopat = roics[period]
            previous, previous_nopat = roics[previous_period]
            trend_count += 1
            decline = previous.value - current.value
            if decline > config.maximum_roic_decline_pp:
                self._add_flag(
                    flags,
                    code="roic_decline",
                    category=RedFlagCategory.ROIC,
                    severity=config.severity,
                    message=(
                        f"ROIC declined {self._format(decline)} percentage points, above the configured "
                        f"{self._format(config.maximum_roic_decline_pp)}-point ceiling."
                    ),
                    evidence=self._evidence(
                        metric="roic_decline",
                        value=decline,
                        unit="percentage_points",
                        threshold=config.maximum_roic_decline_pp,
                        comparison=">",
                        formula="prior ROIC - current ROIC",
                        period=period,
                        values=(current, previous, current_nopat, previous_nopat),
                    ),
                )
        if not roics:
            self._warning(
                warnings,
                code="roic_unavailable",
                category=RedFlagCategory.ROIC,
                message=(
                    "ROIC was unavailable because normalized operating income, "
                    "tax inputs, equity, debt, and cash did not overlap with compatible units."
                ),
                required=RedFlagsConfiguration.required_concepts(RedFlagCategory.ROIC),
            )
        if roics and trend_count == 0:
            self._warning(
                warnings,
                code="roic_trend_unavailable",
                category=RedFlagCategory.ROIC,
                message="ROIC trend was unavailable because complete consecutive ROIC periods were not reported.",
                required=self._ROIC_CONCEPTS,
            )

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
            operating_cash_flow = self._single(current, FinancialConcept.OPERATING_CASH_FLOW)
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

    def _check_concentration(
        self, configuration: RedFlagsConfiguration, warnings: list[RedFlagWarning]
    ) -> None:
        if not configuration.concentration.enabled:
            return
        self._warning(
            warnings,
            code="concentration_data_unavailable",
            category=RedFlagCategory.CONCENTRATION,
            message=(
                "Customer and segment concentration data is not represented in the "
                "normalized financial model; concentration rules were not evaluated."
            ),
            required=(),
        )

    def _check_accounting_quality(
        self,
        by_period: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]],
        periods: list[PeriodKey],
        config: AccountingQualityConfiguration,
        flags: list[RedFlag],
        warnings: list[RedFlagWarning],
    ) -> None:
        if not config.enabled:
            return
        receivable_count = 0
        inventory_count = 0
        goodwill_asset_count = 0
        for index, period in enumerate(periods):
            previous = (
                by_period[periods[index - 1]]
                if index
                and self._consecutive_periods(periods[index - 1], period)
                else None
            )
            current = by_period[period]
            revenue = self._single(current, FinancialConcept.REVENUE)
            prior_revenue = (
                self._single(previous, FinancialConcept.REVENUE)
                if previous is not None
                else None
            )
            revenue_growth = self._growth(revenue, prior_revenue)
            if revenue_growth is not None:
                receivables = self._single(current, FinancialConcept.ACCOUNTS_RECEIVABLE)
                prior_receivables = (
                    self._single(previous, FinancialConcept.ACCOUNTS_RECEIVABLE)
                    if previous is not None
                    else None
                )
                receivable_growth = self._growth(receivables, prior_receivables)
                if receivable_growth is not None:
                    receivable_count += 1
                    premium = receivable_growth.value - revenue_growth.value
                    if premium > config.maximum_receivables_growth_premium_pp:
                        self._add_flag(
                            flags,
                            code="receivables_growth_ahead_of_revenue",
                            category=RedFlagCategory.ACCOUNTING_QUALITY,
                            severity=config.severity,
                            message=(
                                f"Receivables growth exceeded revenue growth by {self._format(premium)} percentage points, "
                                f"above the configured {self._format(config.maximum_receivables_growth_premium_pp)}-point ceiling."
                            ),
                            evidence=self._evidence(
                                metric="receivables_growth_premium",
                                value=premium,
                                unit="percentage_points",
                                threshold=config.maximum_receivables_growth_premium_pp,
                                comparison=">",
                                formula="receivables growth - revenue growth",
                                period=period,
                                values=(receivable_growth, revenue_growth),
                            ),
                        )
                else:
                    self._period_warning(
                        warnings,
                        code="receivables_growth_period_unavailable",
                        category=RedFlagCategory.ACCOUNTING_QUALITY,
                        period=period,
                        message=(
                            "Receivables-versus-revenue growth could not be evaluated "
                            "for this period because compatible consecutive balances "
                            "were not reported."
                        ),
                        required=(
                            FinancialConcept.ACCOUNTS_RECEIVABLE,
                            FinancialConcept.REVENUE,
                        ),
                    )

                inventory = self._single(current, FinancialConcept.INVENTORY)
                prior_inventory = (
                    self._single(previous, FinancialConcept.INVENTORY)
                    if previous is not None
                    else None
                )
                inventory_growth = self._growth(inventory, prior_inventory)
                if inventory_growth is not None:
                    inventory_count += 1
                    premium = inventory_growth.value - revenue_growth.value
                    if premium > config.maximum_inventory_growth_premium_pp:
                        self._add_flag(
                            flags,
                            code="inventory_growth_ahead_of_revenue",
                            category=RedFlagCategory.ACCOUNTING_QUALITY,
                            severity=config.severity,
                            message=(
                                f"Inventory growth exceeded revenue growth by {self._format(premium)} percentage points, "
                                f"above the configured {self._format(config.maximum_inventory_growth_premium_pp)}-point ceiling."
                            ),
                            evidence=self._evidence(
                                metric="inventory_growth_premium",
                                value=premium,
                                unit="percentage_points",
                                threshold=config.maximum_inventory_growth_premium_pp,
                                comparison=">",
                                formula="inventory growth - revenue growth",
                                period=period,
                                values=(inventory_growth, revenue_growth),
                            ),
                        )
                else:
                    self._period_warning(
                        warnings,
                        code="inventory_growth_period_unavailable",
                        category=RedFlagCategory.ACCOUNTING_QUALITY,
                        period=period,
                        message=(
                            "Inventory-versus-revenue growth could not be evaluated "
                            "for this period because compatible consecutive balances "
                            "were not reported."
                        ),
                        required=(
                            FinancialConcept.INVENTORY,
                            FinancialConcept.REVENUE,
                        ),
                    )

            elif index:
                self._period_warning(
                    warnings,
                    code="receivables_growth_period_unavailable",
                    category=RedFlagCategory.ACCOUNTING_QUALITY,
                    period=period,
                    message=(
                        "Receivables-versus-revenue growth could not be evaluated for "
                        "this period because compatible consecutive revenue periods "
                        "were not reported."
                    ),
                    required=(
                        FinancialConcept.ACCOUNTS_RECEIVABLE,
                        FinancialConcept.REVENUE,
                    ),
                )
                self._period_warning(
                    warnings,
                    code="inventory_growth_period_unavailable",
                    category=RedFlagCategory.ACCOUNTING_QUALITY,
                    period=period,
                    message=(
                        "Inventory-versus-revenue growth could not be evaluated for "
                        "this period because compatible consecutive revenue periods "
                        "were not reported."
                    ),
                    required=(
                        FinancialConcept.INVENTORY,
                        FinancialConcept.REVENUE,
                    ),
                )

            goodwill = self._single(current, FinancialConcept.GOODWILL)
            assets = self._single(current, FinancialConcept.TOTAL_ASSETS)
            ratio = self._ratio(goodwill, assets, percentage=True)
            if ratio is not None:
                goodwill_asset_count += 1
                if ratio.value > config.maximum_goodwill_to_assets_pct:
                    self._add_flag(
                        flags,
                        code="goodwill_to_assets_high",
                        category=RedFlagCategory.ACCOUNTING_QUALITY,
                        severity=config.severity,
                        message=(
                            f"Goodwill was {self._format(ratio.value)}% of assets, above the configured "
                            f"{self._format(config.maximum_goodwill_to_assets_pct)}% ceiling."
                        ),
                        evidence=self._evidence(
                            metric="goodwill_to_assets",
                            value=ratio.value,
                            unit="%",
                            threshold=config.maximum_goodwill_to_assets_pct,
                            comparison=">",
                            formula="100 × goodwill / total assets",
                            period=period,
                            values=(ratio,),
                            ),
                        )
            else:
                self._period_warning(
                    warnings,
                    code="goodwill_to_assets_period_unavailable",
                    category=RedFlagCategory.ACCOUNTING_QUALITY,
                    period=period,
                    message=(
                        "Goodwill-to-assets could not be evaluated for this period "
                        "because compatible goodwill and total-assets balances were "
                        "not reported."
                    ),
                    required=(
                        FinancialConcept.GOODWILL,
                        FinancialConcept.TOTAL_ASSETS,
                    ),
                )
        if receivable_count == 0:
            self._warning(
                warnings,
                code="receivables_growth_unavailable",
                category=RedFlagCategory.ACCOUNTING_QUALITY,
                message="Receivables-versus-revenue growth was unavailable because compatible consecutive balances were not reported.",
                required=(FinancialConcept.ACCOUNTS_RECEIVABLE, FinancialConcept.REVENUE),
            )
        if inventory_count == 0:
            self._warning(
                warnings,
                code="inventory_growth_unavailable",
                category=RedFlagCategory.ACCOUNTING_QUALITY,
                message="Inventory-versus-revenue growth was unavailable because compatible consecutive balances were not reported.",
                required=(FinancialConcept.INVENTORY, FinancialConcept.REVENUE),
            )
        if goodwill_asset_count == 0:
            self._warning(
                warnings,
                code="goodwill_to_assets_unavailable",
                category=RedFlagCategory.ACCOUNTING_QUALITY,
                message="Goodwill-to-assets was unavailable because compatible goodwill and total-assets balances were not reported.",
                required=(FinancialConcept.GOODWILL, FinancialConcept.TOTAL_ASSETS),
            )

    @staticmethod
    def _observations_by_period(
        financials: NormalizedCompanyFinancials, granularity: Granularity
    ) -> dict[PeriodKey, dict[FinancialConcept, FinancialObservation]]:
        result: dict[PeriodKey, dict[FinancialConcept, FinancialObservation]] = {}
        for observation in financials.observations:
            if observation.granularity != granularity:
                continue
            current = result.setdefault(observation.period_key, {})
            existing = current.get(observation.concept)
            if existing is None or InvestmentRedFlagsService._observation_key(observation) > InvestmentRedFlagsService._observation_key(existing):
                current[observation.concept] = observation
        return result

    @staticmethod
    def _observation_key(observation: FinancialObservation):
        return (
            observation.filed or datetime.date.min,
            observation.period_end,
            observation.source_concept,
        )

    @staticmethod
    def _periods(
        financials: NormalizedCompanyFinancials, granularity: Granularity
    ) -> list[PeriodKey]:
        return sorted(
            {
                observation.period_key
                for observation in financials.observations
                if observation.granularity == granularity
            },
            key=lambda period: (period[0], FISCAL_PERIOD_PRIORITY[period[1]]),
        )

    @staticmethod
    def _single(
        observations: dict[FinancialConcept, FinancialObservation] | None,
        concept: FinancialConcept,
    ) -> _Value | None:
        if observations is None:
            return None
        observation = observations.get(concept)
        if observation is None:
            return None
        return _Value(observation.value, observation.unit, (observation,))

    @classmethod
    def _combine(
        cls,
        observations: dict[FinancialConcept, FinancialObservation] | None,
        terms: Iterable[tuple[FinancialConcept, int]],
    ) -> _Value | None:
        if observations is None:
            return None
        values = [cls._single(observations, concept) for concept, _ in terms]
        if any(value is None for value in values):
            return None
        present = [value for value in values if value is not None]
        if len({value.unit for value in present}) != 1:
            return None
        value = sum(
            (
                item.value * coefficient
                for item, (_, coefficient) in zip(present, terms, strict=True)
            ),
            Decimal(0),
        )
        return _Value(
            value,
            present[0].unit,
            tuple(observation for item in present for observation in item.observations),
        )

    @classmethod
    def _gross_debt(
        cls, observations: dict[FinancialConcept, FinancialObservation] | None
    ) -> _Value | None:
        if observations is None:
            return None
        # SHORT_TERM_DEBT is normalized from aggregate current-debt rows such
        # as DebtCurrent/CurrentDebt.  When it is present, it already includes
        # current maturities represented by LONG_TERM_DEBT_CURRENT.
        current = cls._single(observations, FinancialConcept.SHORT_TERM_DEBT)
        if current is None:
            current = cls._single(observations, FinancialConcept.LONG_TERM_DEBT_CURRENT)
        values = [
            current,
            cls._single(observations, FinancialConcept.LONG_TERM_DEBT_NONCURRENT),
        ]
        present = [value for value in values if value is not None]
        if not present or len({value.unit for value in present}) != 1:
            return None
        return _Value(
            sum((value.value for value in present), Decimal(0)),
            present[0].unit,
            tuple(observation for value in present for observation in value.observations),
        )

    @classmethod
    def _free_cash_flow(
        cls, observations: dict[FinancialConcept, FinancialObservation] | None
    ) -> _Value | None:
        return cls._combine(
            observations,
            (
                (FinancialConcept.OPERATING_CASH_FLOW, 1),
                (FinancialConcept.CAPITAL_EXPENDITURES, -1),
            ),
        )

    @classmethod
    def _share_count(
        cls, observations: dict[FinancialConcept, FinancialObservation] | None
    ) -> _Value | None:
        return cls._single(observations, FinancialConcept.SHARES_OUTSTANDING) or cls._single(
            observations, FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES
        )

    @staticmethod
    def _ratio(
        numerator: _Value | None,
        denominator: _Value | None,
        *,
        percentage: bool,
    ) -> _Value | None:
        if (
            numerator is None
            or denominator is None
            or numerator.unit != denominator.unit
            or denominator.value == 0
        ):
            return None
        multiplier = Decimal(100) if percentage else Decimal(1)
        return _Value(
            numerator.value / denominator.value * multiplier,
            "%" if percentage else "x",
            numerator.observations + denominator.observations,
        )

    @classmethod
    def _growth(cls, current: _Value | None, previous: _Value | None) -> _Value | None:
        if (
            current is None
            or previous is None
            or current.unit != previous.unit
            or previous.value <= 0
        ):
            return None
        return _Value(
            (current.value - previous.value) / previous.value * Decimal(100),
            "%",
            current.observations + previous.observations,
        )

    @classmethod
    def _nopat(
        cls, observations: dict[FinancialConcept, FinancialObservation] | None
    ) -> _Value | None:
        operating_income = cls._single(observations, FinancialConcept.OPERATING_INCOME)
        pretax_income = cls._single(observations, FinancialConcept.PRETAX_INCOME)
        tax = cls._single(observations, FinancialConcept.INCOME_TAX_EXPENSE)
        if (
            operating_income is None
            or pretax_income is None
            or tax is None
            or len({operating_income.unit, pretax_income.unit, tax.unit}) != 1
            or pretax_income.value == 0
        ):
            return None
        return _Value(
            operating_income.value * (Decimal(1) - tax.value / pretax_income.value),
            operating_income.unit,
            operating_income.observations + pretax_income.observations + tax.observations,
        )

    @classmethod
    def _invested_capital(
        cls, observations: dict[FinancialConcept, FinancialObservation] | None
    ) -> _Value | None:
        equity = cls._single(observations, FinancialConcept.STOCKHOLDERS_EQUITY)
        debt = cls._gross_debt(observations)
        cash = cls._single(observations, FinancialConcept.CASH_AND_EQUIVALENTS)
        if equity is None or debt is None or cash is None:
            return None
        if len({equity.unit, debt.unit, cash.unit}) != 1:
            return None
        return _Value(
            equity.value + debt.value - cash.value,
            equity.unit,
            equity.observations + debt.observations + cash.observations,
        )

    @classmethod
    def _roic(
        cls, observations: dict[FinancialConcept, FinancialObservation]
    ) -> tuple[_Value, _Value] | None:
        nopat = cls._nopat(observations)
        invested_capital = cls._invested_capital(observations)
        if (
            nopat is None
            or invested_capital is None
            or nopat.unit != invested_capital.unit
            or invested_capital.value <= 0
        ):
            return None
        return (
            _Value(
                nopat.value / invested_capital.value * Decimal(100),
                "%",
                nopat.observations + invested_capital.observations,
            ),
            nopat,
        )

    @classmethod
    def _evidence(
        cls,
        *,
        metric: str,
        value: Decimal,
        unit: str,
        threshold: Decimal | None,
        comparison: str,
        formula: str,
        period: PeriodKey,
        values: Iterable[_Value],
    ) -> RedFlagEvidence:
        source_values = tuple(values)
        observations = tuple(
            sorted(
                (
                    RedFlagSourceObservation(
                        concept=observation.concept,
                        value=observation.value,
                        unit=observation.unit,
                        granularity=observation.granularity,
                        fiscal_year=observation.fiscal_year,
                        fiscal_period=observation.fiscal_period,
                        period_end=observation.period_end,
                        provider=observation.provider,
                        source_concept=observation.source_concept,
                    )
                    for value_item in source_values
                    for observation in value_item.observations
                ),
                key=lambda observation: (
                    observation.fiscal_year,
                    FISCAL_PERIOD_PRIORITY[observation.fiscal_period],
                    observation.concept.value,
                    observation.source_concept,
                ),
            )
        )
        return RedFlagEvidence(
            metric=metric,
            value=value,
            unit=unit,
            threshold=threshold,
            threshold_unit=unit if threshold is not None else None,
            comparison=comparison,
            formula=formula,
            fiscal_year=period[0],
            fiscal_period=period[1],
            period_end=max(
                observation.period_end for observation in observations
            ),
            granularity=observations[0].granularity,
            input_concepts=tuple(
                sorted(
                    {observation.concept for observation in observations},
                    key=lambda concept: concept.value,
                )
            ),
            source_observations=observations,
        )

    @staticmethod
    def _add_flag(
        flags: list[RedFlag],
        *,
        code: str,
        category: RedFlagCategory,
        severity: RedFlagSeverity,
        message: str,
        evidence: RedFlagEvidence,
    ) -> None:
        flags.append(
            RedFlag(
                code=code,
                category=category,
                severity=severity,
                message=message,
                evidence=(evidence,),
            )
        )

    @staticmethod
    def _warning(
        warnings: list[RedFlagWarning],
        *,
        code: str,
        category: RedFlagCategory,
        message: str,
        required: Iterable[FinancialConcept],
        period: PeriodKey | None = None,
    ) -> None:
        warnings.append(
            RedFlagWarning(
                code=code,
                category=category,
                message=message,
                period=period,
                required_concepts=tuple(sorted(set(required), key=lambda concept: concept.value)),
            )
        )

    @classmethod
    def _period_warning(
        cls,
        warnings: list[RedFlagWarning],
        *,
        code: str,
        category: RedFlagCategory,
        period: PeriodKey,
        message: str,
        required: Iterable[FinancialConcept],
    ) -> None:
        """Record an unavailable rule for one period, not just globally."""
        cls._warning(
            warnings,
            code=code,
            category=category,
            message=message,
            required=required,
            period=period,
        )

    @staticmethod
    def _consecutive_periods(previous: PeriodKey, current: PeriodKey) -> bool:
        """Return whether two fiscal keys are adjacent at their granularity."""
        if previous[1] == FiscalPeriod.FY or current[1] == FiscalPeriod.FY:
            return (
                previous[1] == FiscalPeriod.FY
                and current[1] == FiscalPeriod.FY
                and current[0] == previous[0] + 1
            )
        quarter_number = {
            FiscalPeriod.Q1: 1,
            FiscalPeriod.Q2: 2,
            FiscalPeriod.Q3: 3,
            FiscalPeriod.Q4: 4,
        }
        if previous[1] not in quarter_number or current[1] not in quarter_number:
            return False
        previous_index = previous[0] * 4 + quarter_number[previous[1]]
        current_index = current[0] * 4 + quarter_number[current[1]]
        return current_index == previous_index + 1

    @classmethod
    def _flag_sort_key(cls, flag: RedFlag):
        evidence = flag.evidence[0]
        return (
            evidence.fiscal_year,
            FISCAL_PERIOD_PRIORITY[evidence.fiscal_period],
            cls._CATEGORY_ORDER.index(flag.category),
            flag.code,
        )

    @classmethod
    def _deduplicate_warnings(
        cls, warnings: list[RedFlagWarning]
    ) -> list[RedFlagWarning]:
        unique = {
            (warning.code, warning.message, warning.period): warning
            for warning in warnings
        }
        return sorted(
            unique.values(),
            key=lambda warning: (
                cls._CATEGORY_ORDER.index(warning.category)
                if warning.category is not None
                else -1,
                (
                    warning.period[0],
                    FISCAL_PERIOD_PRIORITY[warning.period[1]],
                )
                if warning.period is not None
                else (-1, -1),
                warning.code,
            ),
        )

    @staticmethod
    def _format(value: Decimal) -> str:
        return f"{value:.2f}"

    @staticmethod
    def _period_label(period: PeriodKey) -> str:
        """Return a stable human-readable fiscal period label for warnings."""
        year, fiscal_period = period
        return f"{fiscal_period.value} {year}"


# Public aliases keep the API discoverable without forcing one naming style.
RedFlagsService = InvestmentRedFlagsService
RedFlagDetectionService = InvestmentRedFlagsService


__all__ = [
    "InvestmentRedFlagsService",
    "RedFlagDetectionService",
    "RedFlagsService",
]
