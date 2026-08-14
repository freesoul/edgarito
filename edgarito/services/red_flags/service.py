from edgarito.config.red_flags import (  # noqa: F401
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
from edgarito.enums.edgar.period import (  # noqa: F401
    FISCAL_PERIOD_PRIORITY,
    FiscalPeriod,
)
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (  # noqa: F401
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.red_flags import (  # noqa: F401
    RedFlag,
    RedFlagCategory,
    RedFlagEvidence,
    RedFlagSeverity,
    RedFlagSourceObservation,
    RedFlagsReport,
    RedFlagWarning,
)
from edgarito.services.red_flags.rules.accounting_quality import _AccountingQualityRules
from edgarito.services.red_flags.rules.acquisitions import _AcquisitionsRules
from edgarito.services.red_flags.rules.cash_conversion import _CashConversionRules
from edgarito.services.red_flags.rules.completeness import _CompletenessRules
from edgarito.services.red_flags.rules.concentration import _ConcentrationRules
from edgarito.services.red_flags.rules.context import _RuleContext
from edgarito.services.red_flags.rules.context import _Value as _RuleValue
from edgarito.services.red_flags.rules.debt import _DebtRules
from edgarito.services.red_flags.rules.dilution_sbc import _DilutionSbcRules
from edgarito.services.red_flags.rules.fcf_vs_earnings import _FcfVsEarningsRules
from edgarito.services.red_flags.rules.margins_growth import _MarginsGrowthRules
from edgarito.services.red_flags.rules.roic import _RoicRules

PeriodKey = tuple[int, FiscalPeriod]
_Value = _RuleValue


class InvestmentRedFlagsService(
    _CompletenessRules,
    _FcfVsEarningsRules,
    _DebtRules,
    _DilutionSbcRules,
    _AcquisitionsRules,
    _MarginsGrowthRules,
    _RoicRules,
    _CashConversionRules,
    _ConcentrationRules,
    _AccountingQualityRules,
    _RuleContext,
):
    """Run deterministic investment red-flag rules on normalized financials.

    Rules never infer that a company is clean when their required normalized
    observations are absent.  Instead, the report contains a typed warning for
    each unavailable rule.
    """

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


# Public aliases keep the API discoverable without forcing one naming style.
RedFlagsService = InvestmentRedFlagsService
RedFlagDetectionService = InvestmentRedFlagsService


__all__ = [
    "InvestmentRedFlagsService",
    "RedFlagDetectionService",
    "RedFlagsService",
]
