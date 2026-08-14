from edgarito.config.red_flags import RedFlagsConfiguration, RoicConfiguration
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


class _RoicRules:
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
