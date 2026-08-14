import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

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
    RedFlagWarning,
)

PeriodKey = tuple[int, FiscalPeriod]


@dataclass(frozen=True)
class _Value:
    value: Decimal
    unit: str
    observations: tuple[FinancialObservation, ...]


class _RuleContext:
    """Shared calculations and output construction used by red-flag rules."""

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
            if existing is None or _RuleContext._observation_key(
                observation
            ) > _RuleContext._observation_key(existing):
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
            tuple(
                observation for value in present for observation in value.observations
            ),
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
        return cls._single(
            observations, FinancialConcept.SHARES_OUTSTANDING
        ) or cls._single(observations, FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES)

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
            operating_income.observations
            + pretax_income.observations
            + tax.observations,
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
            period_end=max(observation.period_end for observation in observations),
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
                required_concepts=tuple(
                    sorted(set(required), key=lambda concept: concept.value)
                ),
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
