"""Compare normalized financial data from independent providers."""

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional

from edgarito.enums.edgar.period import FISCAL_PERIOD_PRIORITY, FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)


class CrosscheckIssueKind(str, Enum):
    VALUE_MISMATCH = "value_mismatch"
    UNIT_MISMATCH = "unit_mismatch"
    MISSING_FROM_PRIMARY = "missing_from_primary"
    MISSING_FROM_SECONDARY = "missing_from_secondary"


@dataclass(frozen=True)
class ObservationKey:
    concept: FinancialConcept
    granularity: Granularity
    fiscal_year: int
    fiscal_period: FiscalPeriod

    @classmethod
    def from_observation(cls, observation: FinancialObservation) -> "ObservationKey":
        return cls(
            concept=observation.concept,
            granularity=observation.granularity,
            fiscal_year=observation.fiscal_year,
            fiscal_period=observation.fiscal_period,
        )


@dataclass(frozen=True)
class CrosscheckIssue:
    kind: CrosscheckIssueKind
    key: ObservationKey
    primary: Optional[FinancialObservation] = None
    secondary: Optional[FinancialObservation] = None
    absolute_difference: Optional[Decimal] = None
    relative_difference: Optional[Decimal] = None


@dataclass
class CrosscheckReport:
    primary_provider: str
    secondary_provider: str
    matched_observations: int = 0
    issues: list[CrosscheckIssue] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)

    def summary(self) -> str:
        if not self.issues:
            return (
                f"Crosscheck {self.primary_provider} vs {self.secondary_provider}: "
                f"{self.matched_observations} observations matched"
            )
        counts = Counter(issue.kind for issue in self.issues)
        details = ", ".join(
            f"{count} {kind.value.replace('_', ' ')}"
            for kind, count in sorted(counts.items(), key=lambda item: item[0].value)
        )
        return (
            f"Crosscheck {self.primary_provider} vs {self.secondary_provider}: "
            f"{len(self.issues)} issue(s) ({details}); "
            f"{self.matched_observations} observations matched"
        )


class FinancialsCrosschecker:
    """Compare normalized observations without merging or filling either dataset."""

    def __init__(
        self,
        relative_tolerance: Decimal = Decimal("0.01"),
        absolute_tolerance: Decimal = Decimal("1"),
    ):
        if relative_tolerance < 0 or absolute_tolerance < 0:
            raise ValueError("Crosscheck tolerances cannot be negative")
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance

    def compare(
        self,
        primary: NormalizedCompanyFinancials,
        secondary: NormalizedCompanyFinancials,
    ) -> CrosscheckReport:
        report = CrosscheckReport(primary.provider, secondary.provider)
        primary_observations = self._index(primary)
        secondary_observations = self._index(secondary)
        keys = set(primary_observations) | set(secondary_observations)

        for key in sorted(keys, key=self._key_sort_key):
            primary_observation = primary_observations.get(key)
            secondary_observation = secondary_observations.get(key)
            if primary_observation is None:
                report.issues.append(
                    CrosscheckIssue(
                        CrosscheckIssueKind.MISSING_FROM_PRIMARY,
                        key,
                        secondary=secondary_observation,
                    )
                )
                continue
            if secondary_observation is None:
                report.issues.append(
                    CrosscheckIssue(
                        CrosscheckIssueKind.MISSING_FROM_SECONDARY,
                        key,
                        primary=primary_observation,
                    )
                )
                continue
            if primary_observation.unit.upper() != secondary_observation.unit.upper():
                report.issues.append(
                    CrosscheckIssue(
                        CrosscheckIssueKind.UNIT_MISMATCH,
                        key,
                        primary_observation,
                        secondary_observation,
                    )
                )
                continue

            absolute_difference = abs(
                primary_observation.value - secondary_observation.value
            )
            largest_value = max(
                abs(primary_observation.value), abs(secondary_observation.value)
            )
            relative_difference = (
                absolute_difference / largest_value
                if largest_value
                else Decimal(0)
            )
            allowed_difference = max(
                self.absolute_tolerance,
                largest_value * self.relative_tolerance,
            )
            if absolute_difference > allowed_difference:
                report.issues.append(
                    CrosscheckIssue(
                        CrosscheckIssueKind.VALUE_MISMATCH,
                        key,
                        primary_observation,
                        secondary_observation,
                        absolute_difference,
                        relative_difference,
                    )
                )
                continue
            report.matched_observations += 1

        return report

    @staticmethod
    def _index(
        financials: NormalizedCompanyFinancials,
    ) -> dict[ObservationKey, FinancialObservation]:
        indexed = {}
        for observation in financials.observations:
            key = ObservationKey.from_observation(observation)
            if key in indexed:
                raise ValueError(
                    f"{financials.provider} contains duplicate normalized observation: {key}"
                )
            indexed[key] = observation
        return indexed

    @staticmethod
    def _key_sort_key(key: ObservationKey):
        granularity_order = 0 if key.granularity == Granularity.ANNUAL else 1
        return (
            granularity_order,
            key.fiscal_year,
            FISCAL_PERIOD_PRIORITY[key.fiscal_period],
            key.concept.value,
        )


class FinancialDataCrosscheckWarning(UserWarning):
    """A secondary provider failed or disagreed during automatic crosschecking."""
