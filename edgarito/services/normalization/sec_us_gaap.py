import datetime
from dataclasses import dataclass
from typing import Iterable, Optional

from edgarito.enums.edgar.period import FISCAL_PERIOD_PRIORITY, FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    FinancialStatement,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.providers.edgar.company_facts import CompanyFacts, Measurement


@dataclass(frozen=True)
class ConceptDefinition:
    concept: FinancialConcept
    statement: FinancialStatement
    source_concepts: tuple[str, ...]
    unit: str = "USD"
    instant: bool = False


@dataclass(frozen=True)
class _Candidate:
    source_concept: str
    unit: str
    identity: Measurement
    measurement: Measurement

    @property
    def duration_days(self) -> Optional[int]:
        if self.identity.start is None:
            return None
        return (self.identity.end - self.identity.start).days

    @property
    def fiscal_year(self) -> Optional[int]:
        if self.identity.fp == FiscalPeriod.FY:
            return self.identity.end.year
        return self.identity.fy


CONCEPT_DEFINITIONS = (
    ConceptDefinition(
        FinancialConcept.REVENUE,
        FinancialStatement.INCOME_STATEMENT,
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
            "RevenuesNetOfInterestExpense",
        ),
    ),
    ConceptDefinition(
        FinancialConcept.OPERATING_INCOME,
        FinancialStatement.INCOME_STATEMENT,
        ("OperatingIncomeLoss",),
    ),
    ConceptDefinition(
        FinancialConcept.NET_INCOME,
        FinancialStatement.INCOME_STATEMENT,
        ("NetIncomeLoss",),
    ),
    ConceptDefinition(
        FinancialConcept.TOTAL_ASSETS,
        FinancialStatement.BALANCE_SHEET,
        ("Assets",),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.TOTAL_LIABILITIES,
        FinancialStatement.BALANCE_SHEET,
        ("Liabilities",),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.STOCKHOLDERS_EQUITY,
        FinancialStatement.BALANCE_SHEET,
        (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.CASH_AND_EQUIVALENTS,
        FinancialStatement.BALANCE_SHEET,
        (
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.OPERATING_CASH_FLOW,
        FinancialStatement.CASH_FLOW,
        ("NetCashProvidedByUsedInOperatingActivities",),
    ),
    ConceptDefinition(
        FinancialConcept.CAPITAL_EXPENDITURES,
        FinancialStatement.CASH_FLOW,
        ("PaymentsToAcquirePropertyPlantAndEquipment",),
    ),
)


class SecUsGaapNormalizer:
    """Normalize SEC Company Facts into provider-neutral financial observations."""

    _FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}

    def normalize(
        self,
        company_facts: CompanyFacts,
        ticker: Optional[str] = None,
        granularity: Optional[Granularity] = None,
        concepts: Optional[set[FinancialConcept]] = None,
    ) -> NormalizedCompanyFinancials:
        gaap_facts = company_facts.facts.us_gaap
        if not gaap_facts:
            raise ValueError("The SEC response does not contain US-GAAP facts")

        observations: list[FinancialObservation] = []
        for definition in CONCEPT_DEFINITIONS:
            if concepts and definition.concept not in concepts:
                continue

            selected_periods: set[tuple[Granularity, int, FiscalPeriod]] = set()
            for source_concept in definition.source_concepts:
                fact = gaap_facts.get(source_concept)
                if fact is None:
                    continue

                candidates = self._deduplicate_periods(
                    source_concept,
                    definition.unit,
                    fact.units.get(definition.unit),
                )
                source_observations = (
                    self._normalize_instant(definition, candidates)
                    if definition.instant
                    else self._normalize_duration(definition, candidates)
                )

                for observation in source_observations:
                    period_key = (
                        observation.granularity,
                        observation.fiscal_year,
                        observation.fiscal_period,
                    )
                    if period_key not in selected_periods:
                        observations.append(observation)
                        selected_periods.add(period_key)

        if granularity is not None:
            observations = [o for o in observations if o.granularity == granularity]

        observations.sort(key=self._observation_sort_key)
        return NormalizedCompanyFinancials(
            provider="sec",
            company_id=str(company_facts.cik).zfill(10),
            company_name=company_facts.entityName,
            ticker=ticker.upper() if ticker else None,
            observations=observations,
        )

    def _deduplicate_periods(
        self,
        source_concept: str,
        unit: str,
        measurements: Iterable[Measurement],
    ) -> list[_Candidate]:
        """
        Group repeated facts by their economic period.

        The first filing supplies the original fiscal identity; the latest filing
        supplies a potentially restated value. This prevents later comparative
        filings from moving an older fact into the later filing's fiscal year.
        """
        groups: dict[tuple[Optional[datetime.date], datetime.date], list[Measurement]] = {}
        for measurement in measurements:
            if measurement.form not in self._FORMS or measurement.fp is None:
                continue
            groups.setdefault((measurement.start, measurement.end), []).append(measurement)

        candidates = []
        for group in groups.values():
            ordered = sorted(group, key=lambda m: (m.filed, m.accn, m.form.endswith("/A")))
            identity = ordered[0]
            measurement = ordered[-1]
            if identity.fy is None and identity.fp != FiscalPeriod.FY:
                continue
            candidates.append(_Candidate(source_concept, unit, identity, measurement))
        return candidates

    def _normalize_instant(
        self,
        definition: ConceptDefinition,
        candidates: list[_Candidate],
    ) -> list[FinancialObservation]:
        results = []
        chosen: dict[tuple[Granularity, int, FiscalPeriod], _Candidate] = {}

        for candidate in candidates:
            fiscal_year = candidate.fiscal_year
            fiscal_period = candidate.identity.fp
            if fiscal_year is None or fiscal_period is None:
                continue

            if fiscal_period == FiscalPeriod.FY:
                annual_key = (Granularity.ANNUAL, fiscal_year, FiscalPeriod.FY)
                quarterly_key = (Granularity.QUARTERLY, fiscal_year, FiscalPeriod.Q4)
                self._choose_latest(chosen, annual_key, candidate)
                self._choose_latest(chosen, quarterly_key, candidate)
            elif fiscal_period in (FiscalPeriod.Q1, FiscalPeriod.Q2, FiscalPeriod.Q3):
                key = (Granularity.QUARTERLY, fiscal_year, fiscal_period)
                self._choose_latest(chosen, key, candidate)

        for (granularity, fiscal_year, fiscal_period), candidate in chosen.items():
            results.append(
                self._to_observation(
                    definition,
                    candidate,
                    granularity,
                    fiscal_year,
                    fiscal_period,
                )
            )
        return results

    def _normalize_duration(
        self,
        definition: ConceptDefinition,
        candidates: list[_Candidate],
    ) -> list[FinancialObservation]:
        results: list[FinancialObservation] = []
        by_year: dict[int, list[_Candidate]] = {}
        for candidate in candidates:
            if candidate.fiscal_year is not None:
                by_year.setdefault(candidate.fiscal_year, []).append(candidate)

        for fiscal_year, year_candidates in by_year.items():
            annual_candidate = self._pick_annual(year_candidates)
            if annual_candidate is not None:
                results.append(
                    self._to_observation(
                        definition,
                        annual_candidate,
                        Granularity.ANNUAL,
                        fiscal_year,
                        FiscalPeriod.FY,
                    )
                )

            quarterly: dict[FiscalPeriod, FinancialObservation] = {}
            ytd: dict[FiscalPeriod, _Candidate] = {}
            for period in (FiscalPeriod.Q1, FiscalPeriod.Q2, FiscalPeriod.Q3):
                period_candidates = [c for c in year_candidates if c.identity.fp == period]
                discrete = self._pick_discrete(period_candidates)
                if discrete is not None:
                    quarterly[period] = self._to_observation(
                        definition,
                        discrete,
                        Granularity.QUARTERLY,
                        fiscal_year,
                        period,
                    )
                cumulative = self._pick_ytd(period_candidates)
                if cumulative is not None:
                    ytd[period] = cumulative

            if FiscalPeriod.Q2 not in quarterly and FiscalPeriod.Q2 in ytd and FiscalPeriod.Q1 in quarterly:
                quarterly[FiscalPeriod.Q2] = self._derive_observation(
                    definition,
                    ytd[FiscalPeriod.Q2],
                    fiscal_year,
                    FiscalPeriod.Q2,
                    ytd[FiscalPeriod.Q2].measurement.val - quarterly[FiscalPeriod.Q1].value,
                    "Q2 = Q2 YTD - Q1",
                    period_start=quarterly[FiscalPeriod.Q1].period_end + datetime.timedelta(days=1),
                )

            if FiscalPeriod.Q3 not in quarterly and FiscalPeriod.Q3 in ytd:
                if FiscalPeriod.Q2 in ytd:
                    q3_value = ytd[FiscalPeriod.Q3].measurement.val - ytd[FiscalPeriod.Q2].measurement.val
                    derivation = "Q3 = Q3 YTD - Q2 YTD"
                elif FiscalPeriod.Q1 in quarterly and FiscalPeriod.Q2 in quarterly:
                    q3_value = (
                        ytd[FiscalPeriod.Q3].measurement.val
                        - quarterly[FiscalPeriod.Q1].value
                        - quarterly[FiscalPeriod.Q2].value
                    )
                    derivation = "Q3 = Q3 YTD - Q1 - Q2"
                else:
                    q3_value = None
                    derivation = ""
                if q3_value is not None:
                    q2 = quarterly.get(FiscalPeriod.Q2)
                    quarterly[FiscalPeriod.Q3] = self._derive_observation(
                        definition,
                        ytd[FiscalPeriod.Q3],
                        fiscal_year,
                        FiscalPeriod.Q3,
                        q3_value,
                        derivation,
                        period_start=q2.period_end + datetime.timedelta(days=1) if q2 else None,
                    )

            if annual_candidate is not None and all(
                period in quarterly for period in (FiscalPeriod.Q1, FiscalPeriod.Q2, FiscalPeriod.Q3)
            ):
                q4_value = annual_candidate.measurement.val - sum(
                    quarterly[period].value
                    for period in (FiscalPeriod.Q1, FiscalPeriod.Q2, FiscalPeriod.Q3)
                )
                quarterly[FiscalPeriod.Q4] = self._derive_observation(
                    definition,
                    annual_candidate,
                    fiscal_year,
                    FiscalPeriod.Q4,
                    q4_value,
                    "Q4 = FY - Q1 - Q2 - Q3",
                    period_start=quarterly[FiscalPeriod.Q3].period_end + datetime.timedelta(days=1),
                )

            results.extend(quarterly.values())

        return results

    @staticmethod
    def _choose_latest(
        chosen: dict[tuple[Granularity, int, FiscalPeriod], _Candidate],
        key: tuple[Granularity, int, FiscalPeriod],
        candidate: _Candidate,
    ) -> None:
        existing = chosen.get(key)
        if existing is None or candidate.measurement.filed > existing.measurement.filed:
            chosen[key] = candidate

    @staticmethod
    def _pick_annual(candidates: list[_Candidate]) -> Optional[_Candidate]:
        annual = [
            candidate
            for candidate in candidates
            if candidate.identity.fp == FiscalPeriod.FY
            and candidate.duration_days is not None
            and 300 <= candidate.duration_days <= 400
        ]
        return max(annual, key=lambda c: c.measurement.filed, default=None)

    @staticmethod
    def _pick_discrete(candidates: list[_Candidate]) -> Optional[_Candidate]:
        discrete = [
            candidate
            for candidate in candidates
            if candidate.duration_days is not None and 60 <= candidate.duration_days <= 120
        ]
        return max(discrete, key=lambda c: c.measurement.filed, default=None)

    @staticmethod
    def _pick_ytd(candidates: list[_Candidate]) -> Optional[_Candidate]:
        cumulative = [
            candidate
            for candidate in candidates
            if candidate.duration_days is not None and 120 < candidate.duration_days < 300
        ]
        return max(cumulative, key=lambda c: (c.duration_days or 0, c.measurement.filed), default=None)

    @staticmethod
    def _to_observation(
        definition: ConceptDefinition,
        candidate: _Candidate,
        granularity: Granularity,
        fiscal_year: int,
        fiscal_period: FiscalPeriod,
    ) -> FinancialObservation:
        measurement = candidate.measurement
        return FinancialObservation(
            concept=definition.concept,
            statement=definition.statement,
            value=measurement.val,
            unit=candidate.unit,
            granularity=granularity,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            period_start=candidate.identity.start,
            period_end=candidate.identity.end,
            provider="sec",
            taxonomy="us-gaap",
            source_concept=candidate.source_concept,
            accession_number=measurement.accn,
            form=measurement.form,
            filed=measurement.filed,
        )

    @classmethod
    def _derive_observation(
        cls,
        definition: ConceptDefinition,
        candidate: _Candidate,
        fiscal_year: int,
        fiscal_period: FiscalPeriod,
        value,
        derivation: str,
        period_start: Optional[datetime.date],
    ) -> FinancialObservation:
        observation = cls._to_observation(
            definition,
            candidate,
            Granularity.QUARTERLY,
            fiscal_year,
            fiscal_period,
        )
        observation.value = value
        observation.period_start = period_start
        observation.is_derived = True
        observation.derivation = derivation
        return observation

    @staticmethod
    def _observation_sort_key(observation: FinancialObservation):
        granularity_order = 0 if observation.granularity == Granularity.ANNUAL else 1
        return (
            granularity_order,
            observation.fiscal_year,
            FISCAL_PERIOD_PRIORITY[observation.fiscal_period],
            observation.concept.value,
        )
