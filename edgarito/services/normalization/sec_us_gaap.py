import datetime
from dataclasses import dataclass
from typing import Iterable, Optional

from edgarito.enums.edgar.period import FISCAL_PERIOD_PRIORITY, FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
    ObservationDerivationKind,
)
from edgarito.schemas.providers.edgar.company_facts import CompanyFacts, Measurement


@dataclass(frozen=True)
class ConceptDefinition:
    concept: FinancialConcept
    source_concepts: tuple[str, ...]
    unit: str = "USD"
    instant: bool = False
    taxonomy: str = "us-gaap"
    additive: bool = True
    fallback_component_groups: tuple[tuple[str, ...], ...] = ()


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
        ("OperatingIncomeLoss",),
    ),
    ConceptDefinition(
        FinancialConcept.PRETAX_INCOME,
        (
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
        ),
    ),
    ConceptDefinition(
        FinancialConcept.INCOME_TAX_EXPENSE,
        (
            "IncomeTaxExpenseBenefit",
            "IncomeTaxExpenseBenefitContinuingOperations",
        ),
    ),
    ConceptDefinition(
        FinancialConcept.NET_INCOME,
        ("NetIncomeLoss",),
    ),
    ConceptDefinition(
        FinancialConcept.NET_INCOME_COMMON,
        (
            "NetIncomeLossAvailableToCommonStockholdersBasic",
            "NetIncomeLossAvailableToCommonStockholdersDiluted",
        ),
    ),
    ConceptDefinition(
        FinancialConcept.INTEREST_EXPENSE,
        (
            "InterestExpenseNonoperating",
            "InterestExpenseDebt",
            "InterestExpense",
        ),
    ),
    ConceptDefinition(
        FinancialConcept.TOTAL_ASSETS,
        ("Assets",),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.CURRENT_ASSETS,
        ("AssetsCurrent",),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.ACCOUNTS_RECEIVABLE,
        (
            "AccountsReceivableNetCurrent",
            "AccountsReceivableNet",
        ),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.INVENTORY,
        ("InventoryNet",),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.PREPAID_AND_OTHER_CURRENT_ASSETS,
        (
            "PrepaidExpenseAndOtherAssetsCurrent",
            "OtherAssetsCurrent",
            "PrepaidExpenseCurrent",
            "OtherPrepaidExpenseCurrent",
        ),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.TOTAL_LIABILITIES,
        ("Liabilities",),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.CURRENT_LIABILITIES,
        ("LiabilitiesCurrent",),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.ACCOUNTS_PAYABLE,
        ("AccountsPayableCurrent", "AccountsPayable"),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.ACCRUED_LIABILITIES,
        (
            "AccruedLiabilitiesCurrent",
            "OtherAccruedLiabilitiesCurrent",
            "AccruedLiabilities",
        ),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.DEFERRED_REVENUE_CURRENT,
        (
            "ContractWithCustomerLiabilityCurrent",
            "DeferredRevenueCurrent",
        ),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.SHORT_TERM_DEBT,
        (
            "ShortTermBorrowings",
            "ShortTermDebtCurrent",
            "OtherShortTermBorrowings",
            "CommercialPaper",
        ),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.LONG_TERM_DEBT_CURRENT,
        ("LongTermDebtCurrent",),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.LONG_TERM_DEBT_NONCURRENT,
        ("LongTermDebtNoncurrent",),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.STOCKHOLDERS_EQUITY,
        (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.COMMON_EQUITY,
        ("StockholdersEquity",),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.CASH_AND_EQUIVALENTS,
        (
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.SHORT_TERM_INVESTMENTS,
        ("ShortTermInvestments", "MarketableSecuritiesCurrent"),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.NONCURRENT_INVESTMENTS,
        (
            "LongTermInvestments",
            "EquitySecuritiesWithoutReadilyDeterminableFairValueAmount",
        ),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.GOODWILL,
        ("Goodwill",),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.INTANGIBLE_ASSETS_NET,
        (
            "IntangibleAssetsNetExcludingGoodwill",
            "FiniteLivedIntangibleAssetsNet",
            "OtherIntangibleAssetsNet",
        ),
        instant=True,
    ),
    ConceptDefinition(
        FinancialConcept.OPERATING_CASH_FLOW,
        ("NetCashProvidedByUsedInOperatingActivities",),
    ),
    ConceptDefinition(
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
        (
            "DepreciationDepletionAndAmortization",
            "DepreciationAmortizationAndAccretionNet",
            "DepreciationAndAmortization",
        ),
        fallback_component_groups=(
            ("Depreciation", "AmortizationOfIntangibleAssets"),
            ("Depreciation",),
        ),
    ),
    ConceptDefinition(
        FinancialConcept.CAPITAL_EXPENDITURES,
        ("PaymentsToAcquirePropertyPlantAndEquipment",),
    ),
    ConceptDefinition(
        FinancialConcept.DIVIDENDS_PAID,
        ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends"),
    ),
    ConceptDefinition(
        FinancialConcept.DEBT_ISSUANCE,
        (
            "ProceedsFromIssuanceOfLongTermDebt",
            "ProceedsFromIssuanceOfDebt",
        ),
    ),
    ConceptDefinition(
        FinancialConcept.DEBT_REPAYMENT,
        (
            "RepaymentsOfLongTermDebt",
            "RepaymentsOfDebt",
        ),
    ),
    ConceptDefinition(
        FinancialConcept.DIVIDENDS_PER_SHARE,
        (
            "CommonStockDividendsPerShareCashPaid",
            "CommonStockDividendsPerShareDeclared",
        ),
        unit="USD/shares",
    ),
    ConceptDefinition(
        FinancialConcept.SHARES_OUTSTANDING,
        ("EntityCommonStockSharesOutstanding",),
        unit="shares",
        instant=True,
        taxonomy="dei",
    ),
    ConceptDefinition(
        FinancialConcept.WEIGHTED_AVERAGE_BASIC_SHARES,
        (
            "WeightedAverageNumberOfSharesOutstandingBasic",
            "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
        ),
        unit="shares",
        additive=False,
    ),
    ConceptDefinition(
        FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES,
        (
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
        ),
        unit="shares",
        additive=False,
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

        facts_by_taxonomy = {
            "us-gaap": gaap_facts,
            "dei": company_facts.facts.dei,
        }
        observations: list[FinancialObservation] = []
        for definition in CONCEPT_DEFINITIONS:
            if concepts and definition.concept not in concepts:
                continue

            taxonomy_facts = facts_by_taxonomy[definition.taxonomy]
            selected_periods: set[tuple[Granularity, int, FiscalPeriod]] = set()
            for source_concept in definition.source_concepts:
                fact = taxonomy_facts.get(source_concept)
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

            for component_group in definition.fallback_component_groups:
                for observation in self._normalize_component_fallback(
                    definition,
                    taxonomy_facts,
                    component_group,
                ):
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

    def _normalize_component_fallback(
        self,
        definition: ConceptDefinition,
        taxonomy_facts,
        component_group: tuple[str, ...],
    ) -> list[FinancialObservation]:
        """Compose a canonical concept when the filer reports all atomic parts."""
        by_component: list[
            dict[tuple[Granularity, int, FiscalPeriod], FinancialObservation]
        ] = []
        for source_concept in component_group:
            fact = taxonomy_facts.get(source_concept)
            if fact is None:
                return []
            candidates = self._deduplicate_periods(
                source_concept,
                definition.unit,
                fact.units.get(definition.unit),
            )
            normalized = (
                self._normalize_instant(definition, candidates)
                if definition.instant
                else self._normalize_duration(definition, candidates)
            )
            by_component.append(
                {
                    (
                        observation.granularity,
                        observation.fiscal_year,
                        observation.fiscal_period,
                    ): observation
                    for observation in normalized
                }
            )

        common_periods = set.intersection(
            *(set(component) for component in by_component)
        )
        results = []
        source_label = " + ".join(component_group)
        derivation_kind = (
            ObservationDerivationKind.COMPONENT_AGGREGATION
            if len(component_group) > 1
            else ObservationDerivationKind.CONCEPT_FALLBACK
        )
        for period_key in common_periods:
            components = [component[period_key] for component in by_component]
            if len({component.unit for component in components}) != 1:
                continue
            base = max(
                components,
                key=lambda component: component.filed or datetime.date.min,
            )
            results.append(
                base.model_copy(
                    update={
                        "value": sum(
                            (component.value for component in components),
                            start=0,
                        ),
                        "source_concept": source_label,
                        "derivation_kind": derivation_kind,
                        "derivation": f"{definition.concept.value} = {source_label}",
                    }
                )
            )
        return results

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
        groups: dict[
            tuple[Optional[datetime.date], datetime.date], list[Measurement]
        ] = {}
        for measurement in measurements:
            if measurement.form not in self._FORMS or measurement.fp is None:
                continue
            groups.setdefault((measurement.start, measurement.end), []).append(
                measurement
            )

        candidates = []
        for group in groups.values():
            ordered = sorted(
                group, key=lambda m: (m.filed, m.accn, m.form.endswith("/A"))
            )
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

            if self._is_comparative_fiscal_year_end(candidate, candidates):
                fiscal_year = candidate.identity.end.year
                annual_key = (Granularity.ANNUAL, fiscal_year, FiscalPeriod.FY)
                quarterly_key = (Granularity.QUARTERLY, fiscal_year, FiscalPeriod.Q4)
                self._choose_latest(chosen, annual_key, candidate)
                self._choose_latest(chosen, quarterly_key, candidate)
            elif fiscal_period == FiscalPeriod.FY:
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

    @staticmethod
    def _is_comparative_fiscal_year_end(
        candidate: _Candidate,
        candidates: list[_Candidate],
    ) -> bool:
        """Identify a prior FY balance first disclosed in a quarterly filing."""
        identity = candidate.identity
        if identity.fp not in (FiscalPeriod.Q1, FiscalPeriod.Q2, FiscalPeriod.Q3):
            return False
        return any(
            other.identity.accn == identity.accn
            and other.identity.fy == identity.fy
            and other.identity.fp == identity.fp
            and other.identity.end > identity.end
            for other in candidates
        )

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
                period_candidates = [
                    c for c in year_candidates if c.identity.fp == period
                ]
                discrete = self._pick_discrete(period_candidates)
                if discrete is not None:
                    quarterly[period] = self._to_observation(
                        definition,
                        discrete,
                        Granularity.QUARTERLY,
                        fiscal_year,
                        period,
                    )
                if definition.additive:
                    cumulative = self._pick_ytd(period_candidates)
                    if cumulative is not None:
                        ytd[period] = cumulative

            if (
                FiscalPeriod.Q2 not in quarterly
                and FiscalPeriod.Q2 in ytd
                and FiscalPeriod.Q1 in quarterly
            ):
                quarterly[FiscalPeriod.Q2] = self._derive_observation(
                    definition,
                    ytd[FiscalPeriod.Q2],
                    fiscal_year,
                    FiscalPeriod.Q2,
                    ytd[FiscalPeriod.Q2].measurement.val
                    - quarterly[FiscalPeriod.Q1].value,
                    "Q2 = Q2 YTD - Q1",
                    period_start=quarterly[FiscalPeriod.Q1].period_end
                    + datetime.timedelta(days=1),
                )

            if FiscalPeriod.Q3 not in quarterly and FiscalPeriod.Q3 in ytd:
                if FiscalPeriod.Q2 in ytd:
                    q3_value = (
                        ytd[FiscalPeriod.Q3].measurement.val
                        - ytd[FiscalPeriod.Q2].measurement.val
                    )
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
                        period_start=q2.period_end + datetime.timedelta(days=1)
                        if q2
                        else None,
                    )

            if (
                definition.additive
                and annual_candidate is not None
                and all(
                    period in quarterly
                    for period in (FiscalPeriod.Q1, FiscalPeriod.Q2, FiscalPeriod.Q3)
                )
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
                    period_start=quarterly[FiscalPeriod.Q3].period_end
                    + datetime.timedelta(days=1),
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
            if candidate.duration_days is not None
            and 60 <= candidate.duration_days <= 120
        ]
        return max(discrete, key=lambda c: c.measurement.filed, default=None)

    @staticmethod
    def _pick_ytd(candidates: list[_Candidate]) -> Optional[_Candidate]:
        cumulative = [
            candidate
            for candidate in candidates
            if candidate.duration_days is not None
            and 120 < candidate.duration_days < 300
        ]
        return max(
            cumulative,
            key=lambda c: (c.duration_days or 0, c.measurement.filed),
            default=None,
        )

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
            statement=definition.concept.statement,
            value=measurement.val,
            unit=candidate.unit,
            granularity=granularity,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            period_start=candidate.identity.start,
            period_end=candidate.identity.end,
            provider="sec",
            taxonomy=definition.taxonomy,
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
        observation.derivation_kind = ObservationDerivationKind.PERIOD_RECONSTRUCTION
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
