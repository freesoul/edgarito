import re
from decimal import Decimal
from typing import Optional

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.classification import (
    NormalizedCompanyClassification,
    Sector,
)
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.valuation.models import (
    BusinessArchetype,
    CompanyLifecycle,
    Cyclicality,
    EconomicTrait,
    ValuationInput,
    ValuationProfile,
    ValuationProfileOverrides,
)


class ValuationProfileBuilder:
    """Infer valuation-relevant economics from normalized provider data."""

    _BANK_OR_INSURER = re.compile(
        r"\b(banks?|banking|insurance|insurer|reinsurance|thrifts?|mortgage finance)\b"
    )
    _ASSET_MANAGER = re.compile(
        r"\b(asset management|investment management|wealth management|fund manager)\b"
    )
    _REIT = re.compile(
        r"\b(reit|real estate investment trust|property trust|equity real estate)\b"
    )
    _RESOURCE = re.compile(
        r"\b(oil.*(exploration|production)|exploration.*production|upstream|"
        r"mining|minerals?|gold|silver|copper|coal|uranium|timber)\b"
    )
    _PIPELINE = re.compile(
        r"\b(biotech|biotechnology|clinical|drug discovery|development stage)\b"
    )
    _HOLDING = re.compile(r"\b(holding company|investment holding)\b")
    _CONGLOMERATE = re.compile(r"\b(conglomerate|diversified operations)\b")
    _HIGH_CYCLICAL = re.compile(
        r"\b(semiconductors?|airlines?|automobiles?|auto manufacturers?|steel|chemicals?|"
        r"mining|oil.*(exploration|production)|commodity)\b"
    )
    _LEASE_INTENSIVE = re.compile(r"\b(retail|airline|restaurant|hotel)\b")
    _BACKLOG_DRIVEN = re.compile(
        r"\b(defense|aerospace|engineering.*construction|government contractor)\b"
    )

    def build(
        self,
        financials: NormalizedCompanyFinancials,
        classification: Optional[NormalizedCompanyClassification] = None,
        overrides: Optional[ValuationProfileOverrides] = None,
    ) -> ValuationProfile:
        overrides = overrides or ValuationProfileOverrides()
        self._validate_same_company(financials, classification)

        sector = overrides.sector or (classification.sector if classification else None)
        industry = overrides.industry or (
            classification.industry if classification else None
        )
        industry_key = self._key(industry)
        notes = []

        inferred_archetype = self._archetype(sector, industry_key)
        archetype = overrides.business_archetype or inferred_archetype
        notes.append(
            f"Business archetype {'overridden' if overrides.business_archetype else 'inferred'} "
            f"as {archetype.value}"
        )

        annual = self._annual_by_year(financials)
        years = tuple(sorted(annual))
        growth_rates = self._revenue_growth_rates(annual)
        fcf_by_year = self._free_cash_flow_by_year(annual)
        earnings = self._concept_values(annual, FinancialConcept.NET_INCOME)
        equities = self._concept_values(annual, FinancialConcept.STOCKHOLDERS_EQUITY)
        latest_equity = equities[-1].value if equities else None

        inferred_lifecycle = self._lifecycle(
            archetype,
            annual,
            growth_rates,
            fcf_by_year,
            earnings,
            latest_equity,
        )
        lifecycle = overrides.lifecycle or inferred_lifecycle
        notes.append(
            f"Lifecycle {'overridden' if overrides.lifecycle else 'inferred'} as "
            f"{lifecycle.value}"
        )

        inferred_cyclicality = self._cyclicality(sector, industry_key)
        cyclicality = overrides.cyclicality or inferred_cyclicality
        notes.append(
            f"Cyclicality {'overridden' if overrides.cyclicality else 'inferred'} "
            f"as {cyclicality.value}"
        )

        traits = self._traits(sector, industry_key, archetype)
        traits.update(overrides.economic_traits)
        available_inputs = self._available_inputs(
            annual, fcf_by_year, earnings, equities, classification
        )
        available_inputs.update(overrides.available_inputs)
        if overrides.peer_count is not None and overrides.peer_count >= 5:
            available_inputs.add(ValuationInput.PEER_SET)

        return ValuationProfile(
            provider=financials.provider,
            company_id=financials.company_id,
            company_name=financials.company_name,
            ticker=financials.ticker,
            sector=sector,
            industry=industry,
            business_archetype=archetype,
            lifecycle=lifecycle,
            cyclicality=cyclicality,
            economic_traits=traits,
            annual_fiscal_years=years,
            revenue_growth_rates=growth_rates,
            positive_fcf_periods=sum(value > 0 for value in fcf_by_year.values()),
            positive_earnings_periods=sum(value.value > 0 for value in earnings),
            latest_book_equity=latest_equity,
            available_inputs=available_inputs,
            peer_count=overrides.peer_count,
            inference_notes=notes,
        )

    @staticmethod
    def _validate_same_company(
        financials: NormalizedCompanyFinancials,
        classification: Optional[NormalizedCompanyClassification],
    ) -> None:
        if classification is None:
            return
        if (
            financials.company_id.isdigit()
            and classification.company_id.isdigit()
            and int(financials.company_id) != int(classification.company_id)
        ):
            raise ValueError(
                "Financials and classification refer to different company IDs"
            )

    @staticmethod
    def _annual_by_year(
        financials: NormalizedCompanyFinancials,
    ) -> dict[int, dict[FinancialConcept, FinancialObservation]]:
        annual: dict[int, dict[FinancialConcept, FinancialObservation]] = {}
        for observation in financials.observations:
            if (
                observation.granularity == Granularity.ANNUAL
                and observation.fiscal_period == FiscalPeriod.FY
            ):
                annual.setdefault(observation.fiscal_year, {}).setdefault(
                    observation.concept, observation
                )
        return annual

    @staticmethod
    def _revenue_growth_rates(
        annual: dict[int, dict[FinancialConcept, FinancialObservation]],
    ) -> tuple[Decimal, ...]:
        growth_rates = []
        years = sorted(annual)
        for previous_year, current_year in zip(years, years[1:], strict=False):
            previous = annual[previous_year].get(FinancialConcept.REVENUE)
            current = annual[current_year].get(FinancialConcept.REVENUE)
            if (
                current_year != previous_year + 1
                or previous is None
                or current is None
                or previous.unit != current.unit
                or previous.value == 0
            ):
                continue
            growth_rates.append(
                (current.value - previous.value) / previous.value * Decimal(100)
            )
        return tuple(growth_rates)

    @staticmethod
    def _free_cash_flow_by_year(
        annual: dict[int, dict[FinancialConcept, FinancialObservation]],
    ) -> dict[int, Decimal]:
        values = {}
        for year, observations in annual.items():
            operating_cash_flow = observations.get(FinancialConcept.OPERATING_CASH_FLOW)
            capital_expenditures = observations.get(
                FinancialConcept.CAPITAL_EXPENDITURES
            )
            if (
                operating_cash_flow is not None
                and capital_expenditures is not None
                and operating_cash_flow.unit == capital_expenditures.unit
            ):
                values[year] = operating_cash_flow.value - capital_expenditures.value
        return values

    @staticmethod
    def _concept_values(
        annual: dict[int, dict[FinancialConcept, FinancialObservation]],
        concept: FinancialConcept,
    ) -> list[FinancialObservation]:
        return [
            annual[year][concept] for year in sorted(annual) if concept in annual[year]
        ]

    @classmethod
    def _archetype(
        cls, sector: Optional[Sector], industry_key: str
    ) -> BusinessArchetype:
        if cls._HOLDING.search(industry_key):
            return BusinessArchetype.HOLDING_COMPANY
        if cls._CONGLOMERATE.search(industry_key):
            return BusinessArchetype.CONGLOMERATE
        if cls._REIT.search(industry_key):
            return BusinessArchetype.REIT_PROPERTY
        if cls._BANK_OR_INSURER.search(industry_key):
            return BusinessArchetype.FINANCIAL_INTERMEDIARY
        if cls._ASSET_MANAGER.search(industry_key):
            return BusinessArchetype.ASSET_MANAGER
        if cls._RESOURCE.search(industry_key):
            return BusinessArchetype.RESOURCE_PRODUCER
        if cls._PIPELINE.search(industry_key):
            return BusinessArchetype.PROJECT_PIPELINE
        return BusinessArchetype.GENERAL_OPERATING

    @classmethod
    def _cyclicality(cls, sector: Optional[Sector], industry_key: str) -> Cyclicality:
        if cls._HIGH_CYCLICAL.search(industry_key):
            return Cyclicality.HIGH
        if sector in {Sector.CONSUMER_STAPLES, Sector.UTILITIES}:
            return Cyclicality.LOW
        if sector in {
            Sector.CONSUMER_DISCRETIONARY,
            Sector.ENERGY,
            Sector.INDUSTRIALS,
            Sector.MATERIALS,
            Sector.REAL_ESTATE,
        }:
            return Cyclicality.MODERATE
        return Cyclicality.UNKNOWN

    @classmethod
    def _traits(
        cls,
        sector: Optional[Sector],
        industry_key: str,
        archetype: BusinessArchetype,
    ) -> set[EconomicTrait]:
        traits = set()
        if archetype == BusinessArchetype.FINANCIAL_INTERMEDIARY:
            traits.add(EconomicTrait.REGULATED_CAPITAL)
        if sector == Sector.UTILITIES:
            traits.add(EconomicTrait.REGULATED_CAPITAL)
        if cls._LEASE_INTENSIVE.search(industry_key):
            traits.add(EconomicTrait.LEASE_INTENSIVE)
        if cls._BACKLOG_DRIVEN.search(industry_key):
            traits.add(EconomicTrait.BACKLOG_DRIVEN)
        if archetype == BusinessArchetype.CONGLOMERATE:
            traits.add(EconomicTrait.MULTI_SEGMENT)
        return traits

    @staticmethod
    def _lifecycle(
        archetype: BusinessArchetype,
        annual: dict[int, dict[FinancialConcept, FinancialObservation]],
        growth_rates: tuple[Decimal, ...],
        fcf_by_year: dict[int, Decimal],
        earnings: list[FinancialObservation],
        latest_equity: Optional[Decimal],
    ) -> CompanyLifecycle:
        revenues = [
            values[FinancialConcept.REVENUE]
            for _, values in sorted(annual.items())
            if FinancialConcept.REVENUE in values
        ]
        latest_revenue = revenues[-1].value if revenues else None
        if archetype == BusinessArchetype.PROJECT_PIPELINE and (
            latest_revenue is None or latest_revenue <= 0
        ):
            return CompanyLifecycle.PRE_REVENUE
        if latest_equity is not None and latest_equity <= 0:
            return CompanyLifecycle.DISTRESSED

        average_growth = (
            sum(growth_rates, Decimal(0)) / len(growth_rates) if growth_rates else None
        )
        latest_earnings = earnings[-1].value if earnings else None
        latest_fcf = fcf_by_year[max(fcf_by_year)] if fcf_by_year else None
        if (
            (
                (latest_earnings is not None and latest_earnings <= 0)
                or (latest_fcf is not None and latest_fcf <= 0)
            )
            and average_growth is not None
            and average_growth > 0
        ):
            return CompanyLifecycle.UNPROFITABLE_GROWTH
        if average_growth is not None and average_growth > Decimal(15):
            return CompanyLifecycle.GROWTH
        if average_growth is not None and average_growth < Decimal("-5"):
            return CompanyLifecycle.DECLINING
        if (
            len(annual) >= 3
            and latest_earnings is not None
            and latest_earnings > 0
            and latest_fcf is not None
            and latest_fcf > 0
        ):
            return CompanyLifecycle.MATURE
        return CompanyLifecycle.UNKNOWN

    @staticmethod
    def _available_inputs(
        annual: dict[int, dict[FinancialConcept, FinancialObservation]],
        fcf_by_year: dict[int, Decimal],
        earnings: list[FinancialObservation],
        equities: list[FinancialObservation],
        classification: Optional[NormalizedCompanyClassification],
    ) -> set[ValuationInput]:
        inputs = set()
        if classification is not None:
            inputs.add(ValuationInput.CLASSIFICATION)
        if any(FinancialConcept.REVENUE in values for values in annual.values()):
            inputs.add(ValuationInput.REVENUE_HISTORY)
        if fcf_by_year:
            inputs.add(ValuationInput.FCF_HISTORY)
        if earnings:
            inputs.add(ValuationInput.EARNINGS_HISTORY)
        if equities:
            inputs.add(ValuationInput.BOOK_EQUITY)
        if any(
            {
                FinancialConcept.TOTAL_ASSETS,
                FinancialConcept.TOTAL_LIABILITIES,
            }
            <= values.keys()
            for values in annual.values()
        ):
            inputs.add(ValuationInput.BALANCE_SHEET)
        return inputs

    @staticmethod
    def _key(value: Optional[str]) -> str:
        return re.sub(r"\s+", " ", value.casefold()).strip() if value else ""
