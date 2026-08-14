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
from edgarito.schemas.valuation.selection import (
    BusinessArchetype,
    CompanyLifecycle,
    Cyclicality,
    EconomicTrait,
    FinancialInstitutionKind,
    ValuationInput,
    ValuationProfile,
    ValuationProfileOverrides,
)
from edgarito.services.metrics import FinancialMetric, FinancialMetricsService


class ValuationProfileBuilder:
    """Infer valuation-relevant economics from normalized provider data."""

    _BANK_OR_INSURER = re.compile(
        r"\b(banks?|banking|insurance|insurer|reinsurance|thrifts?|mortgage finance)\b"
    )
    _BANK = re.compile(r"\b(banks?|banking|thrifts?|mortgage finance)\b")
    _INSURER = re.compile(r"\b(insurance|insurer|reinsurance)\b")
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

    @staticmethod
    def required_concepts() -> set[FinancialConcept]:
        concepts = {
            FinancialConcept.REVENUE,
            FinancialConcept.NET_INCOME,
            FinancialConcept.NET_INCOME_COMMON,
            FinancialConcept.OPERATING_CASH_FLOW,
            FinancialConcept.CAPITAL_EXPENDITURES,
            FinancialConcept.STOCKHOLDERS_EQUITY,
            FinancialConcept.COMMON_EQUITY,
            FinancialConcept.DIVIDENDS_PAID,
            FinancialConcept.TOTAL_ASSETS,
            FinancialConcept.TOTAL_LIABILITIES,
            FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES,
        }
        concepts.update(
            FinancialMetricsService.required_concepts(
                {
                    FinancialMetric.NET_DEBT,
                    FinancialMetric.TANGIBLE_BOOK_EQUITY,
                }
            )
        )
        return concepts

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
        institution_kind = (
            overrides.financial_institution_kind
            or self._institution_kind(industry_key, archetype)
        )
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
        revenues = self._concept_values(annual, FinancialConcept.REVENUE)
        latest_revenue = revenues[-1] if revenues else None

        inferred_lifecycle = self._lifecycle(
            archetype,
            annual,
            growth_rates,
            fcf_by_year,
            earnings,
            latest_equity,
        )
        has_lifecycle_override = overrides.lifecycle not in {
            None,
            CompanyLifecycle.UNKNOWN,
        }
        lifecycle = (
            overrides.lifecycle if has_lifecycle_override else inferred_lifecycle
        )
        notes.append(
            f"Lifecycle {'overridden' if has_lifecycle_override else 'inferred'} as "
            f"{lifecycle.value}"
        )

        inferred_cyclicality = self._cyclicality(sector, industry_key)
        has_cyclicality_override = overrides.cyclicality not in {
            None,
            Cyclicality.UNKNOWN,
        }
        cyclicality = (
            overrides.cyclicality if has_cyclicality_override else inferred_cyclicality
        )
        notes.append(
            f"Cyclicality {'overridden' if has_cyclicality_override else 'inferred'} "
            f"as {cyclicality.value}"
        )

        traits = self._traits(sector, industry_key, archetype)
        dividend_traits = self._dividend_traits(annual)
        traits.update(dividend_traits)
        traits.update(overrides.economic_traits)
        valuation_metrics = FinancialMetricsService().calculate(
            financials,
            granularity=Granularity.ANNUAL,
            metrics={
                FinancialMetric.NET_DEBT,
                FinancialMetric.TANGIBLE_BOOK_EQUITY,
            },
        )
        available_inputs = self._available_inputs(
            annual,
            fcf_by_year,
            earnings,
            equities,
            classification,
            {
                observation.metric
                for observation in valuation_metrics.observations
                if years and observation.fiscal_year == years[-1]
            },
        )
        available_inputs.update(overrides.available_inputs)
        if overrides.peer_count is not None and overrides.peer_count >= 5:
            available_inputs.add(ValuationInput.PEER_SET)

        return ValuationProfile(
            provider=financials.provider,
            company_id=financials.company_id,
            company_name=financials.company_name,
            ticker=financials.ticker,
            identifiers=(financials.identifiers or classification.identifiers)
            if classification is not None
            else financials.identifiers,
            sector=sector,
            industry=industry,
            country=classification.country if classification else None,
            exchange=classification.exchange if classification else None,
            reporting_currency=latest_revenue.unit if latest_revenue else None,
            latest_revenue=latest_revenue.value if latest_revenue else None,
            business_archetype=archetype,
            financial_institution_kind=institution_kind,
            actuarial_detail_supplied=overrides.actuarial_detail_supplied,
            regulatory_capital_constraints_supplied=(
                overrides.regulatory_capital_constraints_supplied
            ),
            lifecycle=lifecycle,
            cyclicality=cyclicality,
            economic_traits=traits,
            evidence_group=overrides.evidence_group,
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
        if sector in {Sector.FINANCIALS, Sector.REAL_ESTATE}:
            return BusinessArchetype.UNRESOLVED
        return BusinessArchetype.GENERAL_OPERATING

    @classmethod
    def _institution_kind(
        cls, industry_key: str, archetype: BusinessArchetype
    ) -> FinancialInstitutionKind:
        if archetype != BusinessArchetype.FINANCIAL_INTERMEDIARY:
            return FinancialInstitutionKind.OTHER
        if cls._BANK.search(industry_key):
            return FinancialInstitutionKind.BANK
        if cls._INSURER.search(industry_key):
            return FinancialInstitutionKind.INSURER
        return FinancialInstitutionKind.OTHER

    @staticmethod
    def _dividend_traits(
        annual: dict[int, dict[FinancialConcept, FinancialObservation]],
    ) -> set[EconomicTrait]:
        positive: list[tuple[int, Decimal]] = []
        payouts: list[Decimal] = []
        for year, observations in sorted(annual.items()):
            dividend = observations.get(FinancialConcept.DIVIDENDS_PAID)
            income = observations.get(
                FinancialConcept.NET_INCOME_COMMON
            ) or observations.get(FinancialConcept.NET_INCOME)
            if dividend is None or dividend.value <= 0:
                continue
            positive.append((year, dividend.value))
            if income is not None and income.value > 0 and income.unit == dividend.unit:
                payouts.append(dividend.value / income.value)
        traits: set[EconomicTrait] = set()
        if positive:
            traits.add(EconomicTrait.DIVIDEND_PAYER)
        if len(positive) >= 3:
            recent = positive[-3:]
            consecutive = tuple(item[0] for item in recent) == tuple(
                range(recent[0][0], recent[0][0] + 3)
            )
            recent_payouts = payouts[-3:]
            if (
                consecutive
                and len(recent_payouts) == 3
                and max(recent_payouts) - min(recent_payouts) <= Decimal("0.20")
            ):
                traits.add(EconomicTrait.STABLE_PAYOUT)
        return traits

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

        recent_growth = growth_rates[-3:]
        average_growth = (
            sum(recent_growth, Decimal(0)) / len(recent_growth)
            if recent_growth
            else None
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
        available_metrics: set[FinancialMetric],
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
        dividend_years = [
            year
            for year, values in sorted(annual.items())
            if (dividend := values.get(FinancialConcept.DIVIDENDS_PAID)) is not None
            and dividend.value > 0
        ]
        if dividend_years:
            inputs.add(ValuationInput.DIVIDEND_HISTORY)
        if len(dividend_years) >= 3 and dividend_years[-3:] == list(
            range(dividend_years[-3], dividend_years[-3] + 3)
        ):
            clean_payouts = []
            for year in dividend_years[-3:]:
                values = annual[year]
                income = values.get(FinancialConcept.NET_INCOME_COMMON) or values.get(
                    FinancialConcept.NET_INCOME
                )
                dividend = values[FinancialConcept.DIVIDENDS_PAID]
                if (
                    income is not None
                    and income.value > 0
                    and income.unit == dividend.unit
                ):
                    clean_payouts.append(dividend.value / income.value)
            if len(clean_payouts) == 3 and max(clean_payouts) - min(
                clean_payouts
            ) <= Decimal("0.20"):
                inputs.update(
                    {ValuationInput.PAYOUT_POLICY, ValuationInput.DIVIDEND_FORECAST}
                )
        if equities:
            inputs.add(ValuationInput.BOOK_EQUITY)
        clean_roe_years = []
        for year, values in sorted(annual.items()):
            income = values.get(FinancialConcept.NET_INCOME_COMMON) or values.get(
                FinancialConcept.NET_INCOME
            )
            equity = values.get(FinancialConcept.COMMON_EQUITY) or values.get(
                FinancialConcept.STOCKHOLDERS_EQUITY
            )
            if (
                income is not None
                and equity is not None
                and income.value > 0
                and equity.value > 0
                and income.unit == equity.unit
            ):
                clean_roe_years.append(year)
        if len(clean_roe_years) >= 3 and clean_roe_years[-3:] == list(
            range(clean_roe_years[-3], clean_roe_years[-3] + 3)
        ):
            inputs.add(ValuationInput.FORECAST_ROE)
        if any(FinancialConcept.COMMON_EQUITY in values for values in annual.values()):
            inputs.add(ValuationInput.COMMON_EQUITY)
        latest = annual[max(annual)] if annual else {}
        diluted_shares = latest.get(FinancialConcept.WEIGHTED_AVERAGE_DILUTED_SHARES)
        if diluted_shares is not None and diluted_shares.unit == "shares":
            inputs.add(ValuationInput.DILUTED_SHARES)
        if FinancialMetric.NET_DEBT in available_metrics:
            inputs.add(ValuationInput.NET_DEBT)
        if FinancialMetric.TANGIBLE_BOOK_EQUITY in available_metrics:
            inputs.add(ValuationInput.TANGIBLE_BOOK_EQUITY)
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
