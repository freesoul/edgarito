from dataclasses import dataclass

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.schemas.providers.edgar.company_facts import CompanyFacts, Measurement
from edgarito.schemas.valuation.specialized import (
    ExtractedFieldOrigin,
    ExtractedValuationField,
    ExtractionPeriodKind,
    ExtractionReadiness,
    SpecializedInputType,
    SpecializedValuationExtraction,
)


@dataclass(frozen=True)
class _FieldDefinition:
    name: str
    source_concepts: tuple[str, ...]
    units: tuple[str, ...] = ("USD",)


class _SecSpecializedExtractor:
    input_type: SpecializedInputType
    definitions: tuple[_FieldDefinition, ...] = ()
    missing_inputs: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def extract(
        self,
        company_facts: CompanyFacts,
        *,
        ticker: str | None = None,
        historical_periods: int = 5,
    ) -> SpecializedValuationExtraction:
        if historical_periods < 1:
            raise ValueError("historical_periods must be at least 1")
        gaap = company_facts.facts.us_gaap or {}
        fields = [
            field
            for definition in self.definitions
            for field in self._reported_fields(gaap, definition)
        ]
        fields.extend(self._derived_fields(fields))
        latest_period_ends = sorted({field.period_end for field in fields})[
            -historical_periods:
        ]
        fields = [field for field in fields if field.period_end in latest_period_ends]
        fields.sort(key=lambda item: (item.period_end, item.fiscal_year, item.name))
        return SpecializedValuationExtraction(
            provider="sec",
            company_id=str(company_facts.cik).zfill(10),
            company_name=company_facts.entityName,
            ticker=ticker.upper() if ticker else None,
            input_type=self.input_type,
            readiness=(
                ExtractionReadiness.PARTIAL if fields else ExtractionReadiness.BLOCKED
            ),
            fields=tuple(fields),
            missing_inputs=self.missing_inputs,
            limitations=self.limitations,
        )

    def _derived_fields(
        self, fields: list[ExtractedValuationField]
    ) -> list[ExtractedValuationField]:
        return []

    @classmethod
    def _reported_fields(
        cls,
        facts,
        definition: _FieldDefinition,
    ) -> list[ExtractedValuationField]:
        selected: dict[tuple[int, FiscalPeriod], tuple[str, str, Measurement]] = {}
        for source_concept in definition.source_concepts:
            fact = facts.get(source_concept)
            if fact is None:
                continue
            for unit in definition.units:
                measurements = fact.units.get(unit)
                if not measurements:
                    continue
                by_period: dict[tuple[int, FiscalPeriod], list[Measurement]] = {}
                for measurement in measurements:
                    if (
                        measurement.form not in {"10-K", "10-K/A", "10-Q", "10-Q/A"}
                        or measurement.fp is None
                    ):
                        continue
                    fiscal_year = (
                        measurement.end.year
                        if measurement.fp == FiscalPeriod.FY
                        else measurement.fy or measurement.end.year
                    )
                    by_period.setdefault((fiscal_year, measurement.fp), []).append(
                        measurement
                    )
                for period_key, period_measurements in by_period.items():
                    if period_key in selected:
                        continue
                    latest = max(
                        period_measurements,
                        key=lambda item: (
                            item.end,
                            cls._duration_days(item),
                            item.filed,
                            item.form.endswith("/A"),
                            item.accn,
                        ),
                    )
                    selected[period_key] = (source_concept, unit, latest)
                break

        return [
            ExtractedValuationField(
                name=definition.name,
                value=measurement.val,
                unit=unit,
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period.value,
                period_kind=cls._period_kind(measurement),
                period_end=measurement.end,
                origin=ExtractedFieldOrigin.REPORTED,
                source_concepts=(source_concept,),
                accession_numbers=(measurement.accn,),
            )
            for (fiscal_year, fiscal_period), (
                source_concept,
                unit,
                measurement,
            ) in selected.items()
        ]

    @staticmethod
    def _duration_days(measurement: Measurement) -> int:
        return (
            (measurement.end - measurement.start).days
            if measurement.start is not None
            else 0
        )

    @classmethod
    def _period_kind(cls, measurement: Measurement) -> ExtractionPeriodKind:
        if measurement.fp == FiscalPeriod.FY:
            return ExtractionPeriodKind.ANNUAL
        if measurement.start is None:
            return ExtractionPeriodKind.INSTANT
        if cls._duration_days(measurement) > 120:
            return ExtractionPeriodKind.YEAR_TO_DATE
        return ExtractionPeriodKind.QUARTERLY


class ReitInputExtractor(_SecSpecializedExtractor):
    input_type = SpecializedInputType.REIT
    definitions = (
        _FieldDefinition("net_income", ("NetIncomeLoss",)),
        _FieldDefinition(
            "reported_depreciation_and_amortization",
            (
                "DepreciationDepletionAndAmortization",
                "DepreciationAmortizationAndAccretionNet",
                "DepreciationAndAmortization",
                "Depreciation",
            ),
        ),
        _FieldDefinition(
            "gain_loss_on_property_sales",
            (
                "GainLossOnSaleOfRealEstate",
                "GainLossOnSaleOfPropertyPlantEquipment",
                "GainLossOnSaleOfProperties",
            ),
        ),
        _FieldDefinition(
            "property_impairment",
            ("ImpairmentOfRealEstate", "AssetImpairmentCharges"),
        ),
    )
    missing_inputs = (
        "NAREIT-defined real-estate depreciation and noncontrolling adjustments",
        "straight-line rent adjustment",
        "recurring building and tenant capital expenditures",
        "leasing commissions and tenant improvements",
        "other company-specific AFFO adjustments",
    )
    limitations = (
        "Company Facts does not identify whether reported D&A is exclusively "
        "attributable to depreciable real estate",
        "The derived FFO proxy is not labeled NAREIT FFO or AFFO",
    )

    def _derived_fields(
        self, fields: list[ExtractedValuationField]
    ) -> list[ExtractedValuationField]:
        by_name_period = {
            (field.name, field.fiscal_year, field.fiscal_period): field
            for field in fields
        }
        periods = sorted({(field.fiscal_year, field.fiscal_period) for field in fields})
        derived = []
        for fiscal_year, fiscal_period in periods:
            net_income = by_name_period.get(("net_income", fiscal_year, fiscal_period))
            depreciation = by_name_period.get(
                (
                    "reported_depreciation_and_amortization",
                    fiscal_year,
                    fiscal_period,
                )
            )
            if (
                net_income is None
                or depreciation is None
                or net_income.unit != depreciation.unit
            ):
                continue
            gain = by_name_period.get(
                ("gain_loss_on_property_sales", fiscal_year, fiscal_period)
            )
            impairment = by_name_period.get(
                ("property_impairment", fiscal_year, fiscal_period)
            )
            compatible_gain = gain if gain and gain.unit == net_income.unit else None
            compatible_impairment = (
                impairment
                if impairment and impairment.unit == net_income.unit
                else None
            )
            components = [net_income, depreciation]
            if compatible_gain:
                components.append(compatible_gain)
            if compatible_impairment:
                components.append(compatible_impairment)
            value = net_income.value + depreciation.value
            if compatible_gain:
                value -= compatible_gain.value
            if compatible_impairment:
                value += compatible_impairment.value
            derived.append(
                ExtractedValuationField(
                    name="ffo_proxy",
                    value=value,
                    unit=net_income.unit,
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period,
                    period_kind=net_income.period_kind,
                    period_end=max(item.period_end for item in components),
                    origin=ExtractedFieldOrigin.DERIVED_PROXY,
                    source_concepts=tuple(
                        concept
                        for item in components
                        for concept in item.source_concepts
                    ),
                    accession_numbers=tuple(
                        dict.fromkeys(
                            accession
                            for item in components
                            for accession in item.accession_numbers
                        )
                    ),
                    derivation=(
                        "net income + reported D&A + reported property impairment "
                        "- reported property-sale gains"
                    ),
                )
            )
        return derived


class ResourceInputExtractor(_SecSpecializedExtractor):
    input_type = SpecializedInputType.RESOURCE
    definitions = (
        _FieldDefinition("exploration_expense", ("ExplorationExpense",)),
        _FieldDefinition(
            "capitalized_exploratory_well_costs",
            ("CapitalizedExploratoryWellCosts",),
        ),
        _FieldDefinition(
            "exploratory_well_cost_additions",
            (
                "CapitalizedExploratoryWellCostAdditionsPendingDeterminationOfProvedReserves",
            ),
        ),
        _FieldDefinition(
            "depreciation_depletion_and_amortization",
            ("DepreciationDepletionAndAmortization",),
        ),
        _FieldDefinition(
            "asset_retirement_obligation",
            ("AssetRetirementObligation", "AssetRetirementObligationLiability"),
        ),
        _FieldDefinition(
            "capital_expenditures",
            ("PaymentsToAcquirePropertyPlantAndEquipment",),
        ),
    )
    missing_inputs = (
        "reserve quantities by commodity, geography, and reserve class",
        "production volumes and depletion schedule",
        "realized and scenario commodity prices",
        "lifting, processing, transport, royalty, and tax costs",
        "development capex by asset and closure timing",
    )
    limitations = (
        "SEC Company Facts removes the dimensional reserve disclosures needed for "
        "asset-level NAV",
        "Corporate exploration and capitalized-cost facts cannot replace a reserve model",
    )


class BiotechInputExtractor(_SecSpecializedExtractor):
    input_type = SpecializedInputType.BIOTECH
    definitions = (
        _FieldDefinition(
            "research_and_development_expense",
            (
                "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
                "ResearchAndDevelopmentExpense",
            ),
        ),
        _FieldDefinition(
            "acquired_in_process_research_and_development",
            (
                "AcquiredInProcessResearchAndDevelopmentExpense",
                "InProcessResearchAndDevelopmentExpense",
            ),
        ),
        _FieldDefinition(
            "cash_and_equivalents",
            ("CashAndCashEquivalentsAtCarryingValue",),
        ),
    )
    missing_inputs = (
        "program and indication names",
        "clinical phase, trial status, and milestone dates",
        "technical and regulatory success probabilities",
        "launch timing, addressable population, pricing, and peak sales",
        "patent or exclusivity expiry and program-specific remaining costs",
    )
    limitations = (
        "Pipeline programs and clinical probabilities are narrative or table data, "
        "not standard Company Facts concepts",
        "Corporate R&D expense is not attributable to individual programs",
    )


class SotpInputExtractor(_SecSpecializedExtractor):
    input_type = SpecializedInputType.SOTP
    definitions = (
        _FieldDefinition(
            "reportable_segment_count",
            ("NumberOfReportableSegments", "NumberOfOperatingSegments"),
            units=("segment", "Segment", "pure"),
        ),
        _FieldDefinition(
            "total_reportable_segment_revenue",
            ("SegmentReportingInformationRevenue",),
        ),
        _FieldDefinition(
            "total_reportable_segment_profit_loss",
            (
                "SegmentReportingSegmentOperatingProfitLoss",
                "SegmentReportingInformationProfitLoss",
            ),
        ),
        _FieldDefinition(
            "total_reportable_segment_assets",
            (
                "SegmentReportingSegmentAssets",
                "SegmentReportingInformationAssets",
            ),
        ),
    )
    missing_inputs = (
        "named reportable segments",
        "revenue, profit, assets, capex, and cash flow by segment",
        "segment-specific growth, margin, risk, and valuation assumptions",
        "corporate costs, intersegment eliminations, and nonoperating assets",
    )
    limitations = (
        "Company Facts generally retains consolidated segment totals but removes "
        "the dimensional members identifying individual segments",
        "Consolidated segment totals are insufficient for a sum-of-the-parts valuation",
    )


class SpecializedValuationExtractor:
    """Dispatch SEC Company Facts to one specialized input extractor."""

    _EXTRACTORS = {
        SpecializedInputType.REIT: ReitInputExtractor,
        SpecializedInputType.RESOURCE: ResourceInputExtractor,
        SpecializedInputType.BIOTECH: BiotechInputExtractor,
        SpecializedInputType.SOTP: SotpInputExtractor,
    }

    def extract(
        self,
        company_facts: CompanyFacts,
        input_type: SpecializedInputType,
        *,
        ticker: str | None = None,
        historical_periods: int = 5,
    ) -> SpecializedValuationExtraction:
        extractor = self._EXTRACTORS[SpecializedInputType(input_type)]()
        return extractor.extract(
            company_facts,
            ticker=ticker,
            historical_periods=historical_periods,
        )


__all__ = [
    "BiotechInputExtractor",
    "ReitInputExtractor",
    "ResourceInputExtractor",
    "SotpInputExtractor",
    "SpecializedValuationExtractor",
]
