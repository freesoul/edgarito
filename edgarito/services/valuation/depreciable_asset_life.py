from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.valuation.models import BusinessArchetype


@dataclass(frozen=True)
class DepreciableAssetLifeResolution:
    """Auditable result of deterministic depreciable-life resolution."""

    value: int | None
    source: str
    methodology: str
    historical_lives: tuple[Decimal, ...] = ()
    historical_median: Decimal | None = None
    warnings: tuple[str, ...] = ()

    @property
    def audit_message(self) -> str:
        if self.warnings:
            return self.warnings[0]
        if self.value is None:
            return f"Depreciable asset life was not inferred: {self.methodology}"
        return (
            f"Depreciable asset life resolved to {self.value} years from {self.source}; "
            f"{self.methodology}"
        )


class DepreciableAssetLifeResolver:
    """Resolve a CAPEX-shock asset life without external or paid services."""

    MIN_INFERRED_LIFE_YEARS = 4
    MAX_LIFE_YEARS = 30
    AUTOMOTIVE_MANUFACTURING_PRIOR = 7
    GENERAL_OPERATING_PRIOR = 7

    _AUTOMOTIVE_MANUFACTURING = re.compile(
        r"\b(auto(?:mobile|motive)?s?|vehicle manufacturers?|car manufacturers?|"
        r"motor vehicles?|manufactur(?:er|ing))\b"
    )
    _FINANCIAL_INDUSTRY = re.compile(
        r"\b(bank(?:s|ing)?|financial(?:s| services?)?|insurance|insurer|reinsurance|"
        r"asset management|investment management|wealth management|fund manager|"
        r"mortgage finance|brokerage|securities|credit services?)\b"
    )
    _ASSET_INDUSTRY = re.compile(
        r"\b(reit|real estate|property trust|oil|gas|exploration|"
        r"upstream|mining|minerals?|timber|pipeline|biotech(?:nology)?|clinical|"
        r"drug discovery|development stage)\b"
    )
    _NON_APPLICABLE_ARCHETYPES = frozenset(
        {
            BusinessArchetype.FINANCIAL_INTERMEDIARY.value,
            BusinessArchetype.ASSET_MANAGER.value,
            BusinessArchetype.REIT_PROPERTY.value,
            BusinessArchetype.RESOURCE_PRODUCER.value,
            BusinessArchetype.PROJECT_PIPELINE.value,
            BusinessArchetype.HOLDING_COMPANY.value,
        }
    )

    def resolve(
        self,
        financials: NormalizedCompanyFinancials,
        *,
        industry: str | None = None,
        business_archetype: BusinessArchetype | str | None = None,
        sector: str | None = None,
    ) -> DepreciableAssetLifeResolution:
        """Resolve a whole-year life from annual history or a bounded prior.

        Annual CAPEX and D&A are paired by fiscal year and reporting unit.  The
        median of the positive CAPEX / D&A ratios is used rather than an average,
        so one expansion year cannot dominate the estimate.  Specialized
        financial and asset-level archetypes are deliberately left unresolved;
        their CAPEX and D&A do not generally describe one operating asset pool.
        """

        industry_key = self._key(industry)
        archetype_key = self._key(business_archetype)
        sector_key = self._key(sector)
        if self._is_not_applicable(
            industry_key=industry_key,
            archetype_key=archetype_key,
            sector_key=sector_key,
        ):
            return DepreciableAssetLifeResolution(
                value=None,
                source="not applicable",
                methodology=(
                    "financial and asset-level archetypes are excluded because a "
                    "generic operating CAPEX/D&A life would not be meaningful"
                ),
            )

        historical_lives = self._historical_lives(financials)
        historical_median = self._median(historical_lives) if historical_lives else None
        prior = self._fallback_prior(industry_key, archetype_key)
        # A low CAPEX/D&A ratio usually reflects growth CAPEX, acquired assets,
        # or amortization mixed into D&A rather than a complete useful-life
        # schedule. Prefer the industry prior in that case.
        if historical_lives and (
            historical_median is not None and historical_median >= Decimal("5")
        ):
            value = self._bounded_year(historical_median)
            methodology = (
                "robust median of positive annual CAPEX/D&A implied lives "
                f"from {len(historical_lives)} matched fiscal years "
                f"(median {self._format_decimal(historical_median)} years), "
                "rounded to a whole year and bounded to 4-30 years; the conservative "
                "four-year floor avoids treating expansion CAPEX/D&A as a complete "
                "asset-life schedule"
            )
            warning = (
                f"Depreciable asset life inferred automatically as {value} years "
                f"from normalized annual CAPEX/D&A history; {methodology}"
            )
            return DepreciableAssetLifeResolution(
                value=value,
                source="automatic: normalized annual CAPEX/D&A history",
                methodology=methodology,
                historical_lives=historical_lives,
                historical_median=historical_median,
                warnings=(warning,),
            )

        if prior is not None:
            source, value, label = prior
            methodology = (
                f"no usable positive same-unit annual CAPEX/D&A pair was available; "
                f"used the bounded {label} prior"
            )
            warning = (
                f"Depreciable asset life inferred automatically as {value} years "
                f"from {source}; {methodology}"
            )
            return DepreciableAssetLifeResolution(
                value=value,
                source=source,
                methodology=methodology,
                warnings=(warning,),
            )

        return DepreciableAssetLifeResolution(
            value=None,
            source="unavailable",
            methodology=(
                "no usable positive same-unit annual CAPEX/D&A pair was available "
                "and no applicable operating-archetype prior was identified"
            ),
        )

    @classmethod
    def _historical_lives(
        cls,
        financials: NormalizedCompanyFinancials,
    ) -> tuple[Decimal, ...]:
        annual: dict[
            int,
            dict[FinancialConcept, list[FinancialObservation]],
        ] = {}
        for observation in financials.observations:
            if (
                observation.granularity != Granularity.ANNUAL
                or observation.fiscal_period != FiscalPeriod.FY
                or observation.concept
                not in {
                    FinancialConcept.CAPITAL_EXPENDITURES,
                    FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
                }
            ):
                continue
            annual.setdefault(observation.fiscal_year, {}).setdefault(
                observation.concept, []
            ).append(observation)

        lives: list[Decimal] = []
        for fiscal_year in sorted(annual):
            values = annual[fiscal_year]
            capex_observations = cls._preferred_observations(
                values.get(FinancialConcept.CAPITAL_EXPENDITURES, [])
            )
            depreciation_observations = cls._preferred_observations(
                values.get(FinancialConcept.DEPRECIATION_AND_AMORTIZATION, [])
            )
            matched = False
            for capex in capex_observations:
                for depreciation in depreciation_observations:
                    if cls._unit_key(capex.unit) != cls._unit_key(depreciation.unit):
                        continue
                    capex_value = abs(capex.value)
                    depreciation_value = abs(depreciation.value)
                    if (
                        not capex_value.is_finite()
                        or not depreciation_value.is_finite()
                        or capex_value <= 0
                        or depreciation_value <= 0
                    ):
                        continue
                    lives.append(capex_value / depreciation_value)
                    matched = True
                    break
                if matched:
                    break
        return tuple(lives)

    @staticmethod
    def _preferred_observations(
        observations: list[FinancialObservation],
    ) -> list[FinancialObservation]:
        """Make duplicate normalized annual observations deterministic."""

        return sorted(
            observations,
            key=lambda item: (
                item.period_end,
                item.filed or datetime.date.min,
                item.provider.casefold(),
                item.taxonomy.casefold(),
                item.source_concept.casefold(),
                str(item.value),
            ),
            reverse=True,
        )

    @staticmethod
    def _unit_key(value: str) -> str:
        return value.strip().casefold()

    @staticmethod
    def _key(value: object) -> str:
        raw = getattr(value, "value", value)
        if raw is None:
            return ""
        return re.sub(r"[^a-z0-9]+", "_", str(raw).casefold()).strip("_")

    @classmethod
    def _is_not_applicable(
        cls,
        *,
        industry_key: str,
        archetype_key: str,
        sector_key: str,
    ) -> bool:
        if archetype_key in cls._NON_APPLICABLE_ARCHETYPES:
            return True
        if sector_key in {"financials", "real_estate"}:
            return True
        industry_text = industry_key.replace("_", " ")
        return bool(
            cls._FINANCIAL_INDUSTRY.search(industry_text)
            or cls._ASSET_INDUSTRY.search(industry_text)
        )

    @classmethod
    def _fallback_prior(
        cls,
        industry_key: str,
        archetype_key: str,
    ) -> tuple[str, int, str] | None:
        industry_text = industry_key.replace("_", " ")
        if cls._AUTOMOTIVE_MANUFACTURING.search(industry_text):
            return (
                "automatic: automotive/manufacturing industry prior",
                cls.AUTOMOTIVE_MANUFACTURING_PRIOR,
                "automotive/manufacturing",
            )
        if archetype_key in {"", BusinessArchetype.GENERAL_OPERATING.value}:
            return (
                "automatic: general-operating archetype prior",
                cls.GENERAL_OPERATING_PRIOR,
                "general-operating",
            )
        return None

    @staticmethod
    def _median(values: tuple[Decimal, ...]) -> Decimal:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / Decimal(2)

    @classmethod
    def _bounded_year(cls, value: Decimal) -> int:
        rounded = int(value.to_integral_value(rounding=ROUND_HALF_UP))
        return max(
            cls.MIN_INFERRED_LIFE_YEARS,
            min(cls.MAX_LIFE_YEARS, rounded),
        )

    @staticmethod
    def _format_decimal(value: Decimal) -> str:
        return format(value.normalize(), "f")


__all__ = [
    "DepreciableAssetLifeResolution",
    "DepreciableAssetLifeResolver",
]
