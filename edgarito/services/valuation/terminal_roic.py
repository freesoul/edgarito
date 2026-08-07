from __future__ import annotations

import datetime
from dataclasses import dataclass
from decimal import Decimal
from statistics import median

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.valuation.assumptions import (
    AssumptionOrigin,
    AssumptionProvenance,
    AssumptionUnit,
    ValuationAssumption,
    ValuationAssumptionKind,
)
from edgarito.services.valuation.models import CompanyLifecycle, Cyclicality


@dataclass(frozen=True)
class TerminalRoicResolution:
    value: Decimal
    source: str
    methodology: str
    confidence: str
    normalized_roic: Decimal | None
    historical_roics: tuple[Decimal, ...]
    persistence: Decimal
    warnings: tuple[str, ...]
    assumption: ValuationAssumption


class TerminalRoicResolver:
    """Resolve sustainable ROIC independently from forecast arithmetic."""

    _MAX_TERMINAL_ROIC = Decimal("60")
    _MIN_GROWTH_SPREAD = Decimal("0.5")

    def resolve(
        self,
        financials: NormalizedCompanyFinancials,
        *,
        wacc: Decimal,
        terminal_growth: Decimal,
        valuation_date: datetime.date,
        currency: str,
        explicit_roic: Decimal | None = None,
        explicit_source: str = "explicit valuation profile",
        lifecycle: CompanyLifecycle | None = None,
        cyclicality: Cyclicality | None = None,
        peer_roics: tuple[Decimal, ...] = (),
    ) -> TerminalRoicResolution:
        minimum = terminal_growth + self._MIN_GROWTH_SPREAD
        if explicit_roic is not None:
            if explicit_roic <= terminal_growth:
                raise ValueError("Explicit terminal ROIC must exceed terminal growth")
            methodology = (
                "Explicit terminal ROIC override; automatic inference bypassed"
            )
            assumption = self._assumption(
                financials,
                explicit_roic,
                valuation_date,
                currency,
                AssumptionOrigin.EXPLICIT,
                explicit_source,
                methodology,
                rationale="Explicit profile/CLI assumptions take precedence",
            )
            return TerminalRoicResolution(
                value=explicit_roic,
                source=explicit_source,
                methodology=methodology,
                confidence="high",
                normalized_roic=None,
                historical_roics=(),
                persistence=Decimal(1),
                warnings=(),
                assumption=assumption,
            )

        historical = self._historical_roics(financials, currency)
        values = tuple(value for _year, value, _date in historical[-5:])
        peer_values = tuple(
            value
            for value in peer_roics
            if value.is_finite() and Decimal("-100") < value <= Decimal("150")
        )
        warnings: list[str] = []
        normalized: Decimal | None = Decimal(median(values)) if values else None
        observed_on = historical[-1][2] if historical else None

        if normalized is None and peer_values:
            anchor = Decimal(median(peer_values))
            persistence = Decimal("0.40")
            confidence = "low"
            source = "automatic: peer ROIC evidence"
            warnings.append(
                "Company ROIC history was unavailable; terminal ROIC relies on peer evidence"
            )
        elif normalized is None:
            spread = {
                CompanyLifecycle.GROWTH: Decimal("4"),
                CompanyLifecycle.MATURE: Decimal("2"),
                CompanyLifecycle.DECLINING: Decimal("1"),
            }.get(lifecycle, Decimal("2"))
            if cyclicality == Cyclicality.HIGH:
                spread = Decimal("0.5")
            anchor = wacc + spread
            persistence = Decimal("0.35")
            confidence = "low"
            source = "automatic: WACC and company maturity fallback"
            warnings.append(
                "Historical and peer ROIC evidence was unavailable; a low-confidence "
                "maturity-adjusted WACC spread was used"
            )
        else:
            anchor = normalized
            if peer_values:
                anchor = normalized * Decimal("0.75") + Decimal(
                    median(peer_values)
                ) * Decimal("0.25")
            persistence = self._persistence(
                values, normalized, wacc, lifecycle, cyclicality
            )
            confidence = self._confidence(values, normalized)
            source = "automatic: normalized historical ROIC"
            if peer_values:
                source += " blended with peer ROIC"
            if len(values) < 3:
                warnings.append(
                    "Fewer than three usable annual ROIC observations lower confidence"
                )
            latest = values[-1]
            if (
                len(values) >= 3
                and latest - normalized >= Decimal("5")
                and latest > normalized * Decimal("1.5")
            ):
                confidence = "medium"
                warnings.append(
                    "Latest ROIC is materially above its historical median; the peak "
                    "was not capitalized into perpetuity"
                )

        raw = wacc + persistence * (anchor - wacc)
        value = min(max(raw, minimum), self._MAX_TERMINAL_ROIC)
        if value != raw:
            if value == minimum:
                warnings.append(
                    "Terminal ROIC was floored at terminal growth plus 0.5 percentage points"
                )
            else:
                warnings.append("Terminal ROIC was capped at 60%")
        methodology = (
            "WACC + persistence × (normalized ROIC anchor - WACC); normalized ROIC "
            "is the median of up to five annual NOPAT / average invested-capital "
            "observations. Persistence reflects history length, dispersion, excess-"
            "return duration, lifecycle and cyclicality"
        )
        assumption = self._assumption(
            financials,
            value,
            valuation_date,
            currency,
            AssumptionOrigin.DERIVED,
            "edgarito",
            methodology,
            observed_on=observed_on,
            rationale=(
                f"anchor={anchor:.3f}%, WACC={wacc:.3f}%, "
                f"persistence={persistence:.3f}, confidence={confidence}"
            ),
        )
        return TerminalRoicResolution(
            value=value,
            source=source,
            methodology=methodology,
            confidence=confidence,
            normalized_roic=normalized,
            historical_roics=values,
            persistence=persistence,
            warnings=tuple(warnings),
            assumption=assumption,
        )

    @staticmethod
    def _persistence(values, normalized, wacc, lifecycle, cyclicality) -> Decimal:
        persistence = {
            1: Decimal("0.35"),
            2: Decimal("0.45"),
            3: Decimal("0.55"),
        }.get(len(values), Decimal("0.65"))
        deviations = [abs(value - normalized) for value in values]
        dispersion = (
            Decimal(median(deviations)) / max(abs(normalized), Decimal("1"))
            if deviations
            else Decimal(1)
        )
        if dispersion <= Decimal("0.15"):
            persistence += Decimal("0.15")
        elif dispersion <= Decimal("0.35"):
            persistence += Decimal("0.05")
        elif dispersion > Decimal("0.60"):
            persistence -= Decimal("0.10")
        excess_fraction = Decimal(sum(value > wacc for value in values)) / Decimal(
            len(values)
        )
        if len(values) >= 3 and excess_fraction == 1:
            persistence += Decimal("0.10")
        elif excess_fraction < Decimal("0.5"):
            persistence -= Decimal("0.10")
        if lifecycle == CompanyLifecycle.GROWTH:
            persistence += Decimal("0.05")
        elif lifecycle in {CompanyLifecycle.DECLINING, CompanyLifecycle.DISTRESSED}:
            persistence -= Decimal("0.10")
        if cyclicality == Cyclicality.HIGH:
            persistence -= Decimal("0.20")
        elif cyclicality == Cyclicality.MODERATE:
            persistence -= Decimal("0.05")
        if (
            len(values) >= 3
            and values[-1] - normalized >= Decimal("5")
            and values[-1] > normalized * Decimal("1.5")
        ):
            persistence -= Decimal("0.15")
        return min(max(persistence, Decimal("0.15")), Decimal("0.85"))

    @staticmethod
    def _confidence(values, normalized) -> str:
        if len(values) < 2:
            return "low"
        dispersion = Decimal(
            median([abs(value - normalized) for value in values])
        ) / max(abs(normalized), Decimal("1"))
        if len(values) >= 4 and dispersion <= Decimal("0.25"):
            return "high"
        return "medium"

    @classmethod
    def _historical_roics(cls, financials, currency):
        annual: dict[int, dict[FinancialConcept, FinancialObservation]] = {}
        for item in financials.observations:
            if (
                item.granularity == Granularity.ANNUAL
                and item.fiscal_period == FiscalPeriod.FY
                and item.unit in {currency, "shares"}
            ):
                annual.setdefault(item.fiscal_year, {}).setdefault(item.concept, item)
        capital = {
            year: cls._invested_capital(values, currency)
            for year, values in annual.items()
        }
        result = []
        for year in sorted(annual):
            values = annual[year]
            operating_income = values.get(FinancialConcept.OPERATING_INCOME)
            pretax = values.get(FinancialConcept.PRETAX_INCOME)
            tax = values.get(FinancialConcept.INCOME_TAX_EXPENSE)
            current_capital = capital.get(year)
            if operating_income is None or current_capital is None:
                continue
            tax_rate = Decimal("25")
            if pretax is not None and tax is not None and pretax.value > 0:
                candidate = abs(tax.value) / pretax.value
                if Decimal(0) <= candidate <= Decimal(1):
                    tax_rate = candidate * Decimal(100)
            prior_capital = capital.get(year - 1)
            denominator = (
                (prior_capital + current_capital) / Decimal(2)
                if prior_capital is not None
                else current_capital
            )
            if denominator <= 0:
                continue
            roic = (
                operating_income.value
                * (Decimal(1) - tax_rate / Decimal(100))
                / denominator
                * Decimal(100)
            )
            if Decimal("-100") < roic <= Decimal("150"):
                result.append((year, roic, operating_income.period_end))
        return result

    @staticmethod
    def _invested_capital(values, currency):
        equity = values.get(FinancialConcept.STOCKHOLDERS_EQUITY)
        cash = values.get(FinancialConcept.CASH_AND_EQUIVALENTS)
        if (
            equity is None
            or cash is None
            or equity.unit != currency
            or cash.unit != currency
        ):
            return None
        current = values.get(FinancialConcept.SHORT_TERM_DEBT) or values.get(
            FinancialConcept.LONG_TERM_DEBT_CURRENT
        )
        noncurrent = values.get(FinancialConcept.LONG_TERM_DEBT_NONCURRENT)
        debt = sum(
            (item.value for item in (current, noncurrent) if item is not None),
            Decimal(0),
        )
        return equity.value + debt - cash.value

    @staticmethod
    def _assumption(
        financials,
        value,
        valuation_date,
        currency,
        origin,
        provider,
        methodology,
        *,
        observed_on=None,
        rationale=None,
    ):
        return ValuationAssumption(
            kind=ValuationAssumptionKind.TERMINAL_ROIC,
            value=value,
            unit=AssumptionUnit.PERCENTAGE_POINTS,
            selected_on=valuation_date,
            currency=currency,
            company_id=financials.company_id,
            rationale=rationale,
            provenance=AssumptionProvenance(
                origin=origin,
                provider=provider,
                observed_on=observed_on,
                methodology=methodology,
            ),
        )


__all__ = ["TerminalRoicResolution", "TerminalRoicResolver"]
