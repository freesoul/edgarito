"""Evaluation-only reconstruction of realized operating and FCFF outcomes."""

from __future__ import annotations

import datetime
from collections import defaultdict
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.financials.effective_tax import calculate_effective_tax_rate
from edgarito.services.metrics.calculator import (
    OPERATING_WORKING_CAPITAL_CONCEPTS,
    operating_working_capital_value,
)

from .contracts import ActualFinancialObservation, ActualOutcomeData

_METRICS = (
    "revenue",
    "ebit",
    "tax_rate",
    "nopat",
    "depreciation_and_amortization",
    "capex",
    "operating_working_capital",
    "delta_nwc",
    "fcff",
)


class ActualOutcomeReconstructor:
    """Object boundary around the evaluation-only reconstruction helper."""

    def reconstruct(
        self,
        financials: NormalizedCompanyFinancials | Any,
        fiscal_years: Iterable[int] | None = None,
        *,
        target_years: Iterable[int] | None = None,
    ) -> ActualOutcomeData:
        return reconstruct_actual_outcomes(
            financials, fiscal_years, target_years=target_years
        )


def _period_rank(item: FinancialObservation) -> tuple[int, datetime.date, str, str]:
    annual_fy = item.granularity == Granularity.ANNUAL and item.fiscal_period == FiscalPeriod.FY
    return (0 if annual_fy else 1, item.period_end, item.provider, item.source_concept)


def _periods(financials: NormalizedCompanyFinancials) -> dict[int, dict[FinancialConcept, FinancialObservation]]:
    """Select one internally consistent annual period per fiscal year."""

    grouped: defaultdict[int, list[FinancialObservation]] = defaultdict(list)
    for item in financials.observations:
        grouped[item.fiscal_year].append(item)
    selected: dict[int, dict[FinancialConcept, FinancialObservation]] = {}
    for year, records in grouped.items():
        # Annual FY records are preferred.  For a non-calendar issuer the
        # source period_end remains untouched; no December assumption is made.
        ranked = sorted(records, key=_period_rank)
        preferred = ranked[0]
        same_period = [
            item
            for item in ranked
            if item.granularity == preferred.granularity
            and item.fiscal_period == preferred.fiscal_period
            and item.period_end == preferred.period_end
        ]
        selected[year] = {}
        for concept in FinancialConcept:
            candidates = [item for item in same_period if item.concept == concept]
            if candidates:
                selected[year][concept] = sorted(
                    candidates, key=lambda item: (item.provider, item.source_concept)
                )[0]
    return selected


def _reference(values: dict[FinancialConcept, FinancialObservation]) -> FinancialObservation | None:
    return next(
        (values.get(concept) for concept in (FinancialConcept.REVENUE, FinancialConcept.OPERATING_INCOME) if values.get(concept) is not None),
        next(iter(values.values()), None),
    )


def _actual(
    *,
    metric: str,
    value: Decimal | None,
    reference: FinancialObservation | None,
    source_concepts: Iterable[FinancialConcept | str] = (),
    source_date: datetime.date | None = None,
    unit: str | None = None,
    scale: Decimal = Decimal(1),
    currency: str | None = None,
    source_id: str | None = None,
    source_concept: str | None = None,
    reconstruction_provenance: str | None = None,
    sign_normalization: str | None = None,
    value_kind: str = "derived",
) -> ActualFinancialObservation:
    return ActualFinancialObservation(
        fiscal_year=reference.fiscal_year if reference is not None else 0,
        fiscal_period=reference.fiscal_period.value if reference is not None else "FY",
        period_start=reference.period_start if reference is not None else None,
        period_end=reference.period_end if reference is not None else datetime.date(1900, 1, 1),
        metric=metric,
        value=value,
        unit="%" if metric == "tax_rate" else (unit or (reference.unit if reference is not None else "currency")),
        scale=scale,
        currency=currency,
        source="normalized_financials_reconstruction",
        source_id=source_id,
        source_concept=source_concept,
        source_date=source_date,
        filing_date=source_date,
        reconstruction_provenance=reconstruction_provenance,
        sign_normalization=sign_normalization,
        value_kind=value_kind,
        provenance="; ".join(
            str(getattr(item, "value", item)) for item in source_concepts
        ) or "normalized financial observation",
    )


def _scale(item: FinancialObservation) -> Decimal:
    value = getattr(item, "scale", Decimal(1))
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _currency(item: FinancialObservation) -> str | None:
    value = getattr(item, "currency", None)
    if value:
        return str(value).strip().upper()
    first = item.unit.strip().split(maxsplit=1)[0]
    return first.upper() if len(first) == 3 and first.isalpha() else None


def _unit_signature(item: FinancialObservation) -> tuple[str, str | None, Decimal]:
    return item.unit.strip().casefold(), _currency(item), _scale(item)


def _source_metadata(
    values: dict[FinancialConcept, FinancialObservation],
    concepts: Iterable[FinancialConcept],
) -> tuple[str | None, str | None, datetime.date | None, str | None, Decimal]:
    selected = tuple(values[concept] for concept in concepts if concept in values)
    source_ids = tuple(
        item.accession_number for item in selected if item.accession_number
    )
    source_concepts = tuple(item.source_concept for item in selected)
    filed = tuple(item.filed for item in selected if item.filed is not None)
    signatures = tuple(_unit_signature(item) for item in selected)
    return (
        ",".join(dict.fromkeys(source_ids)) or None,
        ",".join(dict.fromkeys(source_concepts)) or None,
        max(filed) if filed else None,
        signatures[0][1] if signatures else None,
        signatures[0][2] if signatures else Decimal(1),
    )


def _require_compatible_amounts(
    values: dict[FinancialConcept, FinancialObservation],
) -> tuple[str, str | None, Decimal] | None:
    concepts = (
        FinancialConcept.OPERATING_INCOME,
        FinancialConcept.PRETAX_INCOME,
        FinancialConcept.INCOME_TAX_EXPENSE,
        FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
        FinancialConcept.CAPITAL_EXPENDITURES,
        *tuple(sorted(OPERATING_WORKING_CAPITAL_CONCEPTS, key=lambda item: item.value)),
    )
    selected = tuple(values[concept] for concept in concepts if concept in values)
    signatures = tuple(_unit_signature(item) for item in selected)
    if not signatures:
        return None
    if len(set(signatures)) != 1:
        raise ValueError(
            "Cannot reconstruct actual FCFF metrics from incompatible monetary "
            "units, currencies, or scales"
        )
    return selected[0].unit, signatures[0][1], signatures[0][2]


def _positive_addback(
    item: FinancialObservation | None,
    *,
    metric: str,
) -> tuple[Decimal | None, str]:
    if item is None:
        return None, "unavailable_missing_source"
    if item.value >= 0:
        return item.value, "reported_nonnegative"
    source = item.source_concept.casefold()
    derivation = (item.derivation or "").casefold()
    explicit_negative_capex = metric == "capex" and (
        "paymentstoacquire" in source
        or "negative cash flow" in derivation
        or "cash outflow" in derivation
    )
    explicit_negative_da = metric == "depreciation_and_amortization" and (
        "negative cash flow" in derivation or "cash outflow" in derivation
    )
    if explicit_negative_capex or explicit_negative_da:
        return abs(item.value), "normalized_from_explicit_negative_cash_flow_convention"
    return None, "unavailable_unknown_sign_convention"


def _compatible_period_cadence(
    previous: FinancialObservation,
    current: FinancialObservation,
) -> bool:
    if current.fiscal_year != previous.fiscal_year + 1:
        return False
    if current.fiscal_period != previous.fiscal_period:
        return False
    if current.granularity != previous.granularity:
        return False
    days = (current.period_end - previous.period_end).days
    # This accepts a 52/53-week non-calendar year and normal month/day
    # anniversaries, while rejecting a 15-month Sep-to-Dec pairing.
    if not 355 <= days <= 375:
        return False
    if current.period_start is not None and previous.period_start is not None:
        previous_duration = (previous.period_end - previous.period_start).days
        current_duration = (current.period_end - current.period_start).days
        if previous_duration <= 0 or current_duration <= 0:
            return False
        if abs(current_duration - previous_duration) > 31:
            return False
    return True


def reconstruct_actual_outcomes(
    financials: NormalizedCompanyFinancials | Any,
    fiscal_years: Iterable[int] | None = None,
    *,
    target_years: Iterable[int] | None = None,
) -> ActualOutcomeData:
    """Reconstruct realized metrics using the production OWC/tax conventions.

    This helper does not apply a cutoff and never computes a market price or an
    intrinsic value.  It is intended for a separately supplied outcome data
    set, after the forecast path has run.
    """

    financials = (
        financials
        if isinstance(financials, NormalizedCompanyFinancials)
        else NormalizedCompanyFinancials.model_validate(financials)
    )
    if fiscal_years is not None and target_years is not None:
        raise ValueError("Specify only one of fiscal_years and target_years")
    fiscal_years = target_years if target_years is not None else fiscal_years
    selected = _periods(financials)
    years = tuple(sorted(set(fiscal_years if fiscal_years is not None else selected)))
    outcomes: list[ActualFinancialObservation] = []
    owc_by_year: dict[int, Decimal | None] = {
        year: (
            operating_working_capital_value(values).value
            if operating_working_capital_value(values) is not None
            else None
        )
        for year, values in selected.items()
    }
    period_by_year: dict[int, dict[FinancialConcept, FinancialObservation]] = {}

    for year in years:
        values = selected.get(year, {})
        period_by_year[year] = values
        reference = _reference(values)
        revenue = values.get(FinancialConcept.REVENUE)
        ebit = values.get(FinancialConcept.OPERATING_INCOME)
        pretax = values.get(FinancialConcept.PRETAX_INCOME)
        tax = values.get(FinancialConcept.INCOME_TAX_EXPENSE)
        da = values.get(FinancialConcept.DEPRECIATION_AND_AMORTIZATION)
        capex = values.get(FinancialConcept.CAPITAL_EXPENDITURES)
        common_amount_unit = _require_compatible_amounts(values)
        tax_rate = (
            calculate_effective_tax_rate(pretax.value, tax.value)
            if pretax is not None and tax is not None
            else None
        )
        nopat = (
            ebit.value * (Decimal(1) - tax_rate / Decimal(100))
            if ebit is not None and tax_rate is not None
            else None
        )
        owc_observation = operating_working_capital_value(values)
        owc = owc_observation.value if owc_observation is not None else None
        owc_by_year[year] = owc
        da_value, da_sign = _positive_addback(
            da, metric="depreciation_and_amortization"
        )
        capex_value, capex_sign = _positive_addback(capex, metric="capex")
        metric_values = {
            "revenue": revenue.value if revenue is not None else None,
            "ebit": ebit.value if ebit is not None else None,
            "tax_rate": tax_rate,
            "nopat": nopat,
            # D&A and CAPEX are positive economic uses/add-backs even when a
            # provider stores cash-flow CAPEX as a negative number.
            "depreciation_and_amortization": da_value,
            "capex": capex_value,
            "operating_working_capital": owc,
        }
        for metric in _METRICS[:7]:
            source = {
                "revenue": (FinancialConcept.REVENUE,),
                "ebit": (FinancialConcept.OPERATING_INCOME,),
                "tax_rate": (FinancialConcept.PRETAX_INCOME, FinancialConcept.INCOME_TAX_EXPENSE),
                "nopat": (FinancialConcept.OPERATING_INCOME, FinancialConcept.PRETAX_INCOME, FinancialConcept.INCOME_TAX_EXPENSE),
                "depreciation_and_amortization": (FinancialConcept.DEPRECIATION_AND_AMORTIZATION,),
                "capex": (FinancialConcept.CAPITAL_EXPENDITURES,),
                "operating_working_capital": tuple(values),
            }[metric]
            source_dates = tuple(
                item.filed
                for item in (values.get(concept) for concept in source if isinstance(concept, FinancialConcept))
                if item is not None and item.filed is not None
            )
            metadata_concepts = tuple(
                concept for concept in source if isinstance(concept, FinancialConcept)
            )
            source_id, source_concept, source_date, currency, source_scale = (
                _source_metadata(values, metadata_concepts)
                if metadata_concepts
                else (None, None, None, None, Decimal(1))
            )
            direct = values.get(metadata_concepts[0]) if len(metadata_concepts) == 1 else None
            metric_unit = (
                "%"
                if metric == "tax_rate"
                else direct.unit
                if direct is not None
                else common_amount_unit[0]
                if common_amount_unit is not None
                else reference.unit
                if reference is not None
                else "currency"
            )
            metric_scale = (
                Decimal(1)
                if metric == "tax_rate"
                else _scale(direct)
                if direct is not None
                else common_amount_unit[2]
                if common_amount_unit is not None
                else source_scale
            )
            sign_normalization = (
                da_sign
                if metric == "depreciation_and_amortization"
                else capex_sign
                if metric == "capex"
                else None
            )
            item = _actual(
                metric=metric,
                value=metric_values[metric],
                reference=reference,
                source_concepts=source,
                source_date=max(source_dates) if source_dates else None,
                unit=metric_unit,
                scale=metric_scale,
                currency=None if metric == "tax_rate" else currency,
                source_id=source_id,
                source_concept=source_concept,
                reconstruction_provenance=(
                    "effective_tax_rate=100*income_tax_expense/pretax_income"
                    if metric == "tax_rate"
                    else "NOPAT=EBIT*(1-effective_tax_rate/100)"
                    if metric == "nopat"
                    else "OWC=existing strict operating_working_capital_value"
                    if metric == "operating_working_capital"
                    else "reported direct financial observation"
                ),
                sign_normalization=sign_normalization,
                value_kind=(
                    "derived"
                    if metric
                    in {
                        "tax_rate",
                        "nopat",
                        "operating_working_capital",
                    }
                    else "reported"
                ),
            )
            # The helper's period identity must always be the selected source
            # period, including a 52/53-week or other non-calendar year.
            outcomes.append(item.model_copy(update={"fiscal_year": year}))

    for year in years:
        values = period_by_year[year]
        reference = _reference(values)
        previous_year = year - 1
        previous = owc_by_year.get(previous_year)
        current = owc_by_year.get(year)
        current_reference = _reference(values)
        previous_reference = _reference(selected.get(previous_year, {}))
        current_amount_unit = _require_compatible_amounts(values)
        previous_amount_unit = _require_compatible_amounts(
            selected.get(previous_year, {})
        )
        compatible_previous = (
            previous_reference is not None
            and current_reference is not None
            and previous is not None
            and current is not None
            and _compatible_period_cadence(previous_reference, current_reference)
            and previous_amount_unit is not None
            and current_amount_unit is not None
            and previous_amount_unit == current_amount_unit
        )
        delta = current - previous if compatible_previous else None
        nopat = next((item.value for item in outcomes if item.fiscal_year == year and item.metric == "nopat"), None)
        da = next((item.value for item in outcomes if item.fiscal_year == year and item.metric == "depreciation_and_amortization"), None)
        capex = next((item.value for item in outcomes if item.fiscal_year == year and item.metric == "capex"), None)
        fcff = nopat + da - capex - delta if None not in (nopat, da, capex, delta) else None
        source_dates = tuple(
            item.filed for item in values.values() if item.filed is not None
        )
        source_date = max(source_dates) if source_dates else None
        derived_unit = (
            current_amount_unit[0]
            if current_amount_unit is not None
            else reference.unit
            if reference is not None
            else "currency"
        )
        derived_scale = current_amount_unit[2] if current_amount_unit is not None else Decimal(1)
        derived_currency = current_amount_unit[1] if current_amount_unit is not None else None
        derived_source_id, derived_source_concept, _, _, _ = _source_metadata(
            values, tuple(values)
        )
        outcomes.append(
            _actual(
                metric="delta_nwc",
                value=delta,
                reference=reference,
                source_concepts=("operating_working_capital",),
                source_date=source_date,
                unit=derived_unit,
                scale=derived_scale,
                currency=derived_currency,
                source_id=derived_source_id,
                source_concept=derived_source_concept,
                reconstruction_provenance="delta_nwc=current OWC-prior compatible OWC",
            ).model_copy(update={"fiscal_year": year})
        )
        outcomes.append(
            _actual(
                metric="fcff",
                value=fcff,
                reference=reference,
                source_concepts=(
                    "nopat",
                    "depreciation_and_amortization",
                    "capex",
                    "delta_nwc",
                ),
                source_date=source_date,
                unit=derived_unit,
                scale=derived_scale,
                currency=derived_currency,
                source_id=derived_source_id,
                source_concept=derived_source_concept,
                reconstruction_provenance=(
                    "FCFF=NOPAT+D&A-CAPEX-delta_nwc"
                ),
            ).model_copy(update={"fiscal_year": year})
        )

    period_ends = tuple(sorted({item.period_end for item in outcomes if item.period_end.year > 1900}))
    return ActualOutcomeData(
        company=financials.company_name,
        ticker=financials.ticker,
        observations=tuple(outcomes),
        outcome_dates=period_ends,
    )


reconstruct_actuals = reconstruct_actual_outcomes
actuals_from_normalized_financials = reconstruct_actual_outcomes


__all__ = [
    "ActualOutcomeReconstructor",
    "reconstruct_actual_outcomes",
    "reconstruct_actuals",
    "actuals_from_normalized_financials",
]
