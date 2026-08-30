"""Economic identity and provenance audits for FCFF forecasts."""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.forecasting import (
    FcffForecast,
    FcffForecastDriver,
    FcffForecastObservation,
    ForecastAssumptionSource,
    ForecastSeedType,
    ForecastValue,
)
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.forecasting._fcff.contracts import PERCENT
from edgarito.services.metrics.calculator import operating_working_capital_value

_ECONOMIC_AUDIT_FIELDS = (
    "revenue_growth",
    "revenue",
    "operating_margin",
    "operating_income",
    "tax_rate",
    "nopat",
    "depreciation_and_amortization",
    "capital_expenditures",
    "change_in_operating_working_capital",
    "fcff",
)
_AUDIT_TOLERANCE = Decimal("1e-18")


def build_cell_audits(
    service: Any,
    forecast: FcffForecast,
) -> tuple[dict[str, ForecastValue], ...]:
    return tuple(
        build_observation_audits(service, forecast, index, observation)
        for index, observation in enumerate(forecast.observations)
    )


def build_legacy_inconsistent_audits(
    forecast: FcffForecast,
    issues: tuple[str, ...],
) -> tuple[dict[str, ForecastValue], ...]:
    method = "legacy/inconsistent economic identities: " + "; ".join(issues)
    return tuple(
        {
            field: ForecastValue(
                value=getattr(observation, field),
                source="unknown/legacy/inconsistent",
                method=method,
                confidence="low",
            )
            for field in _ECONOMIC_AUDIT_FIELDS
        }
        for observation in forecast.observations
    )


def economic_identity_issues(
    forecast: FcffForecast,
) -> tuple[str, ...]:
    issues: list[str] = []
    previous_working_capital = forecast.base_operating_working_capital
    for observation in forecast.observations:
        expected = {
            "operating_income": (
                observation.revenue * observation.operating_margin / PERCENT
            ),
            "nopat": observation.operating_income
            * (Decimal(1) - observation.tax_rate / PERCENT),
            "depreciation_and_amortization": (
                observation.revenue * observation.depreciation_to_revenue / PERCENT
            ),
            "capital_expenditures": (
                observation.revenue * observation.capex_to_revenue / PERCENT
            ),
            "operating_working_capital": (
                observation.revenue
                * observation.operating_working_capital_to_revenue
                / PERCENT
            ),
            "change_in_operating_working_capital": (
                observation.operating_working_capital - previous_working_capital
            ),
            "fcff": (
                observation.nopat
                + observation.depreciation_and_amortization
                - observation.capital_expenditures
                - observation.change_in_operating_working_capital
            ),
        }
        for field, expected_value in expected.items():
            actual_value = getattr(observation, field)
            if not audit_close(actual_value, expected_value):
                issues.append(
                    f"FY{observation.fiscal_year}E {field}: expected "
                    f"{expected_value}, got {actual_value}"
                )
        previous_working_capital = observation.operating_working_capital
    return tuple(issues)


def audit_close(actual: Decimal, expected: Decimal) -> bool:
    scale = max(abs(actual), abs(expected), Decimal(1))
    return abs(actual - expected) <= scale * _AUDIT_TOLERANCE


def build_observation_audits(
    service: Any,
    forecast: FcffForecast,
    index: int,
    observation: FcffForecastObservation,
) -> dict[str, ForecastValue]:
    growth_source = service._driver_source(
        forecast, FcffForecastDriver.REVENUE_GROWTH, index
    )
    margin_source = service._driver_source(
        forecast, FcffForecastDriver.OPERATING_MARGIN, index
    )
    tax_source = service._driver_source(forecast, FcffForecastDriver.TAX_RATE, index)
    is_ytd_seed = (
        forecast.ytd_anchor is not None
        and observation.forecast_year == 1
        and forecast.seed_type == ForecastSeedType.YTD_PLUS_FORECAST
    )
    is_revenue_anchor = observation.fiscal_year in forecast.parameters.revenue_anchors
    anchor_source = (
        service._revenue_anchor_source(forecast, observation.fiscal_year)
        if is_revenue_anchor
        else None
    )
    anchor_method = (
        f"{anchor_source.replace('_', ' ')} revenue anchor"
        if anchor_source is not None
        else "explicit revenue anchor"
    )
    revenue_components = service._revenue_components(forecast, index)
    if forecast.ytd_anchor is not None:
        revenue_components = (
            *revenue_components,
            ("projected_remainder", "prior_forecast"),
        )
    growth_components = (
        (
            *revenue_components,
            ("projected_remainder", "prior_forecast"),
        )
        if is_ytd_seed
        else (
            (
                *service._revenue_components(forecast, index - 1),
                (
                    "revenue_anchor",
                    service._revenue_anchor_source(forecast, observation.fiscal_year),
                ),
            )
            if is_revenue_anchor
            else (("revenue_growth", growth_source),)
        )
    )
    stage = (
        forecast.adaptive_stages[index]
        if index < len(forecast.adaptive_stages)
        else None
    )
    effective_margin_components = (
        (
            ("ytd_actual_operating_income", "reported"),
            *revenue_components,
            ("projected_remainder", "prior_forecast"),
            ("operating_margin", margin_source),
        )
        if is_ytd_seed
        else (("operating_margin", margin_source),)
    )
    effective_tax_components = (
        (
            ("ytd_actual_nopat", "reported"),
            ("ytd_actual_tax", "reported"),
            *revenue_components,
            ("projected_remainder", "prior_forecast"),
            ("operating_margin", margin_source),
            ("tax_rate", tax_source),
        )
        if is_ytd_seed
        else (("tax_rate", tax_source),)
    )
    depreciation_components = service._driver_components(
        forecast, FcffForecastDriver.DEPRECIATION_TO_REVENUE, index
    )
    capex_components = service._driver_components(
        forecast, FcffForecastDriver.CAPEX_TO_REVENUE, index
    )
    working_capital_components = service._driver_components(
        forecast,
        FcffForecastDriver.OPERATING_WORKING_CAPITAL_TO_REVENUE,
        index,
    )
    capex_constraint = service._capex_constraint_for(
        forecast.parameters, observation.fiscal_year
    )
    capex_constraint_applied = (
        capex_constraint is not None
        and observation.fiscal_year in forecast.capex_constraints_applied
    )
    capex_constraint_components = (
        ((f"capex_constraint_fy{observation.fiscal_year}", capex_constraint.source),)
        if capex_constraint_applied
        else ()
    )
    capex_method = (
        (
            f"{capex_constraint.source} {capex_constraint.methodology} capex "
            "constraint; "
        )
        if capex_constraint_applied
        else ""
    ) + (
        "actual YTD capex + remaining revenue × capex-to-revenue / 100"
        if is_ytd_seed
        else "revenue × capex-to-revenue / 100"
    )
    fcff_method = (
        (
            f"{capex_constraint.source} {capex_constraint.methodology} capex "
            "constraint; "
        )
        if capex_constraint_applied
        else ""
    ) + (
        "NOPAT + depreciation and amortization - capital expenditures - "
        "change in operating working capital"
    )
    prior_working_capital = "historical_seed" if index == 0 else "prior_forecast"
    ytd_projection_components = (
        (("projected_remainder", "prior_forecast"),) if is_ytd_seed else ()
    )
    ytd_depreciation_components = (
        (("ytd_actual_depreciation", "reported"),) if is_ytd_seed else ()
    )
    ytd_capex_components = (
        (("ytd_actual_capex", "reported"),) if is_ytd_seed else ()
    )

    return {
        "revenue_growth": audit_value(
            observation.revenue_growth,
            growth_components,
            (
                service._stage_method(
                    "effective growth from reported YTD revenue and "
                    "projected remainder",
                    stage,
                )
                if is_ytd_seed
                else service._stage_method(
                    f"effective growth from {anchor_method} and prior revenue",
                    stage,
                )
                if is_revenue_anchor
                else service._driver_method(growth_source, "revenue growth", stage)
            ),
            derived=is_revenue_anchor or is_ytd_seed,
        ),
        "revenue": audit_value(
            observation.revenue,
            revenue_components,
            service._stage_method(
                (
                    "actual YTD revenue plus explicit revenue anchor"
                    if is_ytd_seed and is_revenue_anchor
                    else anchor_method
                    if is_revenue_anchor
                    else (
                        "actual YTD revenue + forecast remaining revenue"
                        if is_ytd_seed
                        else (
                            "seed revenue × (1 + revenue growth / 100)"
                            if observation.forecast_year == 1
                            else "prior revenue × (1 + revenue growth / 100)"
                        )
                    )
                ),
                stage,
            ),
            derived=True,
        ),
        "operating_margin": audit_value(
            observation.operating_margin,
            effective_margin_components,
            (
                "blended YTD actual and remaining forecast operating margin"
                if is_ytd_seed
                else service._driver_method(margin_source, "operating margin", stage)
            ),
            derived=is_ytd_seed,
        ),
        "operating_income": audit_value(
            observation.operating_income,
            (*revenue_components, *effective_margin_components),
            service._stage_method(
                (
                    "actual YTD operating income + remaining revenue × operating "
                    "margin / 100"
                    if is_ytd_seed
                    else "revenue × operating margin / 100"
                ),
                stage,
            ),
            derived=True,
        ),
        "tax_rate": audit_value(
            observation.tax_rate,
            effective_tax_components,
            (
                "blended YTD actual and remaining forecast tax rate"
                if is_ytd_seed
                else service._driver_method(tax_source, "tax rate", stage)
            ),
            derived=is_ytd_seed,
        ),
        "nopat": audit_value(
            observation.nopat,
            (*revenue_components, *effective_margin_components, *effective_tax_components),
            service._stage_method(
                (
                    "actual YTD NOPAT + remaining-period NOPAT"
                    if is_ytd_seed
                    else "operating income × (1 - tax rate / 100)"
                ),
                stage,
            ),
            derived=True,
        ),
        "depreciation_and_amortization": audit_value(
            observation.depreciation_and_amortization,
            (
                *revenue_components,
                *ytd_depreciation_components,
                *ytd_projection_components,
                *depreciation_components,
            ),
            service._stage_method(
                (
                    "actual YTD D&A + remaining revenue × depreciation-to-revenue / 100"
                    if is_ytd_seed
                    else "revenue × depreciation-to-revenue / 100"
                ),
                stage,
            ),
            derived=True,
        ),
        "capital_expenditures": audit_value(
            observation.capital_expenditures,
            (
                *revenue_components,
                *ytd_capex_components,
                *ytd_projection_components,
                *capex_components,
                *capex_constraint_components,
            ),
            service._stage_method(capex_method, stage),
            derived=True,
        ),
        "change_in_operating_working_capital": audit_value(
            observation.change_in_operating_working_capital,
            (
                *revenue_components,
                *ytd_projection_components,
                *working_capital_components,
                ("prior_working_capital", prior_working_capital),
            ),
            service._stage_method(
                "operating working capital - prior operating working capital",
                stage,
            ),
            derived=True,
        ),
        "fcff": audit_value(
            observation.fcff,
            (
                *revenue_components,
                *effective_margin_components,
                *effective_tax_components,
                *ytd_projection_components,
                *depreciation_components,
                *capex_components,
                *capex_constraint_components,
                *working_capital_components,
                ("prior_working_capital", prior_working_capital),
            ),
            service._stage_method(fcff_method, stage),
            derived=True,
        ),
    }


def driver_source(
    forecast: FcffForecast,
    driver: FcffForecastDriver,
    index: int,
) -> str:
    path = forecast.assumption_source_paths.get(driver)
    source = path[index] if path is not None and index < len(path) else None
    if source is None:
        source = forecast.assumption_sources.get(driver)
    return source.value if source is not None else "unknown/legacy"


def driver_components(
    forecast: FcffForecast,
    driver: FcffForecastDriver,
    index: int,
) -> tuple[tuple[str, str], ...]:
    components = []
    for source_index in range(index + 1):
        source = driver_source(forecast, driver, source_index)
        label = driver.value
        if source_index < index:
            label += f"_fy{forecast.observations[source_index].fiscal_year}"
        components.append((label, source))
    return tuple(components)


def revenue_components(
    forecast: FcffForecast,
    index: int,
) -> tuple[tuple[str, str], ...]:
    components: list[tuple[str, str]] = [("seed", "historical_seed")]
    if forecast.ytd_anchor is not None:
        components.append(("ytd_actual", "reported"))
    for source_index in range(index + 1):
        fiscal_year = forecast.observations[source_index].fiscal_year
        if fiscal_year in forecast.parameters.revenue_anchors:
            label = (
                "revenue_anchor"
                if source_index == index
                else f"prior_revenue_anchor_fy{fiscal_year}"
            )
            components.append(
                (label, revenue_anchor_source(forecast, fiscal_year))
            )
        else:
            label = (
                "revenue_growth"
                if source_index == index
                else f"prior_revenue_growth_fy{fiscal_year}"
            )
            components.append(
                (
                    label,
                    driver_source(
                        forecast, FcffForecastDriver.REVENUE_GROWTH, source_index
                    ),
                )
            )
    return tuple(components)


def revenue_anchor_source(forecast: FcffForecast, fiscal_year: int) -> str:
    source = forecast.parameters.revenue_anchor_sources.get(
        fiscal_year, ForecastAssumptionSource.EXPLICIT
    )
    return source.value


def driver_method(source: str, driver_label: str, stage: str | None = None) -> str:
    method_by_source = {
        ForecastAssumptionSource.EXPLICIT.value: "explicit forecast driver",
        ForecastAssumptionSource.DRIVER_BASED.value: (
            "independent operating-economics forecast driver"
        ),
        ForecastAssumptionSource.MANAGEMENT_GUIDANCE.value: (
            "management guidance forecast driver"
        ),
        ForecastAssumptionSource.TRAILING_AVERAGE.value: (
            "trailing historical average forecast driver"
        ),
        ForecastAssumptionSource.FORWARD_EVIDENCE.value: (
            "available forward evidence forecast driver"
        ),
        ForecastAssumptionSource.NORMALIZED_HISTORICAL.value: (
            "normalized historical growth forecast driver"
        ),
        ForecastAssumptionSource.CURRENT_RUN_RATE.value: (
            "current run-rate forecast driver"
        ),
        ForecastAssumptionSource.ADAPTIVE_MULTISTAGE.value: (
            "adaptive multistage forecast driver path"
        ),
    }
    stage_label = f" ({stage})" if stage else ""
    return (
        f"{driver_label}{stage_label}: "
        f"{method_by_source.get(source, 'legacy/unknown driver')}"
    )


def stage_method(method: str, stage: str | None) -> str:
    if stage in {"current", "near_term", "transition", "stable"}:
        return f"{stage} stage: {method}"
    return method


def audit_value(
    value: Decimal,
    sources: tuple[tuple[str, str], ...],
    method: str,
    *,
    derived: bool = False,
) -> ForecastValue:
    unique_sources = tuple(dict.fromkeys(sources))
    source = (
        "derived["
        + ",".join(f"{name}={source}" for name, source in unique_sources)
        + "]"
        if derived
        else " + ".join(source for _, source in unique_sources)
    )
    return ForecastValue(
        value=value,
        source=source,
        method=method,
        confidence=confidence(tuple(source for _, source in unique_sources)),
    )


def confidence(sources: tuple[str, ...]) -> str:
    if not sources or any(source == "unknown/legacy" for source in sources):
        return "low"
    if all(source_confidence(source) == "high" for source in sources):
        return "high"
    if all(source_confidence(source) in {"high", "medium"} for source in sources):
        return "medium"
    return "low"


def source_confidence(source: str) -> str:
    if source in {
        ForecastAssumptionSource.EXPLICIT.value,
        ForecastAssumptionSource.MANAGEMENT_GUIDANCE.value,
        ForecastAssumptionSource.DRIVER_BASED.value,
        "historical_seed",
        "reported",
        "prior_forecast",
    }:
        return "high"
    if source in {
        ForecastAssumptionSource.TRAILING_AVERAGE.value,
        ForecastAssumptionSource.FORWARD_EVIDENCE.value,
        ForecastAssumptionSource.NORMALIZED_HISTORICAL.value,
        ForecastAssumptionSource.ADAPTIVE_MULTISTAGE.value,
    }:
        return "medium"
    if source == ForecastAssumptionSource.CURRENT_RUN_RATE.value:
        return "low"
    return "low"


def incomplete_quarter_warnings(
    core_required_concepts,
    financials: NormalizedCompanyFinancials,
    selected_seed_end: datetime.date,
) -> tuple[str, ...]:
    by_period: dict[
        tuple[int, FiscalPeriod], dict[FinancialConcept, FinancialObservation]
    ] = {}
    for item in financials.observations:
        if (
            item.granularity == Granularity.QUARTERLY
            and item.fiscal_period
            in {FiscalPeriod.Q1, FiscalPeriod.Q2, FiscalPeriod.Q3, FiscalPeriod.Q4}
            and item.period_end > selected_seed_end
        ):
            by_period.setdefault(item.period_key, {}).setdefault(item.concept, item)
    candidates = [
        values
        for values in by_period.values()
        if FinancialConcept.REVENUE in values
    ]
    if not candidates:
        return ()
    values = max(
        candidates,
        key=lambda items: items[FinancialConcept.REVENUE].period_end,
    )
    revenue = values[FinancialConcept.REVENUE]
    missing = sorted(
        core_required_concepts - values.keys(), key=lambda item: item.value
    )
    details = [concept.label for concept in missing]
    if operating_working_capital_value(values) is None:
        details.append("Operating Working Capital Components")
    if not details:
        details.append("a coherent single-currency operating dataset")
    return (
        f"FY{revenue.fiscal_year} {revenue.fiscal_period.value} ending "
        f"{revenue.period_end.isoformat()} is incomplete in the "
        f"{financials.provider.upper()} snapshot; forecast seed falls back to "
        f"{selected_seed_end.isoformat()} because "
        f"{', '.join(details)} are unavailable",
    )


__all__ = [name for name in globals() if not name.startswith("__")]
