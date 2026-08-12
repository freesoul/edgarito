"""Reconcile independent operating revenue with external fallback evidence.

The operating forecast engine deliberately stops at a provider-neutral company
revenue path.  This module is the narrow seam between that path and the
existing FCFF forecast inputs.  It does not retrieve evidence, inspect a
ticker, or call an LLM: callers provide already-normalized values and the
reconciler selects one value for each fiscal year.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from edgarito.schemas.forward import ForwardRevenueEstimate
from edgarito.schemas.operating import CompanyOperatingForecast
from edgarito.services.forecasting.models import (
    FcffForecastParameters,
    ForecastAssumptionSource,
)

_YEAR_MIN = 1900
_YEAR_MAX = 2200
_PERCENT = Decimal(100)

_EXPLICIT_SOURCE = "explicit"
_MANAGEMENT_SOURCE = ForecastAssumptionSource.MANAGEMENT_GUIDANCE.value
_INDEPENDENT_SOURCE = "independent_operating"
_CONSENSUS_SOURCE = "analyst_consensus"
_HISTORICAL_SOURCE = ForecastAssumptionSource.NORMALIZED_HISTORICAL.value
_UNAVAILABLE_SOURCE = "unavailable"

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_SUPPORTED_INDEPENDENT_SOURCES = {
    _EXPLICIT_SOURCE,
    _MANAGEMENT_SOURCE,
    _INDEPENDENT_SOURCE,
    "independent",
    "independent_forecast",
    "mixed",
    "derived",
}
_UNSUPPORTED_INDEPENDENT_SOURCES = {
    "",
    _UNAVAILABLE_SOURCE,
    _HISTORICAL_SOURCE,
    "historical",
    "reported",
    _CONSENSUS_SOURCE,
    "consensus",
    ForecastAssumptionSource.FORWARD_EVIDENCE.value,
}


class ResolvedRevenueYear(BaseModel):
    """The selected revenue evidence for one fiscal year.

    ``independent_revenue`` and ``consensus_revenue`` are retained even when
    they lose precedence.  That makes the selection auditable without making
    the FCFF/DCF layer aware of operating-forecast details.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fiscal_year: int = Field(ge=_YEAR_MIN, le=_YEAR_MAX)
    revenue: Decimal
    source: str
    confidence: str
    independent_revenue: Decimal | None = None
    independent_source: str | None = None
    independent_confidence: str | None = None
    consensus_revenue: Decimal | None = None
    consensus_source: str | None = None
    consensus_confidence: str | None = None
    historical_revenue: Decimal | None = None
    explicit_revenue: Decimal | None = None
    management_revenue: Decimal | None = None
    # Percentage variance of consensus against the independent value.
    variance: Decimal | None = None

    @field_validator(
        "revenue",
        "independent_revenue",
        "consensus_revenue",
        "historical_revenue",
        "explicit_revenue",
        "management_revenue",
    )
    @classmethod
    def validate_revenue_values(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and (not value.is_finite() or value < 0):
            raise ValueError("Resolved revenue values must be finite and non-negative")
        return value

    @field_validator("variance")
    @classmethod
    def validate_variance(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and not value.is_finite():
            raise ValueError("Resolved revenue variance must be finite")
        return value

    @field_validator("source", "independent_source", "consensus_source")
    @classmethod
    def normalize_sources(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(getattr(value, "value", value)).strip()
        return normalized or None

    @field_validator(
        "confidence",
        "independent_confidence",
        "consensus_confidence",
        mode="before",
    )
    @classmethod
    def normalize_confidences(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(getattr(value, "value", value)).strip().casefold()
        if normalized not in _CONFIDENCE_RANK:
            raise ValueError("Revenue confidence must be high, medium, or low")
        return normalized

    @property
    def selected_revenue(self) -> Decimal:
        """Descriptive alias for the selected absolute revenue."""

        return self.revenue


@dataclass(frozen=True)
class RevenueForecastReconciliation:
    """Detailed result retained alongside the public company forecast."""

    forecast: CompanyOperatingForecast
    resolved_years: tuple[ResolvedRevenueYear, ...]

    @property
    def company_forecast(self) -> CompanyOperatingForecast:
        """Compatibility-friendly name for the selected company forecast."""

        return self.forecast

    @property
    def selected_years(self) -> tuple[ResolvedRevenueYear, ...]:
        """Alias emphasizing that each item is the selected value."""

        return self.resolved_years

    @property
    def selected_revenue_by_year(self) -> dict[int, Decimal]:
        return {item.fiscal_year: item.revenue for item in self.resolved_years}

    @property
    def revenue(self) -> tuple[Decimal, ...]:
        return self.forecast.consolidated_revenue

    @property
    def consolidated_revenue(self) -> tuple[Decimal, ...]:
        return self.forecast.consolidated_revenue

    @property
    def consolidated_growth(self) -> tuple[Decimal | None, ...]:
        return self.forecast.consolidated_growth

    @property
    def fiscal_years(self) -> tuple[int, ...]:
        return self.forecast.fiscal_years

    @property
    def source_by_year(self) -> dict[int, str]:
        return self.forecast.source_by_year

    @property
    def confidence_by_year(self) -> dict[int, str]:
        return self.forecast.confidence_by_year

    @property
    def explicit_years(self) -> tuple[int, ...]:
        return self.forecast.explicit_years

    @property
    def transition_start_year(self) -> int | None:
        return self.forecast.transition_start_year

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.forecast.warnings


class RevenueForecastReconciler:
    """Select one absolute revenue value per fiscal year.

    Selection precedence is intentionally per-year:

    ``explicit anchor > management anchor > supported independent > consensus
    > normalized historical``.

    The independent forecast's fiscal-year path is authoritative.  Consensus
    values can fill unavailable years in that path, but do not extend it or
    replace a sufficiently supported independent value.
    """

    def __init__(
        self,
        *,
        minimum_independent_confidence: str = "medium",
        minimum_confidence: str | None = None,
    ) -> None:
        selected_confidence = minimum_confidence or minimum_independent_confidence
        normalized = str(selected_confidence).strip().casefold()
        if normalized not in _CONFIDENCE_RANK:
            raise ValueError(
                "Minimum independent confidence must be high, medium, or low"
            )
        self.minimum_independent_confidence = normalized
        self._last_resolved_years: tuple[ResolvedRevenueYear, ...] = ()

    @staticmethod
    def _independent_candidate(
        forecast: CompanyOperatingForecast,
        index: int,
        year: int,
    ) -> tuple[Decimal | None, str | None, str | None]:
        """Return the independent candidate at a forecast path position."""

        return _independent_candidate(forecast, index, year)

    @property
    def last_resolved_years(self) -> tuple[ResolvedRevenueYear, ...]:
        """Return details from the most recent reconciliation call."""

        return self._last_resolved_years

    def reconcile(
        self,
        independent_forecast: CompanyOperatingForecast,
        consensus: Iterable[ForwardRevenueEstimate] | Mapping[Any, Any] | Any = (),
        historical_revenue: Mapping[Any, Any] | None = None,
        *,
        explicit_anchors: Mapping[Any, Any] | Iterable[Any] | Any | None = None,
        management_anchors: Mapping[Any, Any] | Iterable[Any] | Any | None = None,
        normalized_historical_revenue: Mapping[Any, Any] | None = None,
        consensus_estimates: Iterable[ForwardRevenueEstimate]
        | Mapping[Any, Any]
        | Any
        | None = None,
        return_details: bool = False,
    ) -> CompanyOperatingForecast | RevenueForecastReconciliation:
        """Reconcile an independent company forecast without provider logic.

        ``historical_revenue`` is a normalized ``{fiscal_year: value}`` mapping.
        The longer ``normalized_historical_revenue`` name is accepted as an
        explicit alias for callers that want to make the fallback contract
        obvious.  ``consensus_estimates`` is likewise an alias for
        ``consensus`` and is useful when a resolver result object is passed.
        """

        if normalized_historical_revenue is not None:
            if historical_revenue is not None:
                raise ValueError(
                    "Pass either historical_revenue or "
                    "normalized_historical_revenue, not both"
                )
            historical_revenue = normalized_historical_revenue
        if consensus_estimates is not None:
            if consensus not in ((), None):
                raise ValueError(
                    "Pass either consensus or consensus_estimates, not both"
                )
            consensus = consensus_estimates

        forecast = _coerce_company_forecast(independent_forecast)
        historical = _normalize_year_values(historical_revenue, "historical revenue")
        explicit = _normalize_year_values(explicit_anchors, "explicit revenue anchors")
        management = _normalize_year_values(
            management_anchors, "management revenue anchors"
        )
        estimates = _normalize_consensus(consensus)
        estimates_by_year = _select_consensus_by_year(estimates)

        resolved: list[ResolvedRevenueYear] = []
        warnings = list(forecast.warnings)
        for index, year in enumerate(forecast.fiscal_years):
            independent_value, independent_source, independent_confidence = (
                self._independent_candidate(forecast, index, year)
            )
            consensus_estimate = estimates_by_year.get(year)
            selected = self.resolve_year(
                fiscal_year=year,
                independent=independent_value,
                independent_source=independent_source,
                independent_confidence=independent_confidence,
                consensus=consensus_estimate,
                historical=historical.get(year),
                explicit=explicit.get(year),
                management=management.get(year),
            )
            resolved.append(selected)
            if selected.source == _UNAVAILABLE_SOURCE:
                warnings.append(
                    f"FY{year}: no usable independent, consensus, or normalized "
                    "historical revenue"
                )

        resolved_years = tuple(resolved)
        revenue = tuple(item.revenue for item in resolved_years)
        growth = _growth_path(revenue)
        source_by_year = {item.fiscal_year: item.source for item in resolved_years}
        confidence_by_year = {
            item.fiscal_year: item.confidence for item in resolved_years
        }
        explicit_years = tuple(
            item.fiscal_year
            for item in resolved_years
            if _is_explicit_selection_source(item.source)
        )
        transition_start_year = explicit_years[-1] + 1 if explicit_years else None
        reconciled_forecast = forecast.model_copy(
            update={
                "consolidated_revenue": revenue,
                "consolidated_growth": growth,
                "explicit_years": explicit_years,
                "transition_start_year": transition_start_year,
                "source_by_year": source_by_year,
                "confidence_by_year": confidence_by_year,
                "warnings": tuple(dict.fromkeys(warnings)),
            }
        )
        self._last_resolved_years = resolved_years
        details = RevenueForecastReconciliation(reconciled_forecast, resolved_years)
        return details if return_details else reconciled_forecast

    def reconcile_with_details(
        self,
        independent_forecast: CompanyOperatingForecast,
        consensus: Iterable[ForwardRevenueEstimate] | Mapping[Any, Any] | Any = (),
        historical_revenue: Mapping[Any, Any] | None = None,
        *,
        explicit_anchors: Mapping[Any, Any] | Iterable[Any] | Any | None = None,
        management_anchors: Mapping[Any, Any] | Iterable[Any] | Any | None = None,
        normalized_historical_revenue: Mapping[Any, Any] | None = None,
        consensus_estimates: Iterable[ForwardRevenueEstimate]
        | Mapping[Any, Any]
        | Any
        | None = None,
    ) -> RevenueForecastReconciliation:
        """Return the selected forecast together with per-year audit details."""

        return self.reconcile(
            independent_forecast,
            consensus,
            historical_revenue,
            explicit_anchors=explicit_anchors,
            management_anchors=management_anchors,
            normalized_historical_revenue=normalized_historical_revenue,
            consensus_estimates=consensus_estimates,
            return_details=True,
        )

    def resolve_year(
        self,
        fiscal_year: int | None = None,
        independent: Any = None,
        consensus: Any = None,
        historical: Any = None,
        management_constraints: Any = None,
        *,
        explicit: Any = None,
        management: Any = None,
        explicit_anchor: Any = None,
        management_anchor: Any = None,
        independent_source: str | None = None,
        independent_confidence: str | None = None,
    ) -> ResolvedRevenueYear:
        """Resolve one year for callers that do not need a whole path.

        The positional form mirrors the small resolver contract used by the
        architecture design: independent value, consensus estimate, and
        historical fallback.  Anchor aliases are accepted to keep this helper
        convenient for deterministic callers.
        """

        independent_year, independent_value = _candidate_year_and_value(independent)
        consensus_year, consensus_item = _candidate_consensus(
            consensus, default_year=fiscal_year
        )
        explicit_year, explicit_value = _candidate_year_and_value(explicit)
        management_year, management_value = _candidate_year_and_value(management)
        if explicit_anchor is not None:
            explicit_year, explicit_value = _candidate_year_and_value(explicit_anchor)
        if management_anchor is not None:
            management_year, management_value = _candidate_year_and_value(
                management_anchor
            )

        resolved_year = _coerce_year(
            fiscal_year
            if fiscal_year is not None
            else independent_year
            if independent_year is not None
            else consensus_year
            if consensus_year is not None
            else explicit_year
            if explicit_year is not None
            else management_year
        )
        if resolved_year is None:
            raise ValueError("A fiscal year is required to reconcile one revenue year")

        if management_constraints is not None and management_value is None:
            constraint_year, constraint_value = _candidate_year_and_value(
                management_constraints
            )
            if constraint_year is None or constraint_year == resolved_year:
                management_value = constraint_value

        if independent_source is None and independent is not None:
            independent_source = getattr(independent, "source", None)
        if independent_confidence is None and independent is not None:
            independent_confidence = getattr(independent, "confidence", None)
        independent_source = _normalize_source(independent_source)
        independent_confidence = _normalize_confidence(
            independent_confidence,
            default=(
                "high"
                if independent_source in {_EXPLICIT_SOURCE, _MANAGEMENT_SOURCE}
                else "medium"
            ),
        )
        if independent_value is not None and independent_source is None:
            independent_source = _INDEPENDENT_SOURCE
        if independent_value is None:
            independent_confidence = None

        consensus_revenue = (
            consensus_item.midpoint if consensus_item is not None else None
        )
        consensus_source = (
            _normalize_source(consensus_item.source)
            if consensus_item is not None
            else None
        )
        consensus_confidence = (
            _estimate_confidence(consensus_item) if consensus_item is not None else None
        )
        if consensus_revenue is not None and consensus_revenue <= 0:
            consensus_revenue = None

        historical_value = (
            _to_decimal(historical, f"historical revenue FY{resolved_year}")
            if historical is not None
            else None
        )
        if historical_value is not None and historical_value < 0:
            raise ValueError("Historical revenue cannot be negative")

        explicit_value = _optional_nonnegative(
            explicit_value, f"explicit revenue anchor FY{resolved_year}"
        )
        management_value = _optional_nonnegative(
            management_value, f"management revenue anchor FY{resolved_year}"
        )
        independent_value = _optional_nonnegative(
            independent_value, f"independent revenue FY{resolved_year}"
        )

        supported_independent = (
            independent_value is not None
            and independent_source is not None
            and _is_sufficiently_supported(
                independent_source,
                independent_confidence or "low",
                minimum_confidence=self.minimum_independent_confidence,
            )
        )
        if explicit_value is not None:
            selected_value = explicit_value
            source = _EXPLICIT_SOURCE
            confidence = "high"
        elif management_value is not None:
            selected_value = management_value
            source = _MANAGEMENT_SOURCE
            confidence = "high"
        elif supported_independent:
            selected_value = independent_value
            source = independent_source or _INDEPENDENT_SOURCE
            confidence = independent_confidence or "medium"
        elif consensus_revenue is not None:
            selected_value = consensus_revenue
            source = _CONSENSUS_SOURCE
            confidence = consensus_confidence or "medium"
        elif historical_value is not None:
            selected_value = historical_value
            source = _HISTORICAL_SOURCE
            confidence = "medium"
        elif independent_value is not None and independent_source in {
            _HISTORICAL_SOURCE,
            "historical",
        }:
            # A forecast may carry its normalized-history fallback in the
            # consolidated path when the separate history mapping is sparse.
            selected_value = independent_value
            source = _HISTORICAL_SOURCE
            confidence = "medium"
            historical_value = independent_value
        else:
            selected_value = Decimal(0)
            source = _UNAVAILABLE_SOURCE
            confidence = "low"

        variance = None
        if (
            independent_value is not None
            and consensus_revenue is not None
            and independent_value != 0
        ):
            variance = (consensus_revenue / independent_value - Decimal(1)) * _PERCENT

        return ResolvedRevenueYear(
            fiscal_year=resolved_year,
            revenue=selected_value,
            source=source,
            confidence=confidence,
            independent_revenue=independent_value,
            independent_source=independent_source,
            independent_confidence=independent_confidence,
            consensus_revenue=consensus_revenue,
            consensus_source=consensus_source,
            consensus_confidence=consensus_confidence,
            historical_revenue=historical_value,
            explicit_revenue=explicit_value,
            management_revenue=management_value,
            variance=variance,
        )

    def resolve_years(
        self,
        independent_forecast: CompanyOperatingForecast,
        consensus: Iterable[ForwardRevenueEstimate] | Mapping[Any, Any] | Any = (),
        historical_revenue: Mapping[Any, Any] | None = None,
        *,
        explicit_anchors: Mapping[Any, Any] | Iterable[Any] | Any | None = None,
        management_anchors: Mapping[Any, Any] | Iterable[Any] | Any | None = None,
        normalized_historical_revenue: Mapping[Any, Any] | None = None,
        consensus_estimates: Iterable[ForwardRevenueEstimate]
        | Mapping[Any, Any]
        | Any
        | None = None,
    ) -> tuple[ResolvedRevenueYear, ...]:
        """Return only per-year selections while using the normal path logic."""

        return self.reconcile_with_details(
            independent_forecast,
            consensus,
            historical_revenue,
            explicit_anchors=explicit_anchors,
            management_anchors=management_anchors,
            normalized_historical_revenue=normalized_historical_revenue,
            consensus_estimates=consensus_estimates,
        ).resolved_years

    # Builder/resolver terminology used by adjacent services.
    build = reconcile
    resolve = reconcile
    reconcile_forecast = reconcile

    def reconcile_result(
        self,
        independent_forecast: CompanyOperatingForecast,
        consensus: Iterable[ForwardRevenueEstimate] | Mapping[Any, Any] | Any = (),
        historical_revenue: Mapping[Any, Any] | None = None,
        *,
        explicit_anchors: Mapping[Any, Any] | Iterable[Any] | Any | None = None,
        management_anchors: Mapping[Any, Any] | Iterable[Any] | Any | None = None,
        normalized_historical_revenue: Mapping[Any, Any] | None = None,
        consensus_estimates: Iterable[ForwardRevenueEstimate]
        | Mapping[Any, Any]
        | Any
        | None = None,
    ) -> RevenueForecastReconciliation:
        """Descriptive alias for :meth:`reconcile_with_details`."""

        return self.reconcile_with_details(
            independent_forecast,
            consensus,
            historical_revenue,
            explicit_anchors=explicit_anchors,
            management_anchors=management_anchors,
            normalized_historical_revenue=normalized_historical_revenue,
            consensus_estimates=consensus_estimates,
        )

    @staticmethod
    def materialize_revenue_anchors(
        parameters: FcffForecastParameters,
        selected_revenue: CompanyOperatingForecast
        | RevenueForecastReconciliation
        | Mapping[Any, Any]
        | Iterable[ResolvedRevenueYear],
    ) -> FcffForecastParameters:
        """Materialize selected absolute revenue into FCFF anchor fields."""

        return materialize_revenue_anchors(parameters, selected_revenue)

    materialize_selected_revenue = materialize_revenue_anchors


def materialize_revenue_anchors(
    parameters: FcffForecastParameters,
    selected_revenue: CompanyOperatingForecast
    | RevenueForecastReconciliation
    | Mapping[Any, Any]
    | Iterable[ResolvedRevenueYear],
) -> FcffForecastParameters:
    """Copy selected revenue into ``FcffForecastParameters`` without FCFF changes.

    Existing explicit or management anchors are retained.  Consensus and
    independent operating sources use the existing FCFF ``FORWARD_EVIDENCE``
    provenance value because the legacy FCFF enum predates the independent
    operating source; the detailed source remains available on the
    reconciliation result.
    """

    if parameters.revenue_growth is not None:
        # An explicit percentage path is the existing higher-priority FCFF
        # input and must not be silently converted to absolute anchors.
        return parameters

    records = _selection_records(selected_revenue)
    if not records:
        return parameters

    anchors = dict(parameters.revenue_anchors)
    sources = dict(parameters.revenue_anchor_sources)
    for year, value, source in records:
        normalized_source = _normalize_source(source) or _INDEPENDENT_SOURCE
        if normalized_source == _UNAVAILABLE_SOURCE:
            continue
        normalized_value = _to_decimal(value, f"selected revenue FY{year}")
        if normalized_value <= 0:
            raise ValueError(
                f"Selected revenue FY{year} must be positive for FCFF anchors"
            )
        incoming_source = _fcff_source(normalized_source)
        existing_source = sources.get(year, ForecastAssumptionSource.EXPLICIT)
        if year in anchors and _fcff_source_rank(existing_source) >= _fcff_source_rank(
            incoming_source
        ):
            continue
        anchors[year] = normalized_value
        sources[year] = incoming_source

    if (
        anchors == parameters.revenue_anchors
        and sources == parameters.revenue_anchor_sources
    ):
        return parameters
    return parameters.model_copy(
        update={
            "revenue_anchors": anchors,
            "revenue_anchor_sources": sources,
        }
    )


def materialize_selected_revenue(
    parameters: FcffForecastParameters,
    selected_revenue: CompanyOperatingForecast
    | RevenueForecastReconciliation
    | Mapping[Any, Any]
    | Iterable[ResolvedRevenueYear],
) -> FcffForecastParameters:
    """Alias for :func:`materialize_revenue_anchors`."""

    return materialize_revenue_anchors(parameters, selected_revenue)


def _coerce_company_forecast(
    value: CompanyOperatingForecast,
) -> CompanyOperatingForecast:
    if isinstance(value, CompanyOperatingForecast):
        return value
    return CompanyOperatingForecast.model_validate(value)


def _independent_candidate(
    forecast: CompanyOperatingForecast,
    index: int,
    year: int,
) -> tuple[Decimal | None, str | None, str | None]:
    value = forecast.consolidated_revenue[index]
    raw_source = forecast.source_by_year.get(year)
    source = _normalize_source(raw_source)
    if source is None:
        source = _INDEPENDENT_SOURCE if value != 0 else _UNAVAILABLE_SOURCE
    default_confidence = (
        "high"
        if source in {_EXPLICIT_SOURCE, _MANAGEMENT_SOURCE}
        else "low"
        if source == _UNAVAILABLE_SOURCE
        else "medium"
    )
    confidence = _normalize_confidence(
        forecast.confidence_by_year.get(year), default=default_confidence
    )
    return value, source, confidence


def _normalize_year_values(value: Any, label: str) -> dict[int, Decimal]:
    if value is None:
        return {}
    if isinstance(value, FcffForecastParameters):
        value = value.revenue_anchors
    if isinstance(value, Mapping):
        # Accept one anchor record passed as a mapping as well as a year map.
        if _mapping_has_anchor_year(value):
            year = _coerce_year(value.get("fiscal_year", value.get("year")))
            amount = _anchor_value(value, label)
            return {year: amount} if year is not None and amount is not None else {}
        result: dict[int, Decimal] = {}
        for raw_year, raw_value in value.items():
            year = _coerce_year(raw_year)
            if year is None:
                raise ValueError(f"{label} contains an invalid fiscal year: {raw_year}")
            amount = _anchor_value(raw_value, f"{label} FY{year}")
            if amount is not None:
                result[year] = amount
        return result

    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a year mapping or anchor iterable")
    result = {}
    try:
        items = iter(value)
    except TypeError as error:
        raise ValueError(
            f"{label} must be a year mapping or anchor iterable"
        ) from error
    for item in items:
        if isinstance(item, Mapping):
            raw_year = item.get("fiscal_year", item.get("year"))
            raw_value = item
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            raw_year, raw_value = item
        else:
            raw_year = getattr(item, "fiscal_year", getattr(item, "year", None))
            raw_value = item
        year = _coerce_year(raw_year)
        if year is None:
            raise ValueError(f"{label} contains an anchor without a fiscal year")
        amount = _anchor_value(raw_value, f"{label} FY{year}")
        if amount is not None:
            result[year] = amount
    return result


def _normalize_consensus(value: Any) -> tuple[ForwardRevenueEstimate, ...]:
    if value is None:
        return ()
    if hasattr(value, "estimates") and not isinstance(value, ForwardRevenueEstimate):
        value = value.estimates
    if isinstance(value, ForwardRevenueEstimate):
        return (value,)
    if isinstance(value, Mapping):
        if _mapping_has_anchor_year(value):
            return (_coerce_estimate(value),)
        values = []
        for raw_year, raw_value in value.items():
            year = _coerce_year(raw_year)
            if year is None:
                raise ValueError(
                    f"Consensus contains an invalid fiscal year: {raw_year}"
                )
            if isinstance(raw_value, ForwardRevenueEstimate):
                values.append(raw_value)
            elif isinstance(raw_value, Mapping):
                values.append(_coerce_estimate({**raw_value, "fiscal_year": year}))
            else:
                amount = _anchor_value(raw_value, f"consensus FY{year}")
                if amount is not None:
                    values.append(
                        ForwardRevenueEstimate.from_value(
                            year, amount, source=_CONSENSUS_SOURCE
                        )
                    )
        return tuple(values)
    if isinstance(value, (str, bytes)):
        raise ValueError("Consensus must contain ForwardRevenueEstimate values")
    values = []
    for item in value:
        values.append(_coerce_estimate(item))
    return tuple(values)


def _coerce_estimate(value: Any) -> ForwardRevenueEstimate:
    if isinstance(value, ForwardRevenueEstimate):
        return value
    return ForwardRevenueEstimate.model_validate(value)


def _select_consensus_by_year(
    estimates: Sequence[ForwardRevenueEstimate],
) -> dict[int, ForwardRevenueEstimate]:
    selected: dict[int, ForwardRevenueEstimate] = {}
    for estimate in estimates:
        if estimate.midpoint is None or estimate.midpoint <= 0:
            continue
        current = selected.get(estimate.fiscal_year)
        if current is None or _estimate_rank(estimate) > _estimate_rank(current):
            selected[estimate.fiscal_year] = estimate
    return selected


def _estimate_rank(estimate: ForwardRevenueEstimate) -> tuple[int, int, Decimal]:
    return (
        _CONFIDENCE_RANK.get(_estimate_confidence(estimate), 1),
        estimate.analyst_count or 0,
        estimate.midpoint or Decimal(0),
    )


def _estimate_confidence(estimate: ForwardRevenueEstimate) -> str:
    if estimate.confidence is not None:
        return estimate.confidence
    analysts = estimate.analyst_count
    if analysts is not None and analysts >= 10:
        return "high"
    if analysts is not None and analysts < 3:
        return "low"
    return "medium"


def _selection_records(
    selected_revenue: CompanyOperatingForecast
    | RevenueForecastReconciliation
    | Mapping[Any, Any]
    | Iterable[ResolvedRevenueYear],
) -> tuple[tuple[int, Decimal, str], ...]:
    if isinstance(selected_revenue, RevenueForecastReconciliation):
        return tuple(
            (item.fiscal_year, item.revenue, item.source)
            for item in selected_revenue.resolved_years
        )
    if isinstance(selected_revenue, CompanyOperatingForecast):
        return tuple(
            (
                year,
                revenue,
                selected_revenue.source_by_year.get(
                    year,
                    _EXPLICIT_SOURCE
                    if year in selected_revenue.explicit_years
                    else _INDEPENDENT_SOURCE,
                ),
            )
            for year, revenue in zip(
                selected_revenue.fiscal_years,
                selected_revenue.consolidated_revenue,
                strict=True,
            )
        )
    if isinstance(selected_revenue, Mapping):
        records = []
        for raw_year, raw_value in selected_revenue.items():
            year = _coerce_year(raw_year)
            if year is None:
                raise ValueError(
                    f"Selected revenue contains an invalid fiscal year: {raw_year}"
                )
            if isinstance(raw_value, ResolvedRevenueYear):
                records.append((year, raw_value.revenue, raw_value.source))
                continue
            source = _INDEPENDENT_SOURCE
            value = raw_value
            if isinstance(raw_value, (tuple, list)) and len(raw_value) == 2:
                value, source = raw_value
            records.append(
                (year, _anchor_value(value, f"selected revenue FY{year}"), source)
            )
        return tuple(
            (year, value, source)
            for year, value, source in records
            if value is not None
        )
    if isinstance(selected_revenue, (str, bytes)):
        raise ValueError("Selected revenue must be a forecast or year mapping")
    records = []
    for item in selected_revenue:
        if not isinstance(item, ResolvedRevenueYear):
            item = ResolvedRevenueYear.model_validate(item)
        records.append((item.fiscal_year, item.revenue, item.source))
    return tuple(records)


def _candidate_consensus(
    value: Any,
    *,
    default_year: int | None = None,
) -> tuple[int | None, ForwardRevenueEstimate | None]:
    if value is None:
        return None, None
    if isinstance(value, (int, str, Decimal, float)) and not isinstance(value, bool):
        return default_year, ForwardRevenueEstimate.from_value(
            default_year or 2000,
            _to_decimal(value, "consensus revenue"),
            source=_CONSENSUS_SOURCE,
        )
    if isinstance(value, Mapping) and not _mapping_has_anchor_year(value):
        if len(value) == 1:
            raw_year, raw_value = next(iter(value.items()))
            year = _coerce_year(raw_year)
            if year is None:
                return None, None
            if isinstance(raw_value, ForwardRevenueEstimate):
                return year, raw_value
            amount = _anchor_value(raw_value, f"consensus FY{year}")
            if amount is None:
                return year, None
            return year, ForwardRevenueEstimate.from_value(
                year, amount, source=_CONSENSUS_SOURCE
            )
        return None, None
    estimate = _coerce_estimate(value)
    return estimate.fiscal_year, estimate


def _candidate_year_and_value(value: Any) -> tuple[int | None, Decimal | None]:
    if value is None:
        return None, None
    if isinstance(value, Mapping):
        raw_year = value.get("fiscal_year", value.get("year"))
        if raw_year is None and len(value) == 1:
            raw_year, raw_value = next(iter(value.items()))
            return _coerce_year(raw_year), _anchor_value(raw_value, "revenue candidate")
        return _coerce_year(raw_year), _anchor_value(value, "revenue candidate")
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return _coerce_year(value[0]), _anchor_value(value[1], "revenue candidate")
    raw_year = getattr(value, "fiscal_year", getattr(value, "year", None))
    if raw_year is not None:
        return _coerce_year(raw_year), _anchor_value(value, "revenue candidate")
    return None, _anchor_value(value, "revenue candidate")


def _anchor_value(value: Any, label: str) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, ForwardRevenueEstimate):
        value = value.midpoint
    elif isinstance(value, Mapping):
        if "value" in value:
            value = value["value"]
        elif "revenue" in value:
            value = value["revenue"]
        elif "average" in value:
            value = value["average"]
        elif "point" in value:
            value = value["point"]
        elif (
            value.get("low") is not None
            or value.get("high") is not None
            or value.get("minimum") is not None
            or value.get("maximum") is not None
        ):
            low = value.get("low", value.get("minimum"))
            high = value.get("high", value.get("maximum"))
            if low is not None and high is not None:
                value = (_to_decimal(low, label), _to_decimal(high, label))
            else:
                value = low if low is not None else high
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        low = _to_decimal(value[0], label)
        high = _to_decimal(value[1], label)
        value = (low + high) / Decimal(2)
    elif hasattr(value, "midpoint"):
        value = value.midpoint
    elif hasattr(value, "value"):
        value = value.value
    elif hasattr(value, "revenue"):
        value = value.revenue
    elif hasattr(value, "point"):
        value = value.point
    if value is None:
        return None
    return _to_decimal(value, label)


def _optional_nonnegative(value: Decimal | None, label: str) -> Decimal | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError(f"{label} cannot be negative")
    return value


def _to_decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


def _coerce_year(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    text = str(value).strip().upper()
    if text.startswith("FY"):
        text = text[2:]
    try:
        year = int(text)
    except (TypeError, ValueError):
        return None
    if year < _YEAR_MIN or year > _YEAR_MAX:
        return None
    return year


def _mapping_has_anchor_year(value: Mapping[Any, Any]) -> bool:
    return "fiscal_year" in value or "year" in value


def _normalize_source(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(getattr(value, "value", value)).strip().casefold()
    return normalized or None


def _normalize_confidence(value: Any, *, default: str) -> str:
    normalized = (
        str(getattr(value, "value", value)).strip().casefold() if value else default
    )
    if normalized not in _CONFIDENCE_RANK:
        return default
    return normalized


def _is_sufficiently_supported(
    source: str,
    confidence: str,
    *,
    minimum_confidence: str,
) -> bool:
    normalized_source = _normalize_source(source) or ""
    if normalized_source in _UNSUPPORTED_INDEPENDENT_SOURCES:
        return False
    if normalized_source == _MANAGEMENT_SOURCE:
        return True
    if normalized_source not in _SUPPORTED_INDEPENDENT_SOURCES and (
        "independent" not in normalized_source
    ):
        return False
    return _CONFIDENCE_RANK.get(confidence, 0) >= _CONFIDENCE_RANK[minimum_confidence]


def _is_explicit_selection_source(source: str) -> bool:
    normalized = _normalize_source(source) or ""
    return normalized not in {_HISTORICAL_SOURCE, _UNAVAILABLE_SOURCE, "historical"}


def _growth_path(revenue: tuple[Decimal, ...]) -> tuple[Decimal | None, ...]:
    if not revenue:
        return ()
    growth: list[Decimal | None] = [None]
    for previous, current in zip(revenue[:-1], revenue[1:], strict=True):
        if previous == 0:
            growth.append(Decimal(0) if current == 0 else None)
        else:
            growth.append((current / previous - Decimal(1)) * _PERCENT)
    return tuple(growth)


def _fcff_source(value: Any) -> ForecastAssumptionSource:
    normalized = _normalize_source(value) or _INDEPENDENT_SOURCE
    if normalized == _EXPLICIT_SOURCE:
        return ForecastAssumptionSource.EXPLICIT
    if normalized == _MANAGEMENT_SOURCE:
        return ForecastAssumptionSource.MANAGEMENT_GUIDANCE
    if normalized == _HISTORICAL_SOURCE:
        return ForecastAssumptionSource.NORMALIZED_HISTORICAL
    if normalized == ForecastAssumptionSource.CURRENT_RUN_RATE.value:
        return ForecastAssumptionSource.CURRENT_RUN_RATE
    return ForecastAssumptionSource.FORWARD_EVIDENCE


def _fcff_source_rank(value: Any) -> int:
    normalized = _normalize_source(value) or ""
    return {
        _EXPLICIT_SOURCE: 4,
        _MANAGEMENT_SOURCE: 3,
        ForecastAssumptionSource.FORWARD_EVIDENCE.value: 2,
        ForecastAssumptionSource.NORMALIZED_HISTORICAL.value: 1,
        ForecastAssumptionSource.CURRENT_RUN_RATE.value: 0,
    }.get(normalized, 0)


# Names used by callers that describe the same seam in operating rather than
# revenue terminology.  They are aliases, not separate implementations.
OperatingRevenueReconciler = RevenueForecastReconciler
OperatingForecastReconciler = RevenueForecastReconciler
ResolvedOperatingRevenue = ResolvedRevenueYear
OperatingRevenueReconciliation = RevenueForecastReconciliation
materialize_operating_revenue = materialize_revenue_anchors


__all__ = [
    "OperatingForecastReconciler",
    "OperatingRevenueReconciler",
    "OperatingRevenueReconciliation",
    "ResolvedOperatingRevenue",
    "ResolvedRevenueYear",
    "RevenueForecastReconciliation",
    "RevenueForecastReconciler",
    "materialize_operating_revenue",
    "materialize_revenue_anchors",
    "materialize_selected_revenue",
]
