"""Deterministic segment gross-margin and gross-profit forecasting.

This module is deliberately separate from the revenue archetype engine.  Gross
economics consume an already selected revenue path and can therefore be added
without changing any revenue formula, reconciliation, or FCFF arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from edgarito.schemas.forecasting import (
    ForecastDecision,
    ForecastOverride,
    ForecastPlan,
    ForecastStrategy,
    ForecastValueBasis,
)
from edgarito.schemas.operating import (
    CompanyOperatingEconomicsForecast,
    CompanyOperatingForecast,
    EvidenceReference,
    OperatingDriverObservation,
    OperatingEconomicsDiagnostics,
    OperatingEconomicsForecastConfig,
    OperatingEconomicsMetricDiagnostics,
    OperatingEconomicsYear,
    OperatingSegment,
    SegmentOperatingEconomicsDiagnostics,
    SegmentOperatingEconomicsForecast,
    SegmentRevenueForecast,
    canonical_operating_segment_id,
    operating_periods_compatible,
    operating_units_compatible,
)
from edgarito.services.operating._forecast.consolidation import (
    _combine_sources,
    _select_consolidation_segments,
    _worst_confidence,
)
from edgarito.services.operating._forecast.contracts import _CONFIDENCE_RANK
from edgarito.services.operating._forecast.normalization import (
    _management_guidance_to_observation,
)
from edgarito.services.operating._forecast.opex_ebit import OperatingOpexEbitEngine
from edgarito.services.operating._forecast.tax_nopat import OperatingTaxNopatEngine

_MARGIN = "gross_margin"
_PROFIT = "gross_profit"
_REVENUE = "revenue"
_COST = "cost_of_revenue"
_DIRECT_ORIGINS = frozenset(
    {"reported", "first_party_observation", "extracted_evidence"}
)
_MARGIN_ALIASES = frozenset(
    {
        "gross_margin",
        "gross_margin_percent",
        "gross_margin_percentage",
        "gross_margin_rate",
        "gross_margin_pct",
        "gross_profit_margin",
    }
)
_PROFIT_ALIASES = frozenset(
    {"gross_profit", "gross_profit_amount", "gross_income", "gross_income_amount"}
)
_COST_ALIASES = frozenset(
    {
        "cost_of_revenue",
        "cost_of_sales",
        "cost_of_goods_sold",
        "cogs",
        "cost_of_goods",
    }
)
_REVENUE_ALIASES = frozenset(
    {
        "revenue",
        "segment_revenue",
        "sales",
        "net_sales",
        "total_revenue",
        "total_sales",
    }
)
_RATE_UNITS = frozenset(
    {"%", "ratio", "rate", "percent", "percentage", "percentage_points", "bps", "bp"}
)
_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class _Candidate:
    metric: str
    value: Decimal
    source: str
    confidence: str
    provenance: Any = None
    source_provenance: tuple[EvidenceReference, ...] = ()
    observations: tuple[OperatingDriverObservation, ...] = ()
    method: str = "observed"
    explicit: bool = False
    consumes_revenue: bool = False


@dataclass(frozen=True)
class _RevenueInput:
    value: Decimal | None
    source: str
    confidence: str
    unit: str = "currency"
    provenance: Any = None
    source_provenance: tuple[EvidenceReference, ...] = ()


@dataclass(frozen=True)
class _ExplicitPath:
    values: tuple[Decimal, ...]
    provenance: Any = None
    source_provenance: tuple[EvidenceReference, ...] = ()


class OperatingEconomicsForecastService:
    """Forecast gross economics, then apply the focused OPEX/EBIT stage.

    The exact metric precedence is independent for each segment and year.
    Margin selection is ``explicit margin > management margin > direct
    first-party margin > same-scope/period/unit historical gross profit over
    reported revenue > normalized recent historical margin > final explicit
    GP > final management GP/cost derivation > unavailable``.  Profit selection
    is ``explicit GP > management GP > management cost-derived GP > direct GP
    > same-scope/period/unit historical cost-derived GP > revenue times the
    selected margin``.  A GP-derived margin is never allowed to displace a
    margin-specific source.  Direct historical gross profit remains selected
    truth unless a higher-precedence manual or management path is present;
    identity differences are audit warnings only.
    """

    def __init__(self, config: OperatingEconomicsForecastConfig | Mapping[str, Any] | None = None) -> None:
        self.config = (
            config
            if isinstance(config, OperatingEconomicsForecastConfig)
            else OperatingEconomicsForecastConfig.model_validate(config or {})
        )
        self.opex_ebit_engine = OperatingOpexEbitEngine()
        self.tax_nopat_engine = OperatingTaxNopatEngine()

    def forecast(
        self,
        segments: Iterable[OperatingSegment] = (),
        segment_revenue_forecasts: Iterable[SegmentRevenueForecast]
        | SegmentRevenueForecast
        | CompanyOperatingForecast
        | None = None,
        observations: Iterable[OperatingDriverObservation] = (),
        fiscal_years: Iterable[int] = (),
        *,
        segment_revenue_forecast: SegmentRevenueForecast | None = None,
        existing_segment_revenue_forecast: SegmentRevenueForecast | None = None,
        revenue_forecast: CompanyOperatingForecast | None = None,
        existing_revenue_forecast: CompanyOperatingForecast | None = None,
        management_constraints: Iterable[OperatingDriverObservation] = (),
        plan: ForecastPlan | Mapping[str, Any] | None = None,
        forecast_plan: ForecastPlan | Mapping[str, Any] | None = None,
        overrides: Iterable[ForecastOverride] | Mapping[Any, Any] = (),
        forecast_overrides: Iterable[ForecastOverride] | Mapping[Any, Any] | None = None,
        company_id: str = "company",
        fiscal_period: str = "FY",
        period_key: str | None = None,
        config: OperatingEconomicsForecastConfig | Mapping[str, Any] | None = None,
        ambiguous_segment_ids: Iterable[str] = (),
    ) -> CompanyOperatingEconomicsForecast:
        """Build immutable segment and company gross-economics contracts."""

        policy = self.config if config is None else self._config(config)
        company_revenue = revenue_forecast or existing_revenue_forecast
        if company_revenue is None and isinstance(
            segment_revenue_forecasts, CompanyOperatingForecast
        ):
            company_revenue = segment_revenue_forecasts
        if company_revenue is None and isinstance(
            segment_revenue_forecast, CompanyOperatingForecast
        ):
            company_revenue = segment_revenue_forecast
        if company_revenue is not None and company_id == "company":
            company_id = company_revenue.company_id
        revenue_forecasts = self._revenue_forecasts(
            segment_revenue_forecasts
            if segment_revenue_forecasts is not None
            else segment_revenue_forecast
            if segment_revenue_forecast is not None
            else existing_segment_revenue_forecast,
            company_revenue,
        )
        supplied_segments = tuple(segments)
        supplied_ids = [
            (
                item.segment_id
                if isinstance(item, OperatingSegment)
                else OperatingSegment.model_validate(item).segment_id
            )
            for item in supplied_segments
        ]
        ambiguous_ids = {
            segment_id
            for segment_id in supplied_ids
            if supplied_ids.count(segment_id) > 1
        }
        revenue_segment_ids = [
            forecast.segment.segment_id for forecast in revenue_forecasts
        ]
        ambiguous_ids.update(
            segment_id
            for segment_id in revenue_segment_ids
            if revenue_segment_ids.count(segment_id) > 1
        )
        ambiguous_ids.update(
            canonical_operating_segment_id(segment_id) or str(segment_id)
            for segment_id in ambiguous_segment_ids
        )
        normalized_segments = self._segments(supplied_segments, revenue_forecasts)
        years = self._years(fiscal_years, revenue_forecasts, company_revenue)
        if company_revenue is not None and company_revenue.fiscal_years != years:
            raise ValueError("Revenue and operating-economics years must match")
        normalized_observations = tuple(
            self._coerce_observation(item)
            for item in self._observation_items(observations)
        )
        normalized_management = tuple(
            observation
            for item in self._observation_items(management_constraints)
            for observation in (self._coerce_observation(item, management=True),)
            if observation is not None
        )
        all_observations = (*normalized_observations, *normalized_management)
        normalized_plan = self._coerce_plan(plan or forecast_plan)
        if not normalized_segments and company_revenue is not None:
            # Normalized financial facts are company-only. A synthetic
            # consolidated scope lets the gross and OPEX stages consume those
            # facts without pretending they describe an allocated segment.
            company_segment = OperatingSegment(
                segment_id="company",
                name="Company",
                scope="consolidated",
                currency=company_revenue.unit if company_revenue.unit != "currency" else None,
                source="normalized_historical",
                confidence="medium",
            )
            growth: list[Decimal | None] = [None]
            for previous, current in zip(
                company_revenue.consolidated_revenue[:-1],
                company_revenue.consolidated_revenue[1:],
                strict=True,
            ):
                growth.append(
                    (current / previous - Decimal(1)) * Decimal(100)
                    if previous
                    else None
                )
            synthetic_revenue = SegmentRevenueForecast(
                segment=company_segment,
                fiscal_years=years,
                revenue=company_revenue.consolidated_revenue,
                revenue_growth=tuple(growth),
                source_by_year=company_revenue.source_by_year,
                confidence_by_year=company_revenue.confidence_by_year,
                unit=company_revenue.unit,
            )
            normalized_segments = (company_segment,)
            revenue_forecasts = (synthetic_revenue,)
        if not normalized_segments:
            candidates = (
                (*normalized_plan.decisions, *normalized_plan.overrides)
                if normalized_plan is not None
                else ()
            ) + self._coerce_overrides(
                overrides if forecast_overrides is None else forecast_overrides
            )
            for candidate in candidates:
                if (
                    _metric_key(candidate.metric)
                    in {_MARGIN, _PROFIT, "r_and_d", "sg_and_a", "other_operating_items", "ebit"}
                    and candidate.scope.value == "segment"
                ):
                    raise ValueError(
                        f"Explicit {_metric_key(candidate.metric)} target "
                        f"'{candidate.scope_id}' does not match a supplied canonical segment"
                    )
            raise ValueError("Operating economics requires at least one segment")
        explicit = self._explicit_paths(
            normalized_plan,
            overrides if forecast_overrides is None else forecast_overrides,
            years,
            policy,
            normalized_segments,
            ambiguous_ids,
        )
        forecast_by_id = {
            item.segment.segment_id: item for item in revenue_forecasts
        }
        segment_economics: list[SegmentOperatingEconomicsForecast] = []
        for segment in normalized_segments:
            segment_economics.append(
                self._forecast_segment(
                    segment,
                    forecast_by_id.get(segment.segment_id),
                    all_observations,
                    years,
                    explicit,
                    policy,
                    fiscal_period=fiscal_period,
                    period_key=period_key,
                )
            )

        selection = _select_consolidation_segments(normalized_segments)
        invalid_consolidated_selection = sum(
            segment.scope == "consolidated" for segment in normalized_segments
        ) > 1
        selected = tuple(
            item
            for item in segment_economics
            if not invalid_consolidated_selection
            and item.segment.segment_id
            in {segment.segment_id for segment in selection.segments}
        )
        company_revenue_path = self._company_revenue(
            company_revenue, revenue_forecasts, selection.segments, years
        )
        company = self._consolidate(
            company_id,
            years,
            company_revenue_path,
            segment_economics,
            selected,
            selection.warnings,
            policy,
            fiscal_period=fiscal_period,
            period_key=period_key,
        )
        company = self._apply_company_explicit_gross(
            company,
            explicit,
            policy,
        )
        # Attach the sibling result to the pre-existing revenue contract.  The
        # caller may also use the returned standalone contract directly.
        company = self.opex_ebit_engine.apply(
            company,
            all_observations,
            segments=normalized_segments,
            plan=normalized_plan,
            overrides=overrides if forecast_overrides is None else forecast_overrides,
            config=policy,
            fiscal_period=fiscal_period,
            period_key=period_key,
            ambiguous_segment_ids=ambiguous_ids,
        )
        return self.tax_nopat_engine.apply(
            company,
            all_observations,
            segments=normalized_segments,
            plan=normalized_plan,
            overrides=overrides if forecast_overrides is None else forecast_overrides,
            config=policy,
            fiscal_period=fiscal_period,
            period_key=period_key,
        )

    build = forecast
    forecast_company = forecast

    def forecast_segment(
        self,
        segment: OperatingSegment,
        segment_revenue_forecast: SegmentRevenueForecast,
        observations: Iterable[OperatingDriverObservation] = (),
        fiscal_years: Iterable[int] = (),
        **kwargs: Any,
    ) -> SegmentOperatingEconomicsForecast:
        """Return the one segment contract from the company-shaped service."""

        result = self.forecast(
            segments=(segment,),
            segment_revenue_forecast=segment_revenue_forecast,
            observations=observations,
            fiscal_years=fiscal_years,
            **kwargs,
        )
        return result.segment_economics[0]

    @staticmethod
    def _config(value: OperatingEconomicsForecastConfig | Mapping[str, Any]) -> OperatingEconomicsForecastConfig:
        return value if isinstance(value, OperatingEconomicsForecastConfig) else OperatingEconomicsForecastConfig.model_validate(value)

    @staticmethod
    def _years(
        values: Iterable[int],
        forecasts: Sequence[SegmentRevenueForecast],
        company: CompanyOperatingForecast | None,
    ) -> tuple[int, ...]:
        years = tuple(int(year) for year in values)
        if not years:
            if company is not None:
                years = company.fiscal_years
            elif forecasts:
                years = forecasts[0].fiscal_years
        if not years or tuple(sorted(years)) != years or len(set(years)) != len(years):
            raise ValueError("Operating economics fiscal_years must be sorted and unique")
        return years

    @staticmethod
    def _segments(
        segments: Iterable[OperatingSegment],
        forecasts: Sequence[SegmentRevenueForecast],
    ) -> tuple[OperatingSegment, ...]:
        result: dict[str, OperatingSegment] = {}
        for item in segments:
            segment = item if isinstance(item, OperatingSegment) else OperatingSegment.model_validate(item)
            result.setdefault(segment.segment_id, segment)
        for forecast in forecasts:
            result.setdefault(forecast.segment.segment_id, forecast.segment)
        return tuple(result.values())

    @staticmethod
    def _revenue_forecasts(
        value: Iterable[SegmentRevenueForecast]
        | SegmentRevenueForecast
        | CompanyOperatingForecast
        | None,
        company: CompanyOperatingForecast | None,
    ) -> tuple[SegmentRevenueForecast, ...]:
        if isinstance(value, CompanyOperatingForecast):
            return value.segment_forecasts
        if isinstance(value, SegmentRevenueForecast):
            return (value,)
        if value is None and company is not None:
            return company.segment_forecasts
        return tuple(value or ())

    @staticmethod
    def _coerce_observation(
        value, management: bool = False
    ) -> OperatingDriverObservation | None:
        if management and not isinstance(value, (OperatingDriverObservation, Mapping)):
            guidance_observation = _management_guidance_to_observation(value)
            if guidance_observation is None:
                if hasattr(value, "metric"):
                    return None
                observation = OperatingDriverObservation.model_validate(value)
            else:
                observation = guidance_observation
        else:
            observation = value if isinstance(value, OperatingDriverObservation) else OperatingDriverObservation.model_validate(value)
        if management and observation.origin != "management_guidance":
            observation = observation.model_copy(update={"origin": "management_guidance"})
        return observation

    @staticmethod
    def _observation_items(value) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, (OperatingDriverObservation, Mapping)):
            return (value,)
        if hasattr(value, "records"):
            return tuple(value.records)
        if hasattr(value, "eligible_records"):
            return tuple(value.eligible_records)
        if hasattr(value, "applications"):
            return tuple(item.guidance for item in value.applications)
        if isinstance(value, (str, bytes)):
            return (value,)
        try:
            return tuple(value)
        except TypeError:
            return (value,)

    @staticmethod
    def _coerce_plan(value) -> ForecastPlan | None:
        if value is None:
            return None
        return value if isinstance(value, ForecastPlan) else ForecastPlan.model_validate(value)

    def _explicit_paths(
        self,
        plan: ForecastPlan | None,
        overrides: Iterable[ForecastOverride] | Mapping[Any, Any],
        years: tuple[int, ...],
        policy: OperatingEconomicsForecastConfig,
        segments: Sequence[OperatingSegment],
        ambiguous_segment_ids: set[str],
    ) -> dict[tuple[str, str], _ExplicitPath]:
        records: list[ForecastDecision | ForecastOverride] = []
        if plan is not None:
            records.extend(plan.decisions)
            records.extend(plan.overrides)
        records.extend(self._coerce_overrides(overrides))
        paths: dict[tuple[str, str], _ExplicitPath] = {}
        for record in records:
            metric = _metric_key(record.metric)
            if metric not in {_MARGIN, _PROFIT}:
                continue
            if record.strategy != ForecastStrategy.EXPLICIT:
                continue
            path = record.explicit_path
            if path is None:
                raise ValueError(
                    f"Explicit {metric} decision for {record.scope_id} requires explicit_path"
                )
            values = tuple(Decimal(item) for item in path)
            if len(values) not in {1, len(years)}:
                raise ValueError(
                    f"Explicit {metric} path must contain one value or exactly the fiscal horizon"
                )
            for item in values:
                if not item.is_finite():
                    raise ValueError(f"Explicit {metric} path must contain finite values")
                if metric == _MARGIN and not policy.gross_margin_min <= item <= policy.gross_margin_max:
                    raise ValueError(
                        f"Explicit gross margin must be between {policy.gross_margin_min} and {policy.gross_margin_max} percentage points"
                    )
                if metric == _PROFIT and item < 0:
                    raise ValueError("Explicit gross profit cannot be negative")
            if record.basis is not None:
                expected_basis = (
                    ForecastValueBasis.PERCENT_OF_REVENUE
                    if metric == _MARGIN
                    else ForecastValueBasis.ABSOLUTE
                )
                if record.basis != expected_basis:
                    raise ValueError(
                        f"Explicit {metric} paths require basis={expected_basis.value}"
                    )
            if record.scope.value == "segment":
                canonical_id = canonical_operating_segment_id(record.scope_id) or record.scope_id
                supplied_ids = {segment.segment_id for segment in segments}
                if canonical_id in ambiguous_segment_ids:
                    raise ValueError(
                        f"Explicit {metric} target '{record.scope_id}' is ambiguous "
                        "among supplied canonical segments"
                    )
                if canonical_id not in supplied_ids:
                    raise ValueError(
                        f"Explicit {metric} target '{record.scope_id}' does not match "
                        "a supplied canonical segment"
                    )
                target_id = canonical_id
            else:
                target_id = "company"
            if len(values) == 1:
                values = values * len(years)
            path_references = (
                (record.provenance,)
                if isinstance(record.provenance, EvidenceReference)
                else ()
            )
            paths[(target_id, metric)] = _ExplicitPath(
                values,
                record.provenance,
                path_references,
            )
        return paths

    @staticmethod
    def _coerce_overrides(value) -> tuple[ForecastOverride, ...]:
        if value is None:
            return ()
        if isinstance(value, ForecastOverride):
            return (value,)
        if isinstance(value, Mapping):
            if {"scope", "metric", "strategy"}.issubset(value):
                return (ForecastOverride.model_validate(value),)
            records = []
            for key, item in value.items():
                if isinstance(item, ForecastOverride):
                    records.append(item)
                    continue
                payload = dict(item) if isinstance(item, Mapping) else {}
                if isinstance(key, tuple):
                    if len(key) == 2:
                        payload.setdefault("scope", key[0])
                        payload.setdefault("metric", key[1])
                    elif len(key) == 3:
                        payload.setdefault("scope", key[0])
                        payload.setdefault("scope_id", key[1])
                        payload.setdefault("metric", key[2])
                elif isinstance(key, str) and ":" in key:
                    parts = key.split(":")
                    if len(parts) == 2:
                        payload.setdefault("scope", parts[0])
                        payload.setdefault("metric", parts[1])
                    elif len(parts) == 3:
                        payload.setdefault("scope", parts[0])
                        payload.setdefault("scope_id", parts[1])
                        payload.setdefault("metric", parts[2])
                records.append(payload)
            value = records
        if isinstance(value, (str, bytes)):
            return ()
        if isinstance(value, Mapping):
            return ()
        try:
            records = tuple(value or ())
        except TypeError:
            records = (value,)
        return tuple(
            item if isinstance(item, ForecastOverride) else ForecastOverride.model_validate(item)
            for item in records
        )

    def _forecast_segment(
        self,
        segment: OperatingSegment,
        revenue_forecast: SegmentRevenueForecast | None,
        observations: Sequence[OperatingDriverObservation],
        years: tuple[int, ...],
        explicit: Mapping[tuple[str, str], _ExplicitPath],
        policy: OperatingEconomicsForecastConfig,
        *,
        fiscal_period: str,
        period_key: str | None,
    ) -> SegmentOperatingEconomicsForecast:
        revenue_inputs = self._revenue_inputs(segment, revenue_forecast, years)
        segment_observations = tuple(
            item for item in observations if item.segment_id == segment.segment_id
        )
        margin_path: list[Decimal | None] = []
        profit_path: list[Decimal | None] = []
        records: list[OperatingEconomicsYear] = []
        warnings: list[str] = []
        identity_warnings: list[str] = []
        margin_supported: list[int] = []
        profit_supported: list[int] = []
        margin_errors: list[Decimal] = []
        profit_errors: list[Decimal] = []
        margin_attempted = _MARGIN in {_metric_key(item.driver_id) for item in segment_observations} or (
            segment.segment_id,
            _MARGIN,
        ) in explicit
        profit_attempted = bool(
            {_metric_key(item.driver_id) for item in segment_observations}
            & {_PROFIT, _COST}
        ) or (segment.segment_id, _PROFIT) in explicit
        for observation in segment_observations:
            metric = _metric_key(observation.driver_id)
            if metric not in {_MARGIN, _PROFIT, _COST}:
                continue
            if self._observation_compatible(
                observation,
                segment,
                metric,
                fiscal_period=fiscal_period,
                period_key=period_key,
                policy=policy,
            ):
                continue
            reason = self._observation_incompatibility_reason(
                observation,
                segment,
                metric,
                fiscal_period=fiscal_period,
                period_key=period_key,
                policy=policy,
            )
            warnings.append(
                f"FY{observation.fiscal_year}: {metric} evidence unavailable ({reason})"
            )

        for index, year in enumerate(years):
            revenue = revenue_inputs[index]
            margin = self._select_margin(
                segment,
                year,
                index,
                segment_observations,
                revenue,
                explicit,
                policy,
                fiscal_period=fiscal_period,
                period_key=period_key,
            )
            profit = self._select_profit(
                segment,
                year,
                index,
                segment_observations,
                revenue,
                margin,
                explicit,
                policy,
                fiscal_period=fiscal_period,
                period_key=period_key,
            )
            if margin is not None:
                margin_path.append(margin.value)
                margin_supported.append(year)
            else:
                margin_path.append(None)
            if profit is not None:
                profit_path.append(profit.value)
                profit_supported.append(year)
            else:
                profit_path.append(None)

            expected = None
            identity_error = None
            audit: list[str] = []
            if (
                profit is not None
                and profit.method == "forecast_revenue_less_management_cost_of_revenue"
                and any(
                    _metric_key(item.driver_id) == _PROFIT
                    and item.origin in _DIRECT_ORIGINS
                    and item.fiscal_year == year
                    and self._observation_compatible(
                        item,
                        segment,
                        _PROFIT,
                        fiscal_period=fiscal_period,
                        period_key=period_key,
                        policy=policy,
                    )
                    for item in segment_observations
                )
            ):
                audit.append(
                    "gross_profit_precedence=management_guided_cost_over_direct_reported"
                )
                warnings.append(
                    f"FY{year}: management-guided cost-derived gross profit "
                    "superseded directly reported gross profit"
                )
            if revenue.value is None or margin is None or revenue.value == 0:
                if revenue.value == 0 and margin is not None:
                    warnings.append(f"FY{year}: gross-margin identity unavailable for zero revenue")
            else:
                expected = revenue.value * margin.value / Decimal(100)
                if profit is not None:
                    identity_error = abs(profit.value - expected)
                    margin_errors.append(abs(margin.value - (profit.value / revenue.value * Decimal(100))))
                    profit_errors.append(identity_error)
                    if identity_error != 0:
                        message = (
                            f"FY{year}: gross-profit identity warning; expected {expected} "
                            f"from revenue × gross margin but selected {profit.value}"
                        )
                        identity_warnings.append(message)
                        warnings.append(message)
                    audit.append(f"gross_profit_expected={expected}")
                    audit.append(f"gross_profit_identity_error={identity_error}")
            if margin is None and profit is not None and revenue.value not in (None, Decimal(0)):
                # This branch is normally handled by _select_margin, but keeps
                # the record truthful if a future explicit GP has no usable
                # historical revenue for margin reconstruction.
                warnings.append(f"FY{year}: gross margin unavailable for selected gross profit")
            source, confidence, provenance, source_refs, method = self._combine_metric_candidates(
                margin, profit, revenue
            )
            margin_provenance, margin_chain, margin_refs = _candidate_provenance(
                margin, revenue
            )
            profit_provenance, profit_chain, profit_refs = _candidate_provenance(
                profit, revenue
            )
            provenance_chain = _dedupe_values(
                (*margin_chain, *profit_chain)
            )
            if margin is None and profit is None:
                source = _UNAVAILABLE
                confidence = "low"
                method = "unavailable"
            records.append(
                OperatingEconomicsYear(
                    fiscal_year=year,
                    fiscal_period=fiscal_period,
                    period_key=period_key,
                    revenue=revenue.value,
                    gross_margin=margin.value if margin is not None else None,
                    gross_profit=profit.value if profit is not None else None,
                    source=source,
                    confidence=confidence,
                    provenance=provenance,
                    provenance_chain=provenance_chain,
                    source_provenance=source_refs,
                    gross_margin_provenance=margin_provenance,
                    gross_margin_provenance_chain=margin_chain,
                    gross_margin_source_provenance=margin_refs,
                    gross_profit_provenance=profit_provenance,
                    gross_profit_provenance_chain=profit_chain,
                    gross_profit_source_provenance=profit_refs,
                    method=method,
                    audit=tuple(audit),
                    expected_gross_profit=expected,
                    identity_error=identity_error,
                )
            )

        margin_diag = self._metric_diagnostics(
            _MARGIN,
            years,
            margin_supported,
            [records[index].confidence for index, year in enumerate(years) if year in margin_supported],
            margin_errors,
            margin_attempted or bool(margin_supported),
            warnings,
            identity_warnings,
        )
        profit_diag = self._metric_diagnostics(
            _PROFIT,
            years,
            profit_supported,
            [records[index].confidence for index, year in enumerate(years) if year in profit_supported],
            profit_errors,
            profit_attempted or bool(profit_supported),
            warnings,
            identity_warnings,
        )
        completeness = (
            Decimal(len(set(margin_supported) & set(profit_supported))) / Decimal(len(years))
            if years and (margin_attempted or profit_attempted)
            else None
        )
        diagnostics = SegmentOperatingEconomicsDiagnostics(
            gross_margin=margin_diag,
            gross_profit=profit_diag,
            completeness=completeness,
            identity_warnings=tuple(dict.fromkeys(identity_warnings)),
            warnings=tuple(dict.fromkeys(warnings)),
        )
        return SegmentOperatingEconomicsForecast(
            segment=segment,
            fiscal_years=years,
            revenue=tuple(item.value for item in revenue_inputs),
            gross_margin=tuple(margin_path),
            gross_profit=tuple(profit_path),
            years=tuple(records),
            source_by_year={item.fiscal_year: item.source for item in records},
            confidence_by_year={item.fiscal_year: item.confidence for item in records},
            provenance_by_year={
                item.fiscal_year: item.provenance
                for item in records
                if item.provenance is not None
            },
            provenance_chain_by_year={
                item.fiscal_year: item.provenance_chain
                for item in records
                if item.provenance_chain
            },
            source_provenance_by_year={
                item.fiscal_year: item.source_provenance
                for item in records
                if item.source_provenance
            },
            gross_margin_provenance_by_year={
                item.fiscal_year: item.gross_margin_provenance
                for item in records
                if item.gross_margin_provenance is not None
            },
            gross_margin_provenance_chain_by_year={
                item.fiscal_year: item.gross_margin_provenance_chain
                for item in records
                if item.gross_margin_provenance_chain
            },
            gross_margin_source_provenance_by_year={
                item.fiscal_year: item.gross_margin_source_provenance
                for item in records
                if item.gross_margin_source_provenance
            },
            gross_profit_provenance_by_year={
                item.fiscal_year: item.gross_profit_provenance
                for item in records
                if item.gross_profit_provenance is not None
            },
            gross_profit_provenance_chain_by_year={
                item.fiscal_year: item.gross_profit_provenance_chain
                for item in records
                if item.gross_profit_provenance_chain
            },
            gross_profit_source_provenance_by_year={
                item.fiscal_year: item.gross_profit_source_provenance
                for item in records
                if item.gross_profit_source_provenance
            },
            method_by_year={item.fiscal_year: item.method for item in records},
            audit_by_year={item.fiscal_year: item.audit for item in records if item.audit},
            diagnostics=diagnostics,
            warnings=tuple(dict.fromkeys(warnings)),
            unit=segment.currency or (revenue_forecast.unit if revenue_forecast else "currency"),
        )

    def _select_margin(
        self,
        segment: OperatingSegment,
        year: int,
        index: int,
        observations: Sequence[OperatingDriverObservation],
        revenue: _RevenueInput,
        explicit: Mapping[tuple[str, str], _ExplicitPath],
        policy: OperatingEconomicsForecastConfig,
        *,
        fiscal_period: str,
        period_key: str | None,
    ) -> _Candidate | None:
        key = (segment.segment_id, _MARGIN)
        if key in explicit:
            path = explicit[key]
            return _Candidate(
                _MARGIN,
                path.values[index],
                "explicit",
                "high",
                provenance=path.provenance,
                source_provenance=path.source_provenance,
                method="forecast_plan_explicit_gross_margin",
                explicit=True,
            )
        candidate = self._observation_candidate(
            _MARGIN,
            segment,
            year,
            observations,
            fiscal_period=fiscal_period,
            period_key=period_key,
            management_only=True,
            policy=policy,
        )
        if candidate is not None:
            return candidate
        candidate = self._observation_candidate(
            _MARGIN,
            segment,
            year,
            observations,
            fiscal_period=fiscal_period,
            period_key=period_key,
            direct_only=True,
            policy=policy,
        )
        if candidate is not None:
            return candidate
        candidate = self._historical_margin(
            segment,
            year,
            observations,
            fiscal_period=fiscal_period,
            period_key=period_key,
            policy=policy,
        )
        if candidate is not None:
            return candidate
        historical = self._historical_margin_values(
            segment,
            year,
            observations,
            fiscal_period=fiscal_period,
            period_key=period_key,
            policy=policy,
        )
        if historical:
            values = historical[-policy.historical_window :]
            if policy.normalization_method == "weighted_recent":
                denominator = Decimal(sum(range(1, len(values) + 1)))
                value = sum(
                    (
                        item.value * Decimal(position)
                        for position, item in enumerate(values, 1)
                    ),
                    Decimal(0),
                ) / denominator
                source = "normalized_historical_weighted_recent"
            else:
                ordered = sorted(item.value for item in values)
                midpoint = len(ordered) // 2
                value = (
                    ordered[midpoint]
                    if len(ordered) % 2
                    else (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)
                )
                source = "normalized_historical"
            return _Candidate(
                _MARGIN,
                value,
                source,
                _worst_confidence(tuple(item.confidence for item in values)),
                provenance=values[-1].provenance,
                source_provenance=_references(values),
                observations=tuple(
                    item
                    for value_item in values
                    for item in value_item.observations
                ),
                method=f"{policy.normalization_method}_recent_historical_gross_margin",
            )

        # Gross-profit/cost evidence is deliberately a final margin fallback.
        # It must not displace a margin-specific historical or direct source.
        explicit_profit = explicit.get((segment.segment_id, _PROFIT))
        if (
            explicit_profit is not None
            and revenue.value is not None
            and revenue.source != _UNAVAILABLE
            and revenue.value != 0
            and policy.gross_margin_min
            <= explicit_profit.values[index] / revenue.value * Decimal(100)
            <= policy.gross_margin_max
        ):
            return _Candidate(
                _MARGIN,
                explicit_profit.values[index] / revenue.value * Decimal(100),
                "explicit",
                "high",
                provenance=explicit_profit.provenance or revenue.provenance,
                source_provenance=tuple(
                    dict.fromkeys(
                        (*explicit_profit.source_provenance, *revenue.source_provenance)
                    )
                ),
                method="explicit_gross_profit_over_forecast_revenue",
                consumes_revenue=True,
                explicit=True,
            )
        management_profit = self._observation_candidate(
            _PROFIT,
            segment,
            year,
            observations,
            fiscal_period=fiscal_period,
            period_key=period_key,
            management_only=True,
            policy=policy,
        )
        if (
            management_profit is not None
            and revenue.value is not None
            and revenue.source != _UNAVAILABLE
            and revenue.value != 0
            and policy.gross_margin_min
            <= management_profit.value / revenue.value * Decimal(100)
            <= policy.gross_margin_max
        ):
            return _Candidate(
                _MARGIN,
                management_profit.value / revenue.value * Decimal(100),
                "management_guidance",
                _worst_confidence((management_profit.confidence, revenue.confidence)),
                provenance=management_profit.provenance or revenue.provenance,
                source_provenance=tuple(
                    dict.fromkeys(
                        (*management_profit.source_provenance, *revenue.source_provenance)
                    )
                ),
                observations=management_profit.observations,
                method="management_gross_profit_over_forecast_revenue",
                consumes_revenue=True,
            )
        management_cost = self._observation_candidate(
            _COST,
            segment,
            year,
            observations,
            fiscal_period=fiscal_period,
            period_key=period_key,
            management_only=True,
            policy=policy,
        )
        if (
            management_cost is not None
            and revenue.value is not None
            and revenue.source != _UNAVAILABLE
            and revenue.value != 0
            and policy.gross_margin_min
            <= (revenue.value - management_cost.value) / revenue.value * Decimal(100)
            <= policy.gross_margin_max
        ):
            return _Candidate(
                _MARGIN,
                (revenue.value - management_cost.value)
                / revenue.value
                * Decimal(100),
                "management_guidance",
                _worst_confidence((management_cost.confidence, revenue.confidence)),
                provenance=management_cost.provenance or revenue.provenance,
                source_provenance=tuple(
                    dict.fromkeys(
                        (*management_cost.source_provenance, *revenue.source_provenance)
                    )
                ),
                observations=management_cost.observations,
                method="forecast_revenue_less_management_cost_over_revenue",
                consumes_revenue=True,
            )
        return None

    def _select_profit(
        self,
        segment: OperatingSegment,
        year: int,
        index: int,
        observations: Sequence[OperatingDriverObservation],
        revenue: _RevenueInput,
        margin: _Candidate | None,
        explicit: Mapping[tuple[str, str], _ExplicitPath],
        policy: OperatingEconomicsForecastConfig,
        *,
        fiscal_period: str,
        period_key: str | None,
    ) -> _Candidate | None:
        key = (segment.segment_id, _PROFIT)
        if key in explicit:
            path = explicit[key]
            return _Candidate(
                _PROFIT,
                path.values[index],
                "explicit",
                "high",
                provenance=path.provenance,
                source_provenance=path.source_provenance,
                method="forecast_plan_explicit_gross_profit",
                explicit=True,
            )
        candidate = self._observation_candidate(
            _PROFIT,
            segment,
            year,
            observations,
            fiscal_period=fiscal_period,
            period_key=period_key,
            management_only=True,
            policy=policy,
            )
        if candidate is not None:
            if not self._candidate_matches_revenue_unit(candidate, revenue):
                candidate = None
        if candidate is not None:
            return candidate
        management_cost = self._observation_candidate(
            _COST,
            segment,
            year,
            observations,
            fiscal_period=fiscal_period,
            period_key=period_key,
            management_only=True,
            policy=policy,
        )
        if (
            management_cost is not None
            and revenue.value is not None
            and revenue.source != _UNAVAILABLE
        ):
            return _Candidate(
                _PROFIT,
                revenue.value - management_cost.value,
                "management_guidance",
                _worst_confidence((management_cost.confidence, revenue.confidence)),
                provenance=management_cost.provenance or revenue.provenance,
                source_provenance=tuple(
                    dict.fromkeys(
                        (*management_cost.source_provenance, *revenue.source_provenance)
                    )
                ),
                observations=management_cost.observations,
                method="forecast_revenue_less_management_cost_of_revenue",
                consumes_revenue=True,
            )
        candidate = self._observation_candidate(
            _PROFIT,
            segment,
            year,
            observations,
            fiscal_period=fiscal_period,
            period_key=period_key,
            direct_only=True,
            policy=policy,
        )
        if candidate is not None:
            if self._candidate_matches_revenue_unit(candidate, revenue):
                return candidate
        candidate = self._historical_profit(
            segment,
            year,
            observations,
            fiscal_period=fiscal_period,
            period_key=period_key,
            policy=policy,
        )
        if candidate is not None:
            return candidate
        if margin is None or revenue.value is None or revenue.source == _UNAVAILABLE:
            return None
        return _Candidate(
            _PROFIT,
            revenue.value * margin.value / Decimal(100),
            "derived_gross_margin",
            _worst_confidence((margin.confidence, revenue.confidence)),
            provenance=margin.provenance or revenue.provenance,
            source_provenance=tuple(dict.fromkeys((*margin.source_provenance, *revenue.source_provenance))),
            observations=margin.observations,
            method="revenue_times_gross_margin",
            consumes_revenue=True,
        )

    def _observation_candidate(
        self,
        metric: str,
        segment: OperatingSegment,
        year: int,
        observations: Sequence[OperatingDriverObservation],
        *,
        fiscal_period: str,
        period_key: str | None,
        management_only: bool = False,
        direct_only: bool = False,
        policy: OperatingEconomicsForecastConfig,
    ) -> _Candidate | None:
        candidates: list[tuple[int, OperatingDriverObservation, Decimal]] = []
        for position, observation in enumerate(observations):
            if _metric_key(observation.driver_id) != metric or observation.fiscal_year != year:
                continue
            if management_only and observation.origin != "management_guidance":
                continue
            if direct_only and observation.origin not in _DIRECT_ORIGINS:
                continue
            if not self._observation_compatible(
                observation,
                segment,
                metric,
                fiscal_period=fiscal_period,
                period_key=period_key,
                policy=policy,
            ):
                continue
            value = self._value(observation, metric)
            if value is None:
                continue
            if metric == _PROFIT and observation.origin != "management_guidance" and value < 0:
                # Gross loss is valid evidence; only manual gross-profit paths
                # have the non-negative constraint.
                pass
            candidates.append((position, observation, value))
        if not candidates:
            return None
        _, observation, value = max(
            candidates,
            key=lambda item: (
                1 if item[1].is_total else 0,
                _CONFIDENCE_RANK[item[1].confidence],
                item[1].evidence is not None,
                -item[0],
            ),
        )
        return _Candidate(
            metric,
            value,
            "management_guidance" if observation.origin == "management_guidance" else observation.origin,
            observation.confidence,
            provenance=observation.provenance or observation.evidence,
            source_provenance=_references((observation,)),
            observations=(observation,),
            method="management_guidance_observation" if observation.origin == "management_guidance" else "reported_gross_economics",
        )

    def _historical_margin(
        self,
        segment: OperatingSegment,
        year: int,
        observations: Sequence[OperatingDriverObservation],
        *,
        fiscal_period: str,
        period_key: str | None,
        policy: OperatingEconomicsForecastConfig,
    ) -> _Candidate | None:
        gp = self._observation_candidate(
            _PROFIT,
            segment,
            year,
            observations,
            fiscal_period=fiscal_period,
            period_key=period_key,
            direct_only=True,
            policy=policy,
        )
        revenue = self._historical_revenue_candidate(
            segment, year, observations, fiscal_period=fiscal_period, period_key=period_key, policy=policy
        )
        if gp is None:
            gp = self._historical_profit(
                segment,
                year,
                observations,
                fiscal_period=fiscal_period,
                period_key=period_key,
                policy=policy,
            )
        if (
            gp is None
            or revenue is None
            or revenue.value == 0
            or not self._pair_compatible(gp, revenue)
        ):
            return None
        margin = gp.value / revenue.value * Decimal(100)
        if not policy.gross_margin_min <= margin <= policy.gross_margin_max:
            return None
        return _Candidate(
            _MARGIN,
            margin,
            "derived_historical",
            _worst_confidence((gp.confidence, revenue.confidence)),
            provenance=gp.provenance or revenue.provenance,
            source_provenance=tuple(dict.fromkeys((*gp.source_provenance, *revenue.source_provenance))),
            observations=(*gp.observations, *revenue.observations),
            method="gross_profit_over_revenue",
            consumes_revenue=True,
        )

    def _historical_profit(
        self,
        segment: OperatingSegment,
        year: int,
        observations: Sequence[OperatingDriverObservation],
        *,
        fiscal_period: str,
        period_key: str | None,
        policy: OperatingEconomicsForecastConfig,
    ) -> _Candidate | None:
        cost = self._observation_candidate(
            _COST,
            segment,
            year,
            observations,
            fiscal_period=fiscal_period,
            period_key=period_key,
            direct_only=True,
            policy=policy,
        )
        revenue = self._historical_revenue_candidate(
            segment,
            year,
            observations,
            fiscal_period=fiscal_period,
            period_key=period_key,
            policy=policy,
        )
        if (
            cost is None
            or revenue is None
            or not self._pair_compatible(cost, revenue)
        ):
            return None
        return _Candidate(
            _PROFIT,
            revenue.value - cost.value,
            "derived_historical",
            _worst_confidence((cost.confidence, revenue.confidence)),
            provenance=cost.provenance or revenue.provenance,
            source_provenance=tuple(dict.fromkeys((*cost.source_provenance, *revenue.source_provenance))),
            observations=(*cost.observations, *revenue.observations),
            method="gross_profit_from_revenue_less_cost_of_revenue",
            consumes_revenue=True,
        )

    def _historical_margin_values(
        self,
        segment: OperatingSegment,
        before_year: int,
        observations: Sequence[OperatingDriverObservation],
        *,
        fiscal_period: str,
        period_key: str | None,
        policy: OperatingEconomicsForecastConfig,
    ) -> list[_Candidate]:
        years = sorted(
            {
                item.fiscal_year
                for item in observations
                if item.segment_id == segment.segment_id and item.fiscal_year < before_year
            }
        )
        result = []
        for year in years:
            candidate = self._observation_candidate(
                _MARGIN,
                segment,
                year,
                observations,
                fiscal_period=fiscal_period,
                period_key=period_key,
                direct_only=True,
                policy=policy,
            ) or self._historical_margin(
                segment,
                year,
                observations,
                fiscal_period=fiscal_period,
                period_key=period_key,
                policy=policy,
            )
            if candidate is not None:
                result.append(candidate)
        return result

    def _historical_revenue_candidate(
        self,
        segment: OperatingSegment,
        year: int,
        observations: Sequence[OperatingDriverObservation],
        *,
        fiscal_period: str,
        period_key: str | None,
        policy: OperatingEconomicsForecastConfig,
    ) -> _Candidate | None:
        return self._observation_candidate(
            _REVENUE,
            segment,
            year,
            observations,
            fiscal_period=fiscal_period,
            period_key=period_key,
            direct_only=True,
            policy=policy,
        )

    def _observation_compatible(
        self,
        observation: OperatingDriverObservation,
        segment: OperatingSegment,
        metric: str,
        *,
        fiscal_period: str,
        period_key: str | None,
        policy: OperatingEconomicsForecastConfig,
    ) -> bool:
        if not operating_periods_compatible(
            observation.fiscal_period, fiscal_period, observation.period_key, period_key
        ):
            return False
        if not self._scope_compatible(observation, segment):
            return False
        value = self._value(observation, metric)
        if value is None:
            return False
        if metric == _MARGIN:
            return (
                observation.unit.casefold() in _RATE_UNITS
                or "percent" in observation.unit.casefold()
                or "percentage" in observation.unit.casefold()
                or "basis" in observation.unit.casefold()
            ) and policy.gross_margin_min <= value <= policy.gross_margin_max
        if metric in {_REVENUE, _PROFIT, _COST}:
            if not self._currency_unit(observation.unit):
                return False
            if segment.currency and observation.currency and segment.currency != observation.currency:
                return False
            segment_code = _currency_code(segment.currency or "")
            observation_code = _currency_code(observation.unit)
            if segment_code and observation_code and segment_code != observation_code:
                return False
            return value >= 0 if metric in {_REVENUE, _COST} else True
        return False

    def _observation_incompatibility_reason(
        self,
        observation: OperatingDriverObservation,
        segment: OperatingSegment,
        metric: str,
        *,
        fiscal_period: str,
        period_key: str | None,
        policy: OperatingEconomicsForecastConfig,
    ) -> str:
        if not operating_periods_compatible(
            observation.fiscal_period, fiscal_period, observation.period_key, period_key
        ):
            return "incompatible fiscal period"
        if not self._scope_compatible(observation, segment):
            return "scope mismatch"
        value = self._value(observation, metric)
        if metric == _MARGIN and not (
            value is not None and policy.gross_margin_min <= value <= policy.gross_margin_max
        ):
            return "gross-margin value is outside configured bounds"
        if metric in {_REVENUE, _PROFIT, _COST} and not self._currency_unit(observation.unit):
            return "incompatible currency/unit dimension"
        if segment.currency and observation.currency and segment.currency != observation.currency:
            return "currency mismatch"
        segment_code = _currency_code(segment.currency or "")
        observation_code = _currency_code(observation.unit)
        if segment_code and observation_code and segment_code != observation_code:
            return "currency mismatch"
        if value is None:
            return "missing numeric value"
        return "incompatible currency/unit dimension"

    @staticmethod
    def _scope_compatible(observation: OperatingDriverObservation, segment: OperatingSegment) -> bool:
        if observation.is_component:
            # The revenue forecast is a segment-total path, not a component
            # path.  A component fact therefore cannot be applied to it.
            return False
        scope = (observation.scope or "").casefold()
        if segment.scope == "consolidated":
            return not scope or scope in {"company", "consolidated", "total"}
        if scope in {"company", "consolidated", "total"} and segment.scope != "consolidated":
            return False
        if scope and scope != segment.scope.casefold() and not (
            scope == "segment" and segment.scope == "segment"
        ) and segment.scope != "consolidated":
            return False
        return True

    def _pair_compatible(self, left: _Candidate, right: _Candidate) -> bool:
        if not left.observations or not right.observations:
            return True
        for first in left.observations:
            for second in right.observations:
                if not operating_periods_compatible(
                    first.fiscal_period,
                    second.fiscal_period,
                    first.period_key,
                    second.period_key,
                ):
                    continue
                if first.is_component != second.is_component:
                    continue
                if first.is_component and (
                    first.scope != second.scope
                    or first.scope_evidence != second.scope_evidence
                ):
                    continue
                scopes = {item.scope for item in (first, second) if item.scope}
                if len(scopes) > 1:
                    continue
                if not self._currencies_compatible(first, second):
                    continue
                return True
        return False

    @staticmethod
    def _currencies_compatible(left: OperatingDriverObservation, right: OperatingDriverObservation) -> bool:
        if left.currency and right.currency and left.currency != right.currency:
            return False
        left_code = _currency_code(left.unit)
        right_code = _currency_code(right.unit)
        return not left_code or not right_code or left_code == right_code

    @staticmethod
    def _currency_unit(unit: str) -> bool:
        normalized = unit.casefold()
        return any(token in normalized for token in ("usd", "eur", "gbp", "jpy", "cny", "cad", "aud", "chf", "currency", "dollar", "€", "$", "£"))

    @staticmethod
    def _value(observation: OperatingDriverObservation, metric: str) -> Decimal | None:
        try:
            value = observation.normalized_value
        except ValueError:
            return None
        unit = observation.unit.casefold().replace(" ", "_")
        if metric == _MARGIN:
            if "bp" in unit or "basis_point" in unit:
                value /= Decimal(100)
            elif unit in {"ratio", "rate", "decimal", "fraction"}:
                value *= Decimal(100)
        return value

    @staticmethod
    def _revenue_inputs(
        segment: OperatingSegment,
        forecast: SegmentRevenueForecast | None,
        years: tuple[int, ...],
    ) -> tuple[_RevenueInput, ...]:
        if forecast is None:
            return tuple(
                _RevenueInput(None, _UNAVAILABLE, "low", "currency")
                for _year in years
            )
        if forecast.fiscal_years != years:
            raise ValueError("Segment revenue and operating-economics years must match")
        result = []
        for index, year in enumerate(years):
            source = forecast.source_by_year.get(year, "revenue_forecast")
            value = forecast.revenue[index]
            references = tuple(
                item.provenance
                for item in forecast.driver_forecasts
                if item.fiscal_year == year and isinstance(item.provenance, EvidenceReference)
            )
            result.append(
                _RevenueInput(
                    value,
                    source,
                    forecast.confidence_by_year.get(year, "low"),
                    forecast.unit,
                    next(
                        (
                            item.provenance
                            for item in forecast.driver_forecasts
                            if item.fiscal_year == year and item.provenance is not None
                        ),
                        None,
                    ),
                    tuple(dict.fromkeys(references)),
                )
            )
        return tuple(result)

    @staticmethod
    def _candidate_matches_revenue_unit(
        candidate: _Candidate, revenue: _RevenueInput
    ) -> bool:
        if candidate.metric not in {_PROFIT, _COST} or not candidate.observations:
            return True
        revenue_code = _currency_code(revenue.unit)
        if not revenue_code:
            return True
        return all(
            not _currency_code(item.unit)
            or _currency_code(item.unit) == revenue_code
            for item in candidate.observations
        )

    @staticmethod
    def _combine_metric_candidates(
        margin: _Candidate | None,
        profit: _Candidate | None,
        revenue: _RevenueInput,
    ) -> tuple[str, str, Any, tuple[EvidenceReference, ...], str]:
        candidates = tuple(item for item in (margin, profit) if item is not None)
        if not candidates:
            return _UNAVAILABLE, "low", None, (), "unavailable"
        source = _combine_sources(tuple(item.source for item in candidates))
        confidence = _worst_confidence(tuple(item.confidence for item in candidates))
        provenance = next((item.provenance for item in candidates if item.provenance is not None), revenue.provenance)
        references = tuple(
            dict.fromkeys(
                ref
                for item in candidates
                for ref in item.source_provenance
            )
        )
        if any(item.consumes_revenue for item in candidates):
            references = references + tuple(
                ref for ref in revenue.source_provenance if ref not in references
            )
        methods = tuple(dict.fromkeys(item.method for item in candidates))
        return source, confidence, provenance, references, "+".join(methods)

    @staticmethod
    def _metric_diagnostics(
        metric: str,
        years: tuple[int, ...],
        supported: Sequence[int],
        confidences: Sequence[str],
        errors: Sequence[Decimal],
        attempted: bool,
        warnings: Sequence[str],
        identity_warnings: Sequence[str],
    ) -> OperatingEconomicsMetricDiagnostics:
        coverage = Decimal(len(supported)) / Decimal(len(years)) if attempted and years else None
        error = sum(errors, Decimal(0)) / Decimal(len(errors)) if errors else None
        metric_warnings = tuple(
            dict.fromkeys(
                item
                for item in warnings
                if metric.replace("gross_", "") in item.casefold()
                or metric in item.casefold()
            )
        )
        return OperatingEconomicsMetricDiagnostics(
            metric=metric,
            coverage=coverage,
            supported_years=tuple(sorted(supported)),
            confidence=_worst_confidence(tuple(confidences)) if confidences else "low",
            reconstruction_error=error,
            completeness=coverage,
            warnings=metric_warnings,
            identity_warnings=tuple(identity_warnings),
        )

    def _company_revenue(
        self,
        company: CompanyOperatingForecast | None,
        forecasts: Sequence[SegmentRevenueForecast],
        selected_segments: Sequence[OperatingSegment],
        years: tuple[int, ...],
    ) -> tuple[Decimal | None, ...]:
        if company is not None:
            return tuple(company.consolidated_revenue)
        by_id = {item.segment.segment_id: item for item in forecasts}
        result = []
        for index, _year in enumerate(years):
            values = [
                forecast.revenue[index]
                for segment in selected_segments
                if (forecast := by_id.get(segment.segment_id)) is not None
                and forecast.source_by_year.get(years[index], "revenue_forecast")
                != _UNAVAILABLE
            ]
            result.append(sum(values, Decimal(0)) if len(values) == len(selected_segments) and values else None)
        return tuple(result)

    def _apply_company_explicit_gross(
        self,
        company: CompanyOperatingEconomicsForecast,
        explicit: Mapping[tuple[str, str], _ExplicitPath],
        policy: OperatingEconomicsForecastConfig,
    ) -> CompanyOperatingEconomicsForecast:
        """Apply company gross paths after segment consolidation.

        Segment consolidation remains the source of the base result.  A
        company-scoped explicit path is the only gross input allowed to replace
        that result, and this method updates every related audit field as one
        immutable contract update.  The OPEX stage consequently sees explicit
        company gross values as already selected and cannot replace them.
        """

        margin_path = explicit.get(("company", _MARGIN))
        profit_path = explicit.get(("company", _PROFIT))
        if margin_path is None and profit_path is None:
            return company

        years = company.fiscal_years
        profits = list(company.consolidated_gross_profit)
        margins = list(company.consolidated_gross_margin)
        source_by_year = dict(company.source_by_year)
        confidence_by_year = dict(company.confidence_by_year)
        provenance_by_year = dict(company.provenance_by_year)
        provenance_chain_by_year = dict(company.provenance_chain_by_year)
        source_references = dict(company.source_provenance_by_year)
        margin_provenance = dict(company.gross_margin_provenance_by_year)
        margin_chains = dict(company.gross_margin_provenance_chain_by_year)
        margin_references = dict(company.gross_margin_source_provenance_by_year)
        profit_provenance = dict(company.gross_profit_provenance_by_year)
        profit_chains = dict(company.gross_profit_provenance_chain_by_year)
        profit_references = dict(company.gross_profit_source_provenance_by_year)
        methods = dict(company.method_by_year)
        audits = dict(company.audit_by_year)
        warnings = list(company.warnings)
        identity_warnings = list(company.diagnostics.identity_warnings)
        margin_supported: list[int] = []
        profit_supported: list[int] = []
        margin_errors: list[Decimal] = []
        profit_errors: list[Decimal] = []
        explicit_margin_provenance = margin_path.provenance if margin_path else None
        explicit_profit_provenance = profit_path.provenance if profit_path else None
        explicit_margin_refs = margin_path.source_provenance if margin_path else ()
        explicit_profit_refs = profit_path.source_provenance if profit_path else ()

        for index, year in enumerate(years):
            explicit_margin = margin_path.values[index] if margin_path else None
            explicit_profit = profit_path.values[index] if profit_path else None
            margin = explicit_margin
            profit = explicit_profit
            audit: list[str] = []
            if explicit_margin is not None:
                audit.append(f"explicit_company_gross_margin={explicit_margin}")
            if explicit_profit is not None:
                audit.append(f"explicit_company_gross_profit={explicit_profit}")
            if explicit_margin is not None and explicit_profit is None:
                if company.consolidated_revenue[index] not in (None, Decimal(0)):
                    profit = (
                        company.consolidated_revenue[index]
                        * explicit_margin
                        / Decimal(100)
                    )
                    audit.append(f"gross_profit_derived_from_company_revenue={profit}")
                else:
                    audit.append("gross_profit_unavailable_company_revenue_missing_or_zero")
            elif explicit_profit is not None and explicit_margin is None:
                if company.consolidated_revenue[index] not in (None, Decimal(0)):
                    margin = (
                        explicit_profit
                        / company.consolidated_revenue[index]
                        * Decimal(100)
                    )
                    if not policy.gross_margin_min <= margin <= policy.gross_margin_max:
                        margin = None
                        audit.append("gross_margin_unavailable_outside_configured_bounds")
                    else:
                        audit.append(f"gross_margin_derived_from_company_gross_profit={margin}")
                else:
                    audit.append("gross_margin_unavailable_company_revenue_missing_or_zero")
            elif explicit_margin is not None and explicit_profit is not None:
                if company.consolidated_revenue[index] not in (None, Decimal(0)):
                    expected = (
                        company.consolidated_revenue[index]
                        * explicit_margin
                        / Decimal(100)
                    )
                    error = abs(explicit_profit - expected)
                    profit_errors.append(error)
                    margin_errors.append(error)
                    audit.append(f"gross_profit_expected_from_company_margin={expected}")
                    audit.append(f"company_gross_identity_error={error}")
                    if error:
                        warning = (
                            f"FY{year}: explicit company gross margin and gross profit "
                            f"differ by {error}"
                        )
                        warnings.append(warning)
                        identity_warnings.append(warning)
                else:
                    audit.append("company_gross_identity_unavailable_revenue_missing_or_zero")

            profits[index] = profit
            margins[index] = margin
            if margin is not None:
                margin_supported.append(year)
            if profit is not None:
                profit_supported.append(year)

            provenance_values = tuple(
                item
                for item in (explicit_margin_provenance, explicit_profit_provenance)
                if item is not None
            )
            references = tuple(
                dict.fromkeys((*explicit_margin_refs, *explicit_profit_refs))
            )
            source_by_year[year] = "explicit"
            confidence_by_year[year] = "high"
            methods[year] = "+".join(
                item
                for item in (
                    "forecast_plan_explicit_company_gross_margin"
                    if explicit_margin is not None
                    else None,
                    "forecast_plan_explicit_company_gross_profit"
                    if explicit_profit is not None
                    else None,
                )
                if item is not None
            )
            audits[year] = tuple(audit)
            for mapping in (
                provenance_by_year,
                provenance_chain_by_year,
                source_references,
                margin_provenance,
                margin_chains,
                margin_references,
                profit_provenance,
                profit_chains,
                profit_references,
            ):
                mapping.pop(year, None)
            if provenance_values:
                provenance_by_year[year] = provenance_values[0]
                provenance_chain_by_year[year] = provenance_values
            source_references[year] = references
            if margin is not None:
                margin_origin = (
                    explicit_margin_provenance
                    if explicit_margin is not None
                    else explicit_profit_provenance
                )
                margin_origin_refs = (
                    explicit_margin_refs
                    if explicit_margin is not None
                    else explicit_profit_refs
                )
                if margin_origin is not None:
                    margin_provenance[year] = margin_origin
                    margin_chains[year] = (margin_origin,)
                margin_references[year] = margin_origin_refs
            if profit is not None:
                profit_origin = (
                    explicit_profit_provenance
                    if explicit_profit is not None
                    else explicit_margin_provenance
                )
                profit_origin_refs = (
                    explicit_profit_refs
                    if explicit_profit is not None
                    else explicit_margin_refs
                )
                if profit_origin is not None:
                    profit_provenance[year] = profit_origin
                    profit_chains[year] = (profit_origin,)
                profit_references[year] = profit_origin_refs

        stale_prefix = tuple(
            f"FY{year}: consolidated gross" for year in years
        )
        warnings = [
            warning
            for warning in warnings
            if not any(prefix in warning for prefix in stale_prefix)
        ]
        diagnostics = company.diagnostics.model_copy(
            update={
                "completeness": (
                    Decimal(len(set(margin_supported) & set(profit_supported)))
                    / Decimal(len(years))
                    if years
                    else None
                ),
                "identity_warnings": tuple(dict.fromkeys(identity_warnings)),
                "warnings": tuple(dict.fromkeys(warnings)),
                "gross_margin": company.diagnostics.gross_margin.model_copy(
                    update={
                        "coverage": Decimal(len(margin_supported)) / Decimal(len(years)) if years else None,
                        "supported_years": tuple(margin_supported),
                        "confidence": "high" if margin_supported else "low",
                        "reconstruction_error": sum(margin_errors, Decimal(0)) / Decimal(len(margin_errors)) if margin_errors else Decimal(0),
                        "completeness": Decimal(len(margin_supported)) / Decimal(len(years)) if years else None,
                        "identity_warnings": tuple(dict.fromkeys(identity_warnings)),
                        "warnings": tuple(dict.fromkeys(warnings)),
                    }
                ),
                "gross_profit": company.diagnostics.gross_profit.model_copy(
                    update={
                        "coverage": Decimal(len(profit_supported)) / Decimal(len(years)) if years else None,
                        "supported_years": tuple(profit_supported),
                        "confidence": "high" if profit_supported else "low",
                        "reconstruction_error": sum(profit_errors, Decimal(0)) / Decimal(len(profit_errors)) if profit_errors else Decimal(0),
                        "completeness": Decimal(len(profit_supported)) / Decimal(len(years)) if years else None,
                        "identity_warnings": tuple(dict.fromkeys(identity_warnings)),
                        "warnings": tuple(dict.fromkeys(warnings)),
                    }
                ),
            }
        )
        years_output = tuple(
            item.model_copy(
                update={
                    "gross_profit": profits[index],
                    "gross_margin": margins[index],
                    "source": source_by_year[year],
                    "confidence": confidence_by_year[year],
                    "provenance": provenance_by_year.get(year),
                    "provenance_chain": provenance_chain_by_year.get(year, ()),
                    "source_provenance": source_references.get(year, ()),
                    "gross_margin_provenance": margin_provenance.get(year),
                    "gross_margin_provenance_chain": margin_chains.get(year, ()),
                    "gross_margin_source_provenance": margin_references.get(year, ()),
                    "gross_profit_provenance": profit_provenance.get(year),
                    "gross_profit_provenance_chain": profit_chains.get(year, ()),
                    "gross_profit_source_provenance": profit_references.get(year, ()),
                    "method": methods[year],
                    "audit": audits.get(year, ()),
                    "expected_gross_profit": (
                        company.consolidated_revenue[index] * margins[index] / Decimal(100)
                        if company.consolidated_revenue[index] is not None
                        and margins[index] is not None
                        else None
                    ),
                    "identity_error": (
                        abs(
                            profits[index]
                            - company.consolidated_revenue[index]
                            * margins[index]
                            / Decimal(100)
                        )
                        if profits[index] is not None
                        and company.consolidated_revenue[index] is not None
                        and margins[index] is not None
                        else None
                    ),
                }
            )
            for index, (item, year) in enumerate(zip(company.years, years, strict=True))
        )
        return company.model_copy(
            update={
                "consolidated_gross_profit": tuple(profits),
                "consolidated_gross_margin": tuple(margins),
                "source_by_year": source_by_year,
                "confidence_by_year": confidence_by_year,
                "provenance_by_year": provenance_by_year,
                "provenance_chain_by_year": provenance_chain_by_year,
                "source_provenance_by_year": source_references,
                "gross_margin_provenance_by_year": margin_provenance,
                "gross_margin_provenance_chain_by_year": margin_chains,
                "gross_margin_source_provenance_by_year": margin_references,
                "gross_profit_provenance_by_year": profit_provenance,
                "gross_profit_provenance_chain_by_year": profit_chains,
                "gross_profit_source_provenance_by_year": profit_references,
                "method_by_year": methods,
                "audit_by_year": audits,
                "diagnostics": diagnostics,
                "warnings": tuple(dict.fromkeys(warnings)),
                "years": years_output,
            }
        )

    def _consolidate(
        self,
        company_id: str,
        years: tuple[int, ...],
        revenue: tuple[Decimal | None, ...],
        all_segment_economics: Sequence[SegmentOperatingEconomicsForecast],
        selected: Sequence[SegmentOperatingEconomicsForecast],
        selection_warnings: Sequence[str],
        policy: OperatingEconomicsForecastConfig,
        *,
        fiscal_period: str,
        period_key: str | None,
    ) -> CompanyOperatingEconomicsForecast:
        profits: list[Decimal | None] = []
        margins: list[Decimal | None] = []
        sources: dict[int, str] = {}
        confidences: dict[int, str] = {}
        provenance: dict[int, Any] = {}
        provenance_chains: dict[int, tuple[Any, ...]] = {}
        references: dict[int, tuple[EvidenceReference, ...]] = {}
        margin_provenance: dict[int, Any] = {}
        margin_provenance_chains: dict[int, tuple[Any, ...]] = {}
        margin_references: dict[int, tuple[EvidenceReference, ...]] = {}
        profit_provenance: dict[int, Any] = {}
        profit_provenance_chains: dict[int, tuple[Any, ...]] = {}
        profit_references: dict[int, tuple[EvidenceReference, ...]] = {}
        methods: dict[int, str] = {}
        audits: dict[int, tuple[str, ...]] = {}
        warnings = list(selection_warnings)
        identity_warnings: list[str] = []
        margin_supported: list[int] = []
        profit_supported: list[int] = []
        margin_errors: list[Decimal] = []
        profit_errors: list[Decimal] = []
        if not selected:
            warnings.append("No non-overlapping segment scope is available for gross-economics consolidation")
        for item in selected:
            warnings.extend(
                f"{item.segment.segment_id}: {warning}" for warning in item.warnings
            )
            identity_warnings.extend(item.diagnostics.identity_warnings)
        for index, year in enumerate(years):
            records = [item.years[index] for item in selected]
            valid = bool(records) and all(item.gross_profit is not None for item in records)
            segment_revenues = [item.revenue for item in records]
            revenue_complete = all(value is not None for value in segment_revenues)
            selected_revenue = (
                sum(segment_revenues, Decimal(0))
                if revenue_complete
                else None
            )
            residual = (
                selected_revenue is not None
                and revenue[index] != selected_revenue
            )
            units = {item.unit for item in selected}
            currency_codes = {_currency_code(item) for item in units if _currency_code(item)}
            compatible_units = (
                bool(units)
                and len(currency_codes) == 1
                and all(_currency_code(item) is not None for item in units)
                and (
                    len(units) <= 1
                    or all(
                        operating_units_compatible(next(iter(units)), item)
                        for item in units
                    )
                )
            )
            audit: list[str] = []
            if valid and not compatible_units:
                valid = False
                warnings.append(f"FY{year}: gross-profit consolidation unavailable due to currency/unit mismatch")
            if valid:
                profit = sum((item.gross_profit for item in records), Decimal(0))
                total_revenue = selected_revenue
                margin = (
                    profit / total_revenue * Decimal(100)
                    if total_revenue is not None
                    and total_revenue != 0
                    and not residual
                    and all(item.gross_margin is not None for item in records)
                    else None
                )
                profits.append(profit)
                margins.append(margin)
                profit_supported.append(year)
                if margin is not None:
                    margin_supported.append(year)
                source = _combine_sources(tuple(item.source for item in records))
                confidence = _worst_confidence(tuple(item.confidence for item in records))
                refs = tuple(dict.fromkeys(ref for item in records for ref in item.source_provenance))
                if refs:
                    references[year] = refs
                margin_values = _dedupe_values(
                    value
                    for item in records
                    for value in item.gross_margin_provenance_chain
                )
                profit_values = _dedupe_values(
                    value
                    for item in records
                    for value in item.gross_profit_provenance_chain
                )
                if margin_values:
                    margin_provenance[year] = margin_values[0]
                    margin_provenance_chains[year] = margin_values
                if profit_values:
                    profit_provenance[year] = profit_values[0]
                    profit_provenance_chains[year] = profit_values
                margin_refs = _dedupe_refs(
                    ref
                    for item in records
                    for ref in item.gross_margin_source_provenance
                )
                profit_refs = _dedupe_refs(
                    ref
                    for item in records
                    for ref in item.gross_profit_source_provenance
                )
                if margin_refs:
                    margin_references[year] = margin_refs
                if profit_refs:
                    profit_references[year] = profit_refs
                all_provenance = _dedupe_values((*margin_values, *profit_values))
                if all_provenance:
                    provenance[year] = all_provenance[0]
                    provenance_chains[year] = all_provenance
                methods[year] = "sum_selected_non_overlapping_segment_gross_profit"
                if margin is not None:
                    expected = total_revenue * margin / Decimal(100)
                    error = abs(profit - expected)
                    profit_errors.append(error)
                    margin_errors.extend(
                        item.identity_error for item in records if item.identity_error is not None
                    )
                    audit.append(f"gross_profit_expected={expected}")
                    audit.append(f"gross_profit_identity_error={error}")
                elif residual:
                    warnings.append(
                        f"FY{year}: company gross margin unavailable; company revenue "
                        f"{revenue[index]} differs from selected segment revenue "
                        f"{selected_revenue} (uncovered residual)"
                    )
                sources[year] = source
                confidences[year] = confidence
                audits[year] = tuple(audit)
            else:
                profits.append(None)
                margins.append(None)
                refs = _dedupe_refs(
                    ref for item in records for ref in item.source_provenance
                )
                if refs:
                    references[year] = refs
                margin_values = _dedupe_values(
                    value
                    for item in records
                    for value in item.gross_margin_provenance_chain
                )
                profit_values = _dedupe_values(
                    value
                    for item in records
                    for value in item.gross_profit_provenance_chain
                )
                if margin_values:
                    margin_provenance[year] = margin_values[0]
                    margin_provenance_chains[year] = margin_values
                if profit_values:
                    profit_provenance[year] = profit_values[0]
                    profit_provenance_chains[year] = profit_values
                margin_refs = _dedupe_refs(
                    ref
                    for item in records
                    for ref in item.gross_margin_source_provenance
                )
                profit_refs = _dedupe_refs(
                    ref
                    for item in records
                    for ref in item.gross_profit_source_provenance
                )
                if margin_refs:
                    margin_references[year] = margin_refs
                if profit_refs:
                    profit_references[year] = profit_refs
                all_provenance = _dedupe_values((*margin_values, *profit_values))
                if all_provenance:
                    provenance[year] = all_provenance[0]
                    provenance_chains[year] = all_provenance
                sources[year] = _UNAVAILABLE
                confidences[year] = "low"
                methods[year] = "unavailable_incomplete_segment_gross_profit"
                warnings.append(
                    f"FY{year}: consolidated gross profit unavailable; every selected segment must have valid gross profit"
                )

        margin_attempted = any(item.diagnostics.gross_margin.coverage is not None for item in selected)
        profit_attempted = any(item.diagnostics.gross_profit.coverage is not None for item in selected)
        margin_diag = OperatingEconomicsMetricDiagnostics(
            metric=_MARGIN,
            coverage=Decimal(len(margin_supported)) / Decimal(len(years)) if margin_attempted and years else None,
            supported_years=tuple(margin_supported),
            confidence=_worst_confidence(tuple(confidences[year] for year in margin_supported)) if margin_supported else "low",
            reconstruction_error=sum(margin_errors, Decimal(0)) / Decimal(len(margin_errors)) if margin_errors else None,
            completeness=Decimal(len(margin_supported)) / Decimal(len(years)) if margin_attempted and years else None,
            warnings=tuple(dict.fromkeys(item for item in warnings if _MARGIN in item.casefold() or "gross profit" in item.casefold())),
            identity_warnings=tuple(identity_warnings),
        )
        profit_diag = OperatingEconomicsMetricDiagnostics(
            metric=_PROFIT,
            coverage=Decimal(len(profit_supported)) / Decimal(len(years)) if profit_attempted and years else None,
            supported_years=tuple(profit_supported),
            confidence=_worst_confidence(tuple(confidences[year] for year in profit_supported)) if profit_supported else "low",
            reconstruction_error=sum(profit_errors, Decimal(0)) / Decimal(len(profit_errors)) if profit_errors else None,
            completeness=Decimal(len(profit_supported)) / Decimal(len(years)) if profit_attempted and years else None,
            warnings=tuple(dict.fromkeys(item for item in warnings if "gross profit" in item.casefold())),
            identity_warnings=tuple(identity_warnings),
        )
        diagnostics = OperatingEconomicsDiagnostics(
            gross_margin=margin_diag,
            gross_profit=profit_diag,
            completeness=Decimal(len(set(margin_supported) & set(profit_supported))) / Decimal(len(years)) if years and (margin_attempted or profit_attempted) else None,
            identity_warnings=tuple(identity_warnings),
            warnings=tuple(dict.fromkeys(warnings)),
        )
        records = tuple(
            OperatingEconomicsYear(
                fiscal_year=year,
                fiscal_period=fiscal_period,
                period_key=period_key,
                revenue=revenue[index],
                gross_profit=profits[index],
                gross_margin=margins[index],
                source=sources[year],
                confidence=confidences[year],
                provenance=provenance.get(year),
                provenance_chain=provenance_chains.get(year, ()),
                source_provenance=references.get(year, ()),
                gross_margin_provenance=margin_provenance.get(year),
                gross_margin_provenance_chain=margin_provenance_chains.get(year, ()),
                gross_margin_source_provenance=margin_references.get(year, ()),
                gross_profit_provenance=profit_provenance.get(year),
                gross_profit_provenance_chain=profit_provenance_chains.get(year, ()),
                gross_profit_source_provenance=profit_references.get(year, ()),
                method=methods[year],
                audit=audits.get(year, ()),
                expected_gross_profit=(
                    revenue[index] * margins[index] / Decimal(100)
                    if revenue[index] is not None and margins[index] is not None
                    else None
                ),
                identity_error=(
                    abs(
                        profits[index]
                        - revenue[index] * margins[index] / Decimal(100)
                    )
                    if profits[index] is not None
                    and revenue[index] is not None
                    and margins[index] is not None
                    else None
                ),
            )
            for index, year in enumerate(years)
        )
        return CompanyOperatingEconomicsForecast(
            company_id=company_id,
            fiscal_years=years,
            segment_economics=tuple(all_segment_economics),
            consolidated_revenue=revenue,
            consolidated_gross_profit=tuple(profits),
            consolidated_gross_margin=tuple(margins),
            years=records,
            source_by_year=sources,
            confidence_by_year=confidences,
            provenance_by_year=provenance,
            provenance_chain_by_year=provenance_chains,
            source_provenance_by_year=references,
            gross_margin_provenance_by_year=margin_provenance,
            gross_margin_provenance_chain_by_year=margin_provenance_chains,
            gross_margin_source_provenance_by_year=margin_references,
            gross_profit_provenance_by_year=profit_provenance,
            gross_profit_provenance_chain_by_year=profit_provenance_chains,
            gross_profit_source_provenance_by_year=profit_references,
            method_by_year=methods,
            audit_by_year=audits,
            diagnostics=diagnostics,
            warnings=tuple(dict.fromkeys(warnings)),
            unit=next((item.unit for item in selected if item.unit), "currency"),
        )


def _metric_key(value: Any) -> str:
    normalized = str(getattr(value, "value", value)).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in _MARGIN_ALIASES:
        return _MARGIN
    if normalized in _PROFIT_ALIASES:
        return _PROFIT
    if normalized in _COST_ALIASES:
        return _COST
    if normalized in _REVENUE_ALIASES or normalized.endswith("_revenue") or normalized.endswith("_sales"):
        return _REVENUE
    if normalized in {
        "research_and_development",
        "research_and_development_expense",
        "research_development",
    }:
        return "r_and_d"
    if normalized in {
        "selling_general_and_administrative",
        "selling_general_and_administrative_expense",
    }:
        return "sg_and_a"
    if normalized in {"other_operating_item", "other_operating_income_expense"}:
        return "other_operating_items"
    if normalized in {"operating_income_loss"}:
        return "operating_income"
    return normalized


def _currency_code(unit: str) -> str | None:
    folded = unit.casefold()
    for code in ("usd", "eur", "gbp", "jpy", "cny", "cad", "aud", "chf"):
        if code in folded:
            return code
    return None


def _references(candidates: Iterable[_Candidate | OperatingDriverObservation]) -> tuple[EvidenceReference, ...]:
    return tuple(
        dict.fromkeys(
            ref
            for candidate in candidates
            for ref in (
                *getattr(candidate, "source_provenance", ()),
                getattr(candidate, "evidence", None),
                candidate.provenance
                if isinstance(getattr(candidate, "provenance", None), EvidenceReference)
                else None,
            )
            if ref is not None
        )
    )


def _dedupe_values(values: Iterable[Any]) -> tuple[Any, ...]:
    result: list[Any] = []
    for value in values:
        if value is not None and value not in result:
            result.append(value)
    return tuple(result)


def _dedupe_refs(values: Iterable[EvidenceReference]) -> tuple[EvidenceReference, ...]:
    result: list[EvidenceReference] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _candidate_provenance(
    candidate: _Candidate | None, revenue: _RevenueInput
) -> tuple[Any, tuple[Any, ...], tuple[EvidenceReference, ...]]:
    if candidate is None:
        return None, (), ()
    values = [candidate.provenance]
    for observation in candidate.observations:
        values.extend((observation.provenance, observation.evidence))
    refs = tuple(candidate.source_provenance)
    if candidate.consumes_revenue:
        values.append(revenue.provenance)
        refs = refs + tuple(ref for ref in revenue.source_provenance if ref not in refs)
    chain = _dedupe_values(values)
    return candidate.provenance or (chain[0] if chain else None), chain, refs


GrossEconomicsForecastService = OperatingEconomicsForecastService

__all__ = ["GrossEconomicsForecastService", "OperatingEconomicsForecastService"]
