"""Explicit driver-based FCFF forecasting.

This module is deliberately a composition seam, not another FCFF calculator.
Revenue and gross economics are built by the independent operating service;
reinvestment and the canonical FCFF representation are then supplied by the
existing operating stages.  The legacy consolidated FcffForecastService
future-fallback path, consensus, research, retrieval, and valuation code do
not belong here; deterministic per-metric normalized assumptions remain valid
operating inputs and are audited.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from typing import Any

from edgarito.schemas.forecasting import (
    DriverBasedFcffForecastResult,
    DriverBasedForecastReadiness,
    FcffForecast,
    FcffForecastDriver,
    FcffForecastMethod,
    FcffForecastParameters,
    ForecastAssumptionSource,
    ForecastPlan,
    ForecastSeedType,
)
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    NormalizedCompanyFinancials,
)
from edgarito.schemas.operating import (
    CompanyOperatingEconomicsForecast,
    CompanyOperatingForecast,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingEconomicsForecastConfig,
    OperatingSegment,
    operating_units_compatible,
)
from edgarito.schemas.operating_graph import (
    EconomicEvaluationResult,
    EconomicModel,
    EconomicObservation,
)
from edgarito.services.financials.availability import (
    FinancialObservationAvailabilityService,
    ObservationAvailabilityMode,
)
from edgarito.services.forecasting._fcff.audit import economic_identity_issues
from edgarito.services.forecasting._fcff.context import (
    ForecastContextBuild,
    build_forecast_context,
    future_date,
)
from edgarito.services.forecasting._fcff.paths import historical_fcff
from edgarito.services.forecasting.validation import ForecastValidationService
from edgarito.services.operating._forecast.financials_adapter import (
    normalized_company_financials_to_operating_observations,
)
from edgarito.services.operating._forecast.reinvestment import (
    DriverBasedCanonicalFcffAdapter,
)
from edgarito.services.operating._forecast.service import OperatingForecastService
from edgarito.services.operating._graph import adapt_economic_forecast, evaluate_graph

_PERCENT = Decimal(100)
_DRIVER_METRICS = (
    "revenue",
    "gross_profit",
    "ebit",
    "tax_rate",
    "nopat",
    "depreciation_and_amortization",
    "capital_expenditures",
    "operating_working_capital",
    "change_in_operating_working_capital",
    "fcff",
)


class DriverBasedFcffForecastService:
    """Build a complete FCFF forecast from explicit operating drivers.

    The service accepts either a structured discovery result or its individual
    provider-neutral parts.  Its return value is structured so callers retain
    the operating economics, readiness gate, and read-only validation result;
    :attr:`DriverBasedFcffForecastResult.forecast` is the canonical FCFF
    artifact consumed by existing DCF code.
    """

    _CORE_REQUIRED_CONCEPTS = frozenset(
        {
            FinancialConcept.REVENUE,
            FinancialConcept.OPERATING_INCOME,
            FinancialConcept.PRETAX_INCOME,
            FinancialConcept.INCOME_TAX_EXPENSE,
            FinancialConcept.DEPRECIATION_AND_AMORTIZATION,
            FinancialConcept.CAPITAL_EXPENDITURES,
        }
    )

    def __init__(
        self,
        operating_forecast_service: OperatingForecastService | None = None,
        canonical_adapter: DriverBasedCanonicalFcffAdapter | None = None,
        validation_service: ForecastValidationService | None = None,
        availability_service: FinancialObservationAvailabilityService | None = None,
        *,
        operating_service: OperatingForecastService | None = None,
        adapter: DriverBasedCanonicalFcffAdapter | None = None,
        validator: ForecastValidationService | None = None,
    ) -> None:
        self.operating_forecast_service = (
            operating_forecast_service
            or operating_service
            or OperatingForecastService()
        )
        self.canonical_adapter = canonical_adapter or adapter or DriverBasedCanonicalFcffAdapter()
        self.validation_service = validation_service or validator or ForecastValidationService()
        self.availability_service = availability_service or FinancialObservationAvailabilityService()
        self.last_result: DriverBasedFcffForecastResult | None = None

    @classmethod
    def required_concepts(cls) -> set[FinancialConcept]:
        """Return normalized facts needed only for historical context/evidence."""

        from edgarito.services.metrics.calculator import (
            OPERATING_WORKING_CAPITAL_CONCEPTS,
        )

        return set(cls._CORE_REQUIRED_CONCEPTS | OPERATING_WORKING_CAPITAL_CONCEPTS)

    def forecast(
        self,
        financials: NormalizedCompanyFinancials,
        parameters: FcffForecastParameters | None = None,
        *,
        plan: ForecastPlan | Mapping[str, Any] | None = None,
        forecast_plan: ForecastPlan | Mapping[str, Any] | None = None,
        overrides: Any = (),
        forecast_overrides: Any | None = None,
        evidence: Any = None,
        operating_evidence: Any = None,
        segments: Iterable[OperatingSegment] | None = None,
        definitions: Iterable[OperatingDriverDefinition] | None = None,
        observations: Iterable[OperatingDriverObservation] | None = None,
        management_constraints: Any = None,
        capex_constraints: Mapping[int, Any] | None = None,
        investment_programs: Any = None,
        programs: Any = None,
        config: OperatingEconomicsForecastConfig | Mapping[str, Any] | None = None,
        economics_config: OperatingEconomicsForecastConfig
        | Mapping[str, Any]
        | None = None,
        economic_model: EconomicModel | Mapping[str, Any] | None = None,
        economic_graph: EconomicModel | Mapping[str, Any] | None = None,
        graph_observations: Iterable[EconomicObservation | Mapping[str, Any]]
        | None = None,
        economic_observations: Iterable[EconomicObservation | Mapping[str, Any]]
        | None = None,
        economic_graph_observations: Iterable[EconomicObservation | Mapping[str, Any]]
        | None = None,
        compiled_graph_observations: Iterable[EconomicObservation | Mapping[str, Any]]
        | None = None,
        graph_evaluation: EconomicEvaluationResult | Mapping[str, Any] | None = None,
        economic_evaluation: EconomicEvaluationResult
        | Mapping[str, Any]
        | None = None,
        economic_graph_evaluation: EconomicEvaluationResult
        | Mapping[str, Any]
        | None = None,
        evaluation: EconomicEvaluationResult | Mapping[str, Any] | None = None,
        as_of: datetime.date | None = None,
        availability_mode: ObservationAvailabilityMode = ObservationAvailabilityMode.POINT_IN_TIME,
        company_id: str | None = None,
        ticker: str | None = None,
        **kwargs: Any,
    ) -> DriverBasedFcffForecastResult:
        del kwargs  # The explicit service intentionally has no open-ended side effects.
        parameters = parameters or FcffForecastParameters()
        availability_mode = ObservationAvailabilityMode(availability_mode)
        normalized_plan = self._plan(plan if plan is not None else forecast_plan)
        if normalized_plan is not None and normalized_plan.resolved != FcffForecastMethod.DRIVER_BASED:
            raise ValueError(
                "DriverBasedFcffForecastService requires a resolved driver_based plan"
            )
        source_evidence = operating_evidence if operating_evidence is not None else evidence
        normalized_overrides = (
            forecast_overrides if forecast_overrides is not None else overrides
        )
        graph_value = economic_model if economic_model is not None else economic_graph
        graph_model = self._coerce_graph_model(graph_value) if graph_value is not None else None
        graph_observation_values = self._graph_observation_items(
            graph_observations,
            economic_observations,
            economic_graph_observations,
            compiled_graph_observations,
        )
        graph_evaluation_value = next(
            (
                item
                for item in (
                    graph_evaluation,
                    economic_evaluation,
                    economic_graph_evaluation,
                    evaluation,
                )
                if item is not None
            ),
            None,
        )
        if hasattr(graph_evaluation_value, "compiled_observations"):
            graph_observation_values = (
                *graph_observation_values,
                *self._graph_observation_items(graph_evaluation_value),
            )

        try:
            context_build = self.build_context(
                financials,
                parameters,
                as_of=as_of,
                availability_mode=availability_mode,
            )
        except Exception as exc:
            readiness = DriverBasedForecastReadiness(
                target_years=(),
                seed_errors=(f"Unable to select a real canonical seed: {exc}",),
                diagnostics=("Historical FY/TTM/YTD context selection failed",),
            )
            raise self._incomplete(readiness, normalized_plan) from exc

        available = context_build.financials
        context = context_build.context
        first_year = self._first_forecast_year(context)
        target_years = tuple(first_year + index for index in range(parameters.forecast_years))
        values = self._evidence_values(source_evidence)
        explicit_segments = tuple(
            item
            for item in self._items(values.get("segments")) + self._items(segments)
            if self._available_evidence(item, as_of)
        )
        explicit_definitions = tuple(
            item
            for item in self._items(values.get("definitions")) + self._items(definitions)
            if self._available_evidence(item, as_of)
        )
        explicit_observations = self._items(
            values.get("observations")
            or values.get("eligible_records")
            or values.get("records")
        ) + self._items(observations)
        explicit_observations = tuple(
            item for item in explicit_observations if self._available_evidence(item, as_of)
        )
        normalized_observations = normalized_company_financials_to_operating_observations(
            available,
            availability_mode=availability_mode,
            availability_service=self.availability_service,
        )
        all_observations = (*normalized_observations, *explicit_observations)

        management = self._items(values.get("management_constraints")) + self._items(
            management_constraints
        )
        management = tuple(item for item in management if self._available_evidence(item, as_of))
        constraints = (
            capex_constraints
            if capex_constraints is not None
            else values.get("capex_constraints") or values.get("constraints")
            or parameters.capex_constraints
        )
        investment = (
            self._items(values.get("investment_programs"))
            + self._items(values.get("programs"))
            + self._items(investment_programs)
            + self._items(programs)
        )
        investment = tuple(
            item for item in self._items(investment) if self._available_evidence(item, as_of)
        )
        history = self._historical_revenue(
            context_build,
            values.get("historical_revenue"),
            first_year,
        )
        segment_values = tuple(explicit_segments)
        if not segment_values and self._has_explicit_company_revenue(explicit_observations):
            segment_values = (self._synthetic_company_segment(available),)

        template = self._template(
            available,
            context_build,
            parameters,
            target_years,
            company_id=company_id or available.company_id,
            ticker=ticker,
            availability_mode=availability_mode,
        )
        explicit_revenue_observations = ()
        if graph_model is None:
            explicit_revenue_observations = self._explicit_revenue_observations(
                normalized_plan,
                normalized_overrides,
                parameters,
                target_years,
                template.base_revenue,
                template.unit,
            )
        if explicit_revenue_observations:
            all_observations = (*all_observations, *explicit_revenue_observations)
            if not segment_values:
                segment_values = (self._synthetic_company_segment(available),)
        context_seed = self._context_seed(template, context_build)
        explicit_seed = values.get("reinvestment_seed")
        explicit_seed = explicit_seed if explicit_seed is not None else values.get("seed")

        economics: CompanyOperatingEconomicsForecast | None = None
        operating_forecast: CompanyOperatingForecast | None = None
        canonical: FcffForecast | None = None
        construction_errors: list[str] = []
        try:
            if graph_model is not None:
                operating_forecast = self._graph_operating_forecast(
                    graph_model,
                    graph_observation_values,
                    graph_evaluation_value,
                    target_years,
                    as_of=as_of,
                    company_id=company_id or available.company_id,
                )
                operating_forecast = self._mark_graph_revenue_source(operating_forecast)
                economics_service = getattr(
                    self.operating_forecast_service, "economics_service", None
                )
                if economics_service is None:
                    raise TypeError(
                        "OperatingForecastService does not expose economics_service"
                    )
                economics = economics_service.forecast(
                    segments=tuple(
                        item.segment for item in operating_forecast.segment_forecasts
                    ),
                    segment_revenue_forecasts=operating_forecast.segment_forecasts,
                    observations=all_observations,
                    management_constraints=management,
                    fiscal_years=target_years,
                    revenue_forecast=operating_forecast,
                    plan=normalized_plan,
                    forecast_plan=normalized_plan,
                    overrides=normalized_overrides,
                    forecast_overrides=normalized_overrides,
                    company_id=company_id or available.company_id,
                    config=economics_config or config,
                    investment_programs=investment,
                    capex_constraints=constraints,
                    seed=explicit_seed if explicit_seed is not None else context_seed,
                    reinvestment_seed=explicit_seed,
                )
                economics_by_id = {
                    item.segment.segment_id: item for item in economics.segment_economics
                }
                operating_forecast = operating_forecast.model_copy(
                    update={
                        "segment_forecasts": tuple(
                            item.model_copy(
                                update={
                                    "operating_economics": economics_by_id.get(
                                        item.segment.segment_id
                                    )
                                }
                            )
                            for item in operating_forecast.segment_forecasts
                        ),
                        "operating_economics": economics,
                    }
                )
            else:
                operating_forecast = self.operating_forecast_service.forecast(
                    segments=segment_values,
                    definitions=explicit_definitions,
                    observations=all_observations,
                    management_constraints=management,
                    historical_revenue=history,
                    fiscal_years=target_years,
                    company_id=company_id or available.company_id,
                    plan=normalized_plan,
                    forecast_plan=normalized_plan,
                    overrides=normalized_overrides,
                    forecast_overrides=normalized_overrides,
                    economics_config=economics_config or config,
                    investment_programs=investment,
                    capex_constraints=constraints,
                    seed=explicit_seed if explicit_seed is not None else context_seed,
                    reinvestment_seed=explicit_seed,
                )
            economics = getattr(operating_forecast, "operating_economics", None)
            if economics is None:
                construction_errors.append(
                    "OperatingForecastService did not produce complete operating economics"
                )
            else:
                fiscal_year_end = context.fiscal_year_end or context.latest_annual.period_end
                canonical = self.canonical_adapter.adapt(
                    economics,
                    template,
                    fiscal_year_end=fiscal_year_end,
                )
                if not isinstance(canonical, FcffForecast):
                    canonical = FcffForecast.model_validate(canonical)
                canonical = self._attach_operating_audit(canonical, operating_forecast)
        except Exception as exc:
            construction_errors.append(str(exc))

        readiness = self._readiness(
            template,
            canonical,
            economics,
            target_years,
            context_build,
            construction_errors,
        )
        evidence_warnings = tuple(
            str(item).strip()
            for item in (values.get("warnings") or ())
            if str(item).strip()
        )
        if evidence_warnings:
            readiness = readiness.model_copy(
                update={
                    "warnings": tuple(
                        dict.fromkeys((*readiness.warnings, *evidence_warnings))
                    )
                }
            )
        if not readiness.ready:
            raise self._incomplete(readiness, normalized_plan)

        # Validation is deliberately after canonical construction and receives
        # an independent read-only adapter.  It can report findings without
        # becoming a second activation gate.
        validation = self.validation_service.validate(canonical)
        validation_summary = validation.deterministic_dict()
        validation_warnings = tuple(
            f"{finding.code}: {finding.message}" for finding in validation.findings
        )
        warnings = tuple(
            dict.fromkeys(
                (
                    *readiness.warnings,
                    *getattr(operating_forecast, "warnings", ()),
                    *getattr(economics, "warnings", ()),
                    *validation_warnings,
                )
            )
        )
        audit = tuple(
            dict.fromkeys(
                (
                    "method=driver_based_fcff",
                    "operating_forecast_service=OperatingForecastService",
                    "canonical_adapter=DriverBasedCanonicalFcffAdapter",
                    "forecast_validation=read_only",
                    *(
                        ("revenue_source=economic_graph",)
                        if graph_model is not None
                        else ()
                    ),
                    "validation_findings="
                    + str(validation_summary.get("counts", {}).get("total", 0)),
                    "validation_errors="
                    + str(validation_summary.get("counts", {}).get("error", 0)),
                    *getattr(source_evidence, "audit_records", ()),
                    *(values.get("audit_records") or ()),
                )
            )
        )
        result = DriverBasedFcffForecastResult(
            forecast=canonical,
            company_operating_forecast=operating_forecast,
            company_operating_economics=economics,
            readiness=readiness,
            validation=validation,
            validation_summary=validation_summary,
            warnings=warnings,
            audit=audit,
        )
        self.last_result = result
        return result

    def forecast_canonical(self, *args: Any, **kwargs: Any) -> FcffForecast:
        """Convenience API matching the legacy FCFF service return shape."""

        return self.forecast(*args, **kwargs).forecast

    build = forecast
    run = forecast
    compose = forecast
    create = forecast
    forecast_result = forecast

    def build_context(
        self,
        financials: NormalizedCompanyFinancials,
        parameters: FcffForecastParameters | None = None,
        *,
        as_of: datetime.date | None = None,
        availability_mode: ObservationAvailabilityMode = ObservationAvailabilityMode.POINT_IN_TIME,
    ) -> ForecastContextBuild:
        """Expose the shared historical seed selection without future paths."""

        return build_forecast_context(
            financials,
            parameters or FcffForecastParameters(),
            as_of=as_of,
            availability_mode=availability_mode,
            availability_service=self.availability_service,
            core_required_concepts=self._CORE_REQUIRED_CONCEPTS,
        )

    @staticmethod
    def _coerce_graph_model(value: Any) -> EconomicModel:
        """Normalize a graph without mutating the caller's frozen model."""

        if hasattr(value, "economic_model") and not isinstance(value, Mapping):
            value = value.economic_model
        if hasattr(value, "model") and not isinstance(value, (EconomicModel, Mapping)):
            value = value.model
        return value if isinstance(value, EconomicModel) else EconomicModel.model_validate(value)

    @staticmethod
    def _mark_graph_revenue_source(
        forecast: CompanyOperatingForecast,
    ) -> CompanyOperatingForecast:
        """Make the opt-in source explicit without changing legacy adapter output."""

        return forecast.model_copy(
            update={
                "source_by_year": {
                    year: "economic_graph" for year in forecast.fiscal_years
                },
                "segment_forecasts": tuple(
                    item.model_copy(
                        update={
                            "source_by_year": {
                                year: "economic_graph"
                                for year in item.fiscal_years
                            }
                        }
                    )
                    for item in forecast.segment_forecasts
                ),
            }
        )

    @classmethod
    def _graph_observation_items(cls, *values: Any) -> tuple[EconomicObservation, ...]:
        result: list[EconomicObservation] = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, Mapping) and "node_id" not in value and "node" not in value:
                value = value.get("observations", value.get("compiled_observations", value))
            for item in cls._items(value):
                if hasattr(item, "compiled_observations"):
                    item = item.compiled_observations
                    for nested in cls._items(item):
                        result.append(
                            nested
                            if isinstance(nested, EconomicObservation)
                            else EconomicObservation.model_validate(nested)
                        )
                    continue
                result.append(
                    item
                    if isinstance(item, EconomicObservation)
                    else EconomicObservation.model_validate(item)
                )
        return tuple(result)

    @staticmethod
    def _coerce_graph_evaluation(value: Any) -> EconomicEvaluationResult:
        if hasattr(value, "evaluation") and not isinstance(
            value, (EconomicEvaluationResult, Mapping)
        ):
            value = value.evaluation
        return (
            value
            if isinstance(value, EconomicEvaluationResult)
            else EconomicEvaluationResult.model_validate(value)
        )

    @classmethod
    def _graph_operating_forecast(
        cls,
        model: EconomicModel,
        observations: Sequence[EconomicObservation],
        supplied_evaluation: Any,
        target_years: tuple[int, ...],
        *,
        as_of: datetime.date | None,
        company_id: str,
    ) -> CompanyOperatingForecast:
        """Evaluate and adapt graph revenue, failing closed on missing leaves."""

        evaluated_model = model.model_copy(
            update={"observations": (*model.observations, *observations)}
        )
        expected = evaluate_graph(
            evaluated_model,
            target_years,
            as_of=as_of,
            fiscal_period=evaluated_model.fiscal_period,
        )
        if supplied_evaluation is not None:
            supplied = cls._coerce_graph_evaluation(supplied_evaluation)
            if supplied.target_years != target_years:
                raise ValueError(
                    "Economic graph evaluation years must exactly match driver forecast years"
                )
            if supplied.as_of != expected.as_of:
                raise ValueError(
                    "Economic graph evaluation as_of must exactly match driver forecast as_of"
                )
            if supplied != expected:
                raise ValueError(
                    "Economic graph evaluation must be the deterministic evaluation "
                    "of the compiled economic model"
                )
        try:
            adapted = adapt_economic_forecast(
                evaluated_model,
                expected,
                company_id=company_id,
            )
        except Exception as exc:
            unresolved = tuple(
                dict.fromkeys(
                    f"{item.node_id} FY{item.fiscal_year}: {item.reason}"
                    for item in expected.unresolved_leaf_requirements
                )
            )
            detail = "; ".join(unresolved[:5])
            if detail:
                raise ValueError(f"Economic graph unresolved leaves: {detail}") from exc
            raise
        return adapted.company_forecast

    @staticmethod
    def _plan(value: Any) -> ForecastPlan | None:
        if value is None:
            return None
        return value if isinstance(value, ForecastPlan) else ForecastPlan.model_validate(value)

    @staticmethod
    def _evidence_values(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        names = (
            "segments",
            "definitions",
            "observations",
            "records",
            "eligible_records",
            "management_constraints",
            "capex_constraints",
            "constraints",
            "investment_programs",
            "programs",
            "reinvestment_seed",
            "seed",
            "historical_revenue",
            "warnings",
            "audit_records",
        )
        result = {name: getattr(value, name) for name in names if hasattr(value, name)}
        if "observations" not in result:
            result["observations"] = getattr(value, "eligible_records", ())
        return result

    @staticmethod
    def _items(value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, (OperatingSegment, OperatingDriverDefinition, OperatingDriverObservation, Mapping)):
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

    @classmethod
    def _available_evidence(cls, value: Any, as_of: datetime.date | None) -> bool:
        if as_of is None:
            return True
        if isinstance(value, Mapping):
            date = value.get("filing_date") or value.get("as_of")
            reference = value.get("evidence") or value.get("provenance")
        else:
            date = getattr(value, "filing_date", None) or getattr(value, "as_of", None)
            reference = getattr(value, "evidence", None) or getattr(value, "provenance", None)
        date = date or getattr(reference, "filing_date", None)
        return date is None or date <= as_of

    @staticmethod
    def _first_forecast_year(context) -> int:
        return context.current_fiscal_year or context.base.fiscal_year + 1

    @staticmethod
    def _historical_revenue(
        context_build: ForecastContextBuild,
        supplied: Any,
        first_year: int,
    ) -> dict[int, Decimal]:
        result = {
            item.fiscal_year: item.revenue
            for item in context_build.annual_periods
            if item.fiscal_year < first_year
        }
        if isinstance(supplied, Mapping):
            result.update(
                {
                    int(year): Decimal(str(value))
                    for year, value in supplied.items()
                    if int(year) < first_year and Decimal(str(value)) > 0
                }
            )
        return result

    @staticmethod
    def _has_explicit_company_revenue(observations: Sequence[Any]) -> bool:
        for item in observations:
            segment_id = getattr(item, "segment_id", None)
            scope = str(getattr(item, "scope", "") or "").casefold()
            driver = str(getattr(getattr(item, "driver_id", None), "value", getattr(item, "driver_id", ""))).casefold()
            if (
                driver in {"revenue", "segment_revenue"}
                and (segment_id == "company" or scope in {"company", "consolidated", "total"})
            ):
                return True
        return False

    @staticmethod
    def _synthetic_company_segment(financials: NormalizedCompanyFinancials) -> OperatingSegment:
        unit = None
        for item in financials.observations:
            if item.concept == FinancialConcept.REVENUE:
                unit = item.unit
                break
        return OperatingSegment(
            segment_id="company",
            # The literal company scope is a schema-level alias.  Do not use
            # the issuer name here: identity canonicalization would turn this
            # synthetic consolidated scope into an issuer-specific segment.
            name="Company",
            scope="consolidated",
            currency=unit if unit and len(unit) == 3 else None,
            source="explicit_operating_evidence",
            confidence="medium",
        )

    @staticmethod
    def _explicit_revenue_observations(
        plan: ForecastPlan | None,
        overrides: Any,
        parameters: FcffForecastParameters,
        years: tuple[int, ...],
        base_revenue: Decimal,
        unit: str,
    ) -> tuple[OperatingDriverObservation, ...]:
        """Materialize explicit company revenue paths for company-only callers."""

        records: list[Any] = []
        if plan is not None:
            records.extend(plan.decisions)
            records.extend(plan.overrides)
        if isinstance(overrides, Mapping):
            records.extend(overrides.values())
        elif isinstance(overrides, (str, bytes)):
            pass
        else:
            try:
                records.extend(overrides or ())
            except TypeError:
                records.append(overrides)
        selected = None
        for record in records:
            metric = str(
                getattr(
                    getattr(record, "metric", None),
                    "value",
                    getattr(record, "metric", ""),
                )
            ).strip().casefold().replace("-", "_").replace(" ", "_")
            scope = getattr(getattr(record, "scope", None), "value", getattr(record, "scope", ""))
            if str(scope).casefold() != "company":
                continue
            if metric in {"revenue", "segment_revenue"}:
                path = getattr(record, "explicit_path", None)
                if path is not None:
                    selected = ("revenue", tuple(Decimal(item) for item in path))
            elif metric in {"revenue_growth", "growth"}:
                path = getattr(record, "explicit_path", None)
                if path is not None:
                    selected = ("growth", tuple(Decimal(item) for item in path))
        if selected is None and parameters.revenue_growth is not None:
            # ``FcffForecastParameters.revenue_growth`` is an explicit driver
            # input when supplied to this service, not an inferred normalized
            # path.  It is used only when no company revenue path was supplied.
            selected = ("growth", parameters.revenue_growth)
        if selected is None:
            return ()
        kind, raw_path = selected
        path = raw_path if len(raw_path) == len(years) else raw_path * len(years)
        if len(path) != len(years):
            return ()
        revenues: list[Decimal] = []
        previous = base_revenue
        for value in path:
            if kind == "revenue":
                revenue = value
            else:
                if value <= Decimal(-100):
                    return ()
                revenue = previous * (Decimal(1) + value / _PERCENT)
            if not revenue.is_finite() or revenue <= 0:
                return ()
            revenues.append(revenue)
            previous = revenue
        return tuple(
            OperatingDriverObservation(
                segment_id="company",
                driver_id="revenue",
                fiscal_year=year,
                value=revenue,
                unit=unit,
                scope="company",
                scope_evidence="explicit driver-based revenue path",
                is_total=True,
                origin="first_party_observation",
                confidence="high",
                method="forecast_plan_explicit_revenue" if kind == "revenue" else "forecast_plan_explicit_revenue_growth",
            )
            for year, revenue in zip(years, revenues, strict=True)
        )

    def _template(
        self,
        financials: NormalizedCompanyFinancials,
        build: ForecastContextBuild,
        parameters: FcffForecastParameters,
        target_years: tuple[int, ...],
        *,
        company_id: str,
        ticker: str | None,
        availability_mode: ObservationAvailabilityMode,
    ) -> FcffForecast:
        context = build.context
        base = context.latest_annual if context.seed_type == ForecastSeedType.YTD_PLUS_FORECAST else context.base
        base_tax_rate = self._tax_rate(base)
        seed_errors = () if base_tax_rate is not None else ("Selected canonical base has no valid effective tax rate",)
        del seed_errors  # Readiness derives the blocking seed check from this value.
        base_tax_rate = base_tax_rate if base_tax_rate is not None else Decimal(0)
        previous = (
            build.annual_periods[-2]
            if context.seed_type == ForecastSeedType.FISCAL_YEAR
            and len(build.annual_periods) > 1
            and base.fiscal_year == build.annual_periods[-1].fiscal_year
            else None
        )
        base_nopat = base.operating_income * (Decimal(1) - base_tax_rate / _PERCENT)
        base_period_end = context.seed_period_end
        ytd_anchor = None
        if context.seed_type == ForecastSeedType.YTD_PLUS_FORECAST and context.actual_ytd is not None:
            actual = context.actual_ytd
            actual_rate = self._tax_rate(actual)
            fallback_rate = actual_rate if actual_rate is not None else base_tax_rate
            ytd_anchor = self._ytd_anchor(context, actual, fallback_rate)
            base_period_end = actual.period_end
        assumption_sources = {
            driver: ForecastAssumptionSource.DRIVER_BASED for driver in FcffForecastDriver
        }
        return FcffForecast(
            provider=financials.provider,
            company_id=company_id,
            company_name=financials.company_name,
            ticker=ticker or financials.ticker,
            identifiers=financials.identifiers,
            method="driver_based_fcff",
            seed_type=context.seed_type,
            seed_methodology=context.seed_methodology,
            seed_period_end=context.seed_period_end,
            fiscal_year_end=(
                context.fiscal_year_end
                or future_date(context.latest_annual.period_end, 1)
            ),
            current_fiscal_year=context.current_fiscal_year,
            actual_quarters=context.actual_quarters,
            financial_snapshot_retrieved_at=financials.retrieved_at,
            availability_mode=availability_mode.value,
            base_fiscal_year=base.fiscal_year,
            base_period_end=base_period_end,
            base_revenue=base.revenue,
            base_operating_income=base.operating_income,
            base_tax_rate=base_tax_rate,
            base_nopat=base_nopat,
            base_depreciation_and_amortization=base.depreciation_and_amortization,
            base_capital_expenditures=base.capital_expenditures,
            base_operating_working_capital=base.operating_working_capital,
            base_fcff=historical_fcff(base, previous, base_nopat),
            unit=base.unit,
            parameters=parameters,
            historical_fiscal_years=tuple(
                item.fiscal_year for item in build.annual_periods[-parameters.historical_window :]
            ),
            assumption_sources=assumption_sources,
            observations=[],
            warnings=(),
            ytd_anchor=ytd_anchor,
            dcf_stub=None,
        )

    @staticmethod
    def _tax_rate(period: Any) -> Decimal | None:
        return _effective_tax_rate(period)

    @staticmethod
    def _ytd_anchor(context, actual, fallback_rate: Decimal):
        revenue = actual.revenue

        def ratio(value: Decimal) -> Decimal:
            return value / revenue * _PERCENT if revenue else Decimal(0)
        from edgarito.schemas.forecasting import FcffForecastYtdAnchor

        return FcffForecastYtdAnchor(
            fiscal_year=context.current_fiscal_year or actual.fiscal_year,
            ytd_period_end=actual.period_end,
            fiscal_year_end=context.fiscal_year_end or actual.period_end,
            actual_quarters=context.actual_quarters,
            actual_revenue=actual.revenue,
            actual_operating_income=actual.operating_income,
            actual_pretax_income=actual.pretax_income,
            actual_income_tax_expense=actual.income_tax_expense,
            actual_tax_rate=DriverBasedFcffForecastService._tax_rate(actual),
            actual_depreciation_and_amortization=actual.depreciation_and_amortization,
            actual_capital_expenditures=actual.capital_expenditures,
            actual_operating_working_capital=actual.operating_working_capital,
            latest_annual_revenue=context.latest_annual.revenue,
            revenue_growth=Decimal(0),
            operating_margin=ratio(actual.operating_income),
            tax_rate=fallback_rate,
            depreciation_to_revenue=ratio(actual.depreciation_and_amortization),
            capex_to_revenue=ratio(actual.capital_expenditures),
            operating_working_capital_to_revenue=ratio(actual.operating_working_capital),
        )

    @staticmethod
    def _context_seed(template: FcffForecast, build: ForecastContextBuild):
        context = build.context
        if context.seed_type == ForecastSeedType.YTD_RUN_RATE:
            # A run-rate balance is not a truthful full-year prior-period
            # balance.  Leave the seed absent and let readiness fail closed.
            return None
        return template

    def _readiness(
        self,
        template: FcffForecast,
        canonical: FcffForecast | None,
        economics: CompanyOperatingEconomicsForecast | None,
        target_years: tuple[int, ...],
        build: ForecastContextBuild,
        construction_errors: Sequence[str],
    ) -> DriverBasedForecastReadiness:
        missing: dict[int, list[str]] = {year: [] for year in target_years}
        observations = {
            item.fiscal_year: item for item in (canonical.observations if canonical else ())
        }
        paths = {
            "revenue": getattr(economics, "consolidated_revenue", ()),
            "gross_profit": getattr(economics, "consolidated_gross_profit", ()),
            "ebit": getattr(economics, "consolidated_ebit", ()),
            "tax_rate": getattr(economics, "tax_rate", ()),
            "nopat": getattr(economics, "nopat", ()),
            "depreciation_and_amortization": getattr(economics, "depreciation_and_amortization", ()),
            "capital_expenditures": getattr(economics, "capital_expenditures", ()),
            "operating_working_capital": getattr(economics, "operating_working_capital", ()),
            "change_in_operating_working_capital": getattr(economics, "change_in_operating_working_capital", ()),
            "fcff": getattr(economics, "fcff", ()),
        }
        for index, year in enumerate(target_years):
            row = observations.get(year)
            for metric, path in paths.items():
                value = path[index] if index < len(path) else None
                if row is not None and metric not in {"gross_profit", "ebit"}:
                    value = getattr(row, self._canonical_field(metric), value)
                if value is None:
                    missing[year].append(metric)
            if row is None:
                missing[year].extend(_DRIVER_METRICS)
        missing = {year: tuple(dict.fromkeys(values)) for year, values in missing.items() if values}
        missing_labels = tuple(
            f"FY{year} {metric}" for year, metrics in missing.items() for metric in metrics
        )
        identity = list(economic_identity_issues(canonical)) if canonical is not None else []
        if economics is not None:
            for index, _year in enumerate(target_years):
                values = {
                    metric: path[index] if index < len(path) else None
                    for metric, path in paths.items()
                }
                gross_profit = values.get("gross_profit")
                ebit = values.get("ebit")
                gross_margin = (
                    economics.consolidated_gross_margin[index]
                    if index < len(economics.consolidated_gross_margin)
                    else None
                )
                if (
                    gross_profit is not None
                    and ebit is not None
                    and gross_margin is not None
                    and economics.consolidated_r_and_d[index] is not None
                    and economics.consolidated_sg_and_a[index] is not None
                    and economics.consolidated_other_operating_items[index] is not None
                ):
                    expected_ebit = (
                        gross_profit
                        - economics.consolidated_r_and_d[index]
                        - economics.consolidated_sg_and_a[index]
                        + economics.consolidated_other_operating_items[index]
                    )
                    if ebit != expected_ebit:
                        identity.append(
                            f"FY{target_years[index]} EBIT identity: expected {expected_ebit}, got {ebit}"
                        )
        unit_errors: list[str] = []
        sequence_errors: list[str] = []
        canonical_errors: list[str] = []
        for error in construction_errors:
            folded = error.casefold()
            if "unit" in folded or "currenc" in folded:
                unit_errors.append(error)
            elif any(token in folded for token in ("horizon", "fiscal year", "sequence")):
                sequence_errors.append(error)
            else:
                canonical_errors.append(error)
        if canonical is not None and economics is not None:
            if not operating_units_compatible(canonical.unit, economics.unit):
                unit_errors.append(
                    f"Canonical forecast unit {canonical.unit!r} is incompatible with operating economics unit {economics.unit!r}"
                )
            for item in canonical.observations:
                if not operating_units_compatible(canonical.unit, item.unit):
                    unit_errors.append(f"FY{item.fiscal_year} canonical unit {item.unit!r} is incompatible")
        if canonical is not None:
            actual_years = tuple(item.fiscal_year for item in canonical.observations)
            if actual_years != target_years:
                sequence_errors.append(
                    f"Canonical fiscal years {actual_years} do not match target years {target_years}"
                )
            forecast_years = tuple(item.forecast_year for item in canonical.observations)
            if forecast_years != tuple(range(1, len(target_years) + 1)):
                sequence_errors.append("Canonical forecast_year sequence must start at 1 and be contiguous")
        seed_errors: list[str] = []
        context = build.context
        if context.seed_type == ForecastSeedType.YTD_RUN_RATE:
            seed_errors.append("YTD run-rate context has no unambiguous full-year OWC seed")
        if template.base_revenue <= 0:
            seed_errors.append("Canonical seed revenue must be positive and real")
        if template.base_period_end is None:
            seed_errors.append("Canonical seed period end is missing")
        if template.base_tax_rate == 0 and self._tax_rate(
            context.latest_annual if context.seed_type == ForecastSeedType.YTD_PLUS_FORECAST else context.base
        ) is None:
            seed_errors.append("Canonical base effective tax rate is unavailable")
        if economics is not None and economics.reinvestment_seed is None:
            seed_errors.append("Operating economics did not receive a real OWC seed")
        if canonical is None:
            canonical_errors.append("Driver-based canonical FCFF mapping produced no forecast")
        warnings: list[str] = []
        diagnostics = list(
            getattr(getattr(economics, "diagnostics", None), "warnings", ()) or ()
        )
        automatic_assumption_notes = self._automatic_assumption_notes(
            economics, target_years
        )
        diagnostics.extend(automatic_assumption_notes)
        warnings.extend(automatic_assumption_notes)
        if economics is not None and getattr(economics, "confidence", "low") == "low":
            warnings.append("Driver-based operating economics are low confidence")
        if canonical is not None:
            for item in canonical.observations:
                if abs(item.revenue_growth) > Decimal(30):
                    warnings.append(
                        f"FY{item.fiscal_year} revenue growth is aggressive ({item.revenue_growth}%)"
                    )
                if any(
                    audit.confidence == "low" for audit in item.cell_audits.values()
                ):
                    warnings.append(f"FY{item.fiscal_year} contains low-confidence driver cells")
        return DriverBasedForecastReadiness(
            target_years=target_years,
            missing_metrics_by_year=missing,
            missing_metric_years=missing_labels,
            identity_errors=tuple(dict.fromkeys(identity)),
            unit_errors=tuple(dict.fromkeys(unit_errors)),
            sequence_errors=tuple(dict.fromkeys(sequence_errors)),
            seed_errors=tuple(dict.fromkeys(seed_errors)),
            canonical_errors=tuple(dict.fromkeys(canonical_errors)),
            diagnostics=tuple(dict.fromkeys(diagnostics)),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _automatic_assumption_notes(
        economics: CompanyOperatingEconomicsForecast | None,
        target_years: tuple[int, ...],
    ) -> tuple[str, ...]:
        """Audit valid normalized historical assumptions without rejecting them."""

        if economics is None:
            return ()
        notes: list[str] = []
        maps = (
            ("gross economics", economics.source_by_year, economics.method_by_year),
            (
                "R&D",
                economics.r_and_d_source_by_year,
                economics.r_and_d_method_by_year,
            ),
            (
                "SG&A",
                economics.sg_and_a_source_by_year,
                economics.sg_and_a_method_by_year,
            ),
            (
                "other operating items",
                economics.other_operating_items_source_by_year,
                economics.other_operating_items_method_by_year,
            ),
            ("EBIT", economics.ebit_source_by_year, economics.ebit_method_by_year),
            (
                "tax",
                economics.tax_rate_source_by_year,
                economics.tax_rate_method_by_year,
            ),
            (
                "reinvestment",
                economics.depreciation_and_amortization_source_by_year,
                economics.depreciation_and_amortization_method_by_year,
            ),
            (
                "CAPEX",
                economics.capital_expenditures_source_by_year,
                economics.capital_expenditures_method_by_year,
            ),
            (
                "OWC",
                economics.operating_working_capital_source_by_year,
                economics.operating_working_capital_method_by_year,
            ),
            (
                "delta OWC",
                economics.change_in_operating_working_capital_source_by_year,
                economics.change_in_operating_working_capital_method_by_year,
            ),
            ("FCFF", economics.fcff_source_by_year, economics.fcff_method_by_year),
        )
        for label, sources, methods in maps:
            for year in target_years:
                source = str(sources.get(year, ""))
                method = str(methods.get(year, ""))
                if source == "normalized_historical" or "historical" in method.casefold():
                    notes.append(
                        f"FY{year} {label} uses normalized historical assumption: {method or source}"
                    )
                elif source == "mixed" or "mixed" in method.casefold():
                    notes.append(
                        f"FY{year} {label} uses mixed operating components: {method or source}"
                    )
        for segment in economics.segment_economics:
            for year_record in segment.years:
                if year_record.fiscal_year not in target_years:
                    continue
                method = year_record.method
                if "historical" in method.casefold() or "mixed" in method.casefold():
                    notes.append(
                        f"FY{year_record.fiscal_year} segment {segment.segment.segment_id} "
                        f"gross economics uses automatic component method: {method}"
                    )
        return tuple(dict.fromkeys(notes))

    @staticmethod
    def _attach_operating_audit(
        canonical: FcffForecast,
        operating: CompanyOperatingForecast,
    ) -> FcffForecast:
        """Carry independent revenue source maps without reconciliation fields."""

        return canonical.model_copy(
            update={
                "operating_driver_coverage": operating.driver_coverage,
                "operating_reconstruction_error": operating.reconstruction_error,
                "operating_confidence": operating.confidence,
                "operating_own_supported_years": operating.own_supported_years,
                "operating_consensus_years": (),
                "operating_divergence_by_year": {},
                "operating_divergence": None,
                "operating_transition_start_year": operating.transition_start_year,
                "operating_warnings": operating.warnings,
                "operating_selected_revenue_by_year": {
                    year: value
                    for year, value in zip(
                        operating.fiscal_years,
                        operating.consolidated_revenue,
                        strict=True,
                    )
                    if value is not None
                },
                "operating_source_by_year": operating.source_by_year,
                "operating_confidence_by_year": operating.confidence_by_year,
                "warnings": tuple(
                    dict.fromkeys((*canonical.warnings, *operating.warnings))
                ),
            }
        )

    @staticmethod
    def _canonical_field(metric: str) -> str:
        return {
            "depreciation_and_amortization": "depreciation_and_amortization",
            "capital_expenditures": "capital_expenditures",
            "operating_working_capital": "operating_working_capital",
            "change_in_operating_working_capital": "change_in_operating_working_capital",
            "tax_rate": "tax_rate",
            "nopat": "nopat",
            "fcff": "fcff",
            "revenue": "revenue",
        }.get(metric, metric)

    @staticmethod
    def _incomplete(readiness: DriverBasedForecastReadiness, plan: ForecastPlan | None):
        from edgarito.services.forecasting.orchestration import (
            DriverBasedForecastIncompleteError,
        )

        return DriverBasedForecastIncompleteError(readiness=readiness, plan=plan)


def _effective_tax_rate(period: Any) -> Decimal | None:
    from edgarito.services.financials.effective_tax import calculate_effective_tax_rate

    return calculate_effective_tax_rate(period.pretax_income, period.income_tax_expense)


__all__ = [
    "DriverBasedFcffForecastService",
    "DriverBasedFcffForecastResult",
    "DriverBasedForecastReadiness",
]
