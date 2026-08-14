"""Provider-neutral composition of operating forecasting and reconciliation."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, Mapping

import edgarito.services.operating.contracts as _contracts
from edgarito.schemas.forecasting import (
    AdaptiveMultistagePlan,
    FcffForecast,
    FcffForecastParameters,
    ForecastAssumptionSource,
    ForwardGrowthEvidence,
)
from edgarito.schemas.forward import ForwardRevenueEstimate
from edgarito.schemas.operating import (
    CompanyOperatingForecast,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingSegment,
)
from edgarito.services.financials.availability import (
    ObservationAvailabilityMode,
)
from edgarito.services.forecasting._fcff.service import FcffForecastService
from edgarito.services.forecasting.multistage import (
    AdaptiveMultistageFcffForecastService,
)
from edgarito.services.operating._forecast.service import (
    OperatingForecastService,
    normalize_company_historical_revenue,
)
from edgarito.services.operating.reconciliation import (
    RevenueForecastReconciler,
    materialize_revenue_anchors,
)


class OperatingForecastIntegrationService:
    """Compose deterministic operating evidence with revenue reconciliation.

    This seam deliberately accepts normalized contracts only.  It performs no
    discovery and has no provider dependencies.
    """

    def __init__(
        self,
        forecast_service: OperatingForecastService | None = None,
        reconciler: RevenueForecastReconciler | None = None,
    ) -> None:
        self.forecast_service = forecast_service or OperatingForecastService()
        self.reconciler = reconciler or RevenueForecastReconciler()

    def integrate(
        self,
        segments: Iterable[OperatingSegment],
        definitions: Iterable[OperatingDriverDefinition],
        observations: Iterable[OperatingDriverObservation] = (),
        management_constraints: Any = (),
        historical_revenue: Mapping[Any, Any] | None = None,
        consensus_estimates: Iterable[ForwardRevenueEstimate]
        | Mapping[Any, Any]
        | Any = (),
        explicit_anchors: Mapping[Any, Any] | Iterable[Any] | Any | None = None,
        management_anchors: Mapping[Any, Any] | Iterable[Any] | Any | None = None,
        fiscal_years: Iterable[int] = (),
        parameters: FcffForecastParameters | None = None,
        *,
        fcff_parameters: FcffForecastParameters | None = None,
        company_id: str = "company",
    ) -> _contracts.OperatingForecastIntegrationResult:
        if parameters is not None and fcff_parameters is not None:
            raise ValueError("Pass either parameters or fcff_parameters, not both")
        parameters = fcff_parameters or parameters
        if parameters is None:
            raise TypeError("fcff_parameters is required")
        # Discovery may combine several filing identities before reaching this
        # provider-neutral seam. Canonicalize repeated segment declarations once
        # more at the boundary so downstream Pydantic contracts never see
        # duplicate IDs.
        unique_segments: dict[str, OperatingSegment] = {}
        for item in segments:
            segment = (
                item
                if isinstance(item, OperatingSegment)
                else OperatingSegment.model_validate(item)
            )
            previous = unique_segments.get(segment.segment_id)
            if previous is None or (
                previous.name == previous.segment_id
                and segment.name != segment.segment_id
            ):
                unique_segments[segment.segment_id] = segment
        segments = tuple(unique_segments.values())
        known_segment_ids = {item.segment_id for item in segments}
        normalized_definitions = []
        seen_definition_keys: set[tuple[str, str]] = set()
        for item in definitions:
            definition = (
                item
                if isinstance(item, OperatingDriverDefinition)
                else OperatingDriverDefinition.model_validate(item)
            )
            key = (definition.segment_id, definition.driver_id)
            if key in seen_definition_keys:
                continue
            seen_definition_keys.add(key)
            normalized_definitions.append(definition)
            if definition.segment_id not in known_segment_ids:
                unique_segments[definition.segment_id] = OperatingSegment(
                    segment_id=definition.segment_id,
                    name=definition.segment_id.replace("_", " ").title(),
                    scope="segment",
                    source="first_party_filing",
                    confidence="medium",
                )
                segments = tuple(unique_segments.values())
                known_segment_ids.add(definition.segment_id)
        definitions = tuple(normalized_definitions)
        if (
            historical_revenue is not None
            and not isinstance(historical_revenue, Mapping)
            and not isinstance(historical_revenue, (str, bytes))
        ):
            historical_revenue = tuple(historical_revenue)
        independent = self.forecast_service.forecast(
            segments,
            definitions,
            observations,
            management_constraints,
            historical_revenue,
            fiscal_years,
            company_id=company_id,
        )
        company_history = normalize_company_historical_revenue(
            historical_revenue,
            segments,
        )
        reconciliation = self.reconciler.reconcile_with_details(
            independent,
            consensus_estimates=consensus_estimates,
            historical_revenue=company_history,
            explicit_anchors=explicit_anchors,
            management_anchors=management_anchors,
        )
        materialized = materialize_revenue_anchors(parameters, reconciliation)
        return _contracts.OperatingForecastIntegrationResult(
            independent_forecast=independent,
            reconciled_forecast=reconciliation.forecast,
            reconciliation=reconciliation,
            parameters=materialized,
        )

    forecast = integrate
    compose = integrate
    run = integrate


class OperatingForecastPipelineService:
    """Run structured operating evidence through FCFF and adaptive forecasting.

    The class is intentionally provider-neutral.  Callers may pass an
    ``OperatingForecastDiscoveryResult`` or any object/mapping exposing the
    normalized ``segments``, ``definitions``, ``observations``,
    ``management_constraints``, and ``historical_revenue`` fields.  Discovery,
    ticker resolution, and extraction stay outside this seam.
    """

    def __init__(
        self,
        integration_service: OperatingForecastIntegrationService | None = None,
        fcff_service: FcffForecastService | None = None,
        adaptive_service: AdaptiveMultistageFcffForecastService | None = None,
    ) -> None:
        self.integration_service = (
            integration_service or OperatingForecastIntegrationService()
        )
        self.fcff_service = fcff_service or FcffForecastService()
        self.adaptive_service = (
            adaptive_service or AdaptiveMultistageFcffForecastService(self.fcff_service)
        )

    def forecast(
        self,
        financials,
        evidence: Any = None,
        parameters: FcffForecastParameters | None = None,
        *,
        fcff_parameters: FcffForecastParameters | None = None,
        segments: Iterable[OperatingSegment] | None = None,
        definitions: Iterable[OperatingDriverDefinition] | None = None,
        observations: Iterable[OperatingDriverObservation] | None = None,
        historical_revenue: Mapping[Any, Any] | None = None,
        consensus_estimates: Iterable[ForwardRevenueEstimate]
        | Mapping[Any, Any]
        | Any = (),
        consensus: Iterable[ForwardRevenueEstimate]
        | Mapping[Any, Any]
        | Any
        | None = None,
        explicit_anchors: Mapping[Any, Any] | Iterable[Any] | Any | None = None,
        management_anchors: Mapping[Any, Any] | Iterable[Any] | Any | None = None,
        management_constraints: Any = None,
        fiscal_years: Iterable[int] | None = None,
        terminal_growth_rate: Decimal | None = None,
        adaptive_configuration: Any | None = None,
        forward_evidence: ForwardGrowthEvidence | None = None,
        normalized_tax_rate: Decimal | None = None,
        as_of=None,
        availability_mode: ObservationAvailabilityMode = (
            ObservationAvailabilityMode.POINT_IN_TIME
        ),
        company_id: str | None = None,
    ) -> _contracts.OperatingForecastPipelineResult:
        """Compose the operating selection before adaptive FCFF arithmetic."""

        if parameters is not None and fcff_parameters is not None:
            raise ValueError("Pass either parameters or fcff_parameters, not both")
        requested = fcff_parameters or parameters
        if requested is None:
            raise TypeError("fcff_parameters is required")

        seed = self.fcff_service.forecast(
            financials,
            requested,
            as_of=as_of,
            availability_mode=availability_mode,
        )
        years = tuple(
            fiscal_years
            if fiscal_years is not None
            else (item.fiscal_year for item in seed.observations)
        )
        values = _evidence_values(evidence)
        if segments is not None:
            values["segments"] = _as_items(segments)
        if definitions is not None:
            values["definitions"] = _as_items(definitions)
        if observations is not None:
            values["observations"] = _as_items(observations)
        values["definitions"] = _as_items(values.get("definitions") or ())
        values["observations"] = _as_items(values.get("observations") or ())
        initial_quality = self.quality_gate(values)
        if not initial_quality.accepted:
            raise _contracts.OperatingForecastQualityError(initial_quality)
        if consensus is not None:
            if consensus_estimates not in ((), None):
                raise ValueError(
                    "Pass either consensus or consensus_estimates, not both"
                )
            consensus_estimates = consensus
        historical = historical_revenue or values.get("historical_revenue")
        if historical is None:
            historical = _financial_revenue_history(financials)
        constraints = _as_items(values.get("management_constraints") or ())
        if management_constraints is not None:
            constraints = (*constraints, *_as_items(management_constraints))
        selected_explicit = explicit_anchors
        if selected_explicit is None:
            selected_explicit = _anchors_by_source(
                requested, ForecastAssumptionSource.EXPLICIT
            )
        selected_management = management_anchors
        if selected_management is None:
            selected_management = _anchors_by_source(
                requested, ForecastAssumptionSource.MANAGEMENT_GUIDANCE
            )

        integration = self.integration_service.integrate(
            segments=values.get("segments", ()),
            definitions=values.get("definitions", ()),
            observations=values.get("observations", ()),
            management_constraints=constraints,
            historical_revenue=historical,
            consensus_estimates=consensus_estimates,
            explicit_anchors=selected_explicit,
            management_anchors=selected_management,
            fiscal_years=years,
            parameters=requested,
            company_id=company_id or financials.company_id,
        )
        quality = self.quality_gate(values, integration.reconciled_forecast)
        if not quality.accepted:
            raise _contracts.OperatingForecastQualityError(quality)
        operating_seed = self.fcff_service.forecast(
            financials,
            integration.parameters,
            as_of=as_of,
            availability_mode=availability_mode,
        )
        operating_seed = _attach_operating_audit(
            operating_seed,
            integration.reconciled_forecast,
            additional_warnings=tuple(values.get("warnings") or ()),
        )
        operating_growth = _reconciliation_growth_evidence(
            integration.reconciliation,
            base_revenue=operating_seed.base_revenue,
        )
        combined_growth = _merge_operating_growth_evidence(
            operating_growth,
            forward_evidence,
            years=years,
        )

        if terminal_growth_rate is None or adaptive_configuration is None:
            return _contracts.OperatingForecastPipelineResult(
                integration=integration,
                seed_forecast=operating_seed,
                forecast=operating_seed,
                forward_growth=combined_growth,
                quality=quality,
            )

        forecast, plan = self.adaptive_service.forecast(
            financials,
            operating_seed,
            integration.parameters,
            terminal_growth_rate,
            adaptive_configuration,
            normalized_tax_rate=normalized_tax_rate,
            forward_evidence=combined_growth,
            as_of=as_of,
            availability_mode=availability_mode,
        )
        forecast = _attach_operating_audit(
            forecast,
            integration.reconciled_forecast,
            additional_warnings=tuple(values.get("warnings") or ()),
        )
        plan = _attach_operating_plan_audit(
            plan,
            integration.reconciled_forecast,
            additional_warnings=tuple(values.get("warnings") or ()),
        )
        return _contracts.OperatingForecastPipelineResult(
            integration=integration,
            seed_forecast=operating_seed,
            forecast=forecast,
            adaptive_plan=plan,
            forward_growth=combined_growth,
            quality=quality,
        )

    @staticmethod
    def quality_gate(
        evidence: Any,
        operating_forecast=None,
    ) -> _contracts.OperatingForecastQualityResult:
        """Return whether evidence is safe to activate in the FCFF path.

        The gate deliberately consumes the existing deterministic forecast audit
        fields rather than introducing a second confidence or scoring system.
        """

        values = _evidence_values(evidence)
        definitions_count = len(_as_items(values.get("definitions") or ()))
        observations_count = len(_as_items(values.get("observations") or ()))
        coverage = getattr(operating_forecast, "driver_coverage", None)
        modeled_revenue_share = getattr(
            operating_forecast, "modeled_revenue_share", None
        )
        reconstruction_error = getattr(operating_forecast, "reconstruction_error", None)
        confidence = getattr(operating_forecast, "confidence", None)
        own_supported_years = tuple(
            getattr(operating_forecast, "own_supported_years", ()) or ()
        )
        consensus_years = tuple(
            getattr(operating_forecast, "consensus_years", ()) or ()
        )
        transition_start_year = getattr(
            operating_forecast, "transition_start_year", None
        )

        reasons: list[str] = []
        if definitions_count == 0:
            reasons.append("definitions=0 (requires non-empty definitions)")
        if observations_count == 0:
            reasons.append("observations=0 (requires non-empty observations)")
        if operating_forecast is not None:
            coverage_text = _quality_metric_text(coverage)
            error_text = _quality_metric_text(reconstruction_error)
            confidence_text = confidence or "unavailable"
            if coverage is None or coverage < Decimal("0.60"):
                reasons.append(f"driver coverage={coverage_text} (minimum 0.60)")
            if reconstruction_error is None or reconstruction_error > Decimal("0.10"):
                reasons.append(f"reconstruction error={error_text} (maximum 0.10)")
            if str(confidence_text).casefold() == "low" or confidence is None:
                reasons.append(f"confidence={confidence_text} (must not be low)")

        if reasons:
            reason = "Operating forecast quality rejected: " + "; ".join(reasons)
            accepted = False
        elif operating_forecast is None:
            reason = "Operating forecast quality gate pending deterministic audit"
            accepted = True
        else:
            reason = "Operating forecast quality gate passed"
            accepted = True
        return _contracts.OperatingForecastQualityResult(
            accepted=accepted,
            reason=reason,
            definitions_count=definitions_count,
            observations_count=observations_count,
            driver_coverage=coverage,
            modeled_revenue_share=modeled_revenue_share,
            reconstruction_error=reconstruction_error,
            confidence=confidence,
            own_supported_years=own_supported_years,
            consensus_years=consensus_years,
            transition_start_year=transition_start_year,
            warnings=tuple(values.get("warnings") or ()),
            audit_records=tuple(values.get("audit_records") or ()),
            document_audits=tuple(values.get("document_audits") or ()),
            unusable_evidence=tuple(values.get("unusable_evidence") or ()),
            history_audit=values.get("history_audit"),
            cache_hits=int(values.get("cache_hits", 0) or 0),
            cache_misses=int(values.get("cache_misses", 0) or 0),
            filings_inspected=int(values.get("filings_inspected", 0) or 0),
            documents_inspected=int(values.get("documents_inspected", 0) or 0),
            vocabulary_audit=values.get("vocabulary_audit"),
            vocabulary_terms=tuple(values.get("vocabulary_terms") or ()),
            exhibits_found=int(values.get("exhibits_found", 0) or 0),
            gaps_resolved_sec=tuple(values.get("gaps_resolved_sec") or ()),
            gaps_resolved_ir=tuple(values.get("gaps_resolved_ir") or ()),
            ir_diagnostic=values.get("ir_diagnostic"),
        )

    def forecast_with_evidence_provider(
        self,
        financials,
        provider,
        parameters: FcffForecastParameters,
        **kwargs: Any,
    ) -> _contracts.OperatingForecastPipelineResult:
        """Resolve structured evidence from an injected provider and forecast.

        ``provider`` may expose either synchronous or asynchronous
        ``discover``/``retrieve``.  The async variant is intentionally kept
        separate so normal synchronous fixture callers remain simple and no
        ticker-specific discovery policy is introduced here.
        """

        retrieve = getattr(provider, "discover", None) or getattr(
            provider, "retrieve", None
        )
        if retrieve is None:
            raise TypeError(
                "Operating evidence provider must expose discover or retrieve"
            )
        evidence = retrieve(financials=financials, **kwargs)
        if hasattr(evidence, "__await__"):
            raise TypeError(
                "Asynchronous operating evidence providers must be awaited by the caller"
            )
        return self.forecast(financials, evidence, parameters, **kwargs)

    async def aforecast_with_evidence_provider(
        self,
        financials,
        provider,
        parameters: FcffForecastParameters,
        **kwargs: Any,
    ) -> _contracts.OperatingForecastPipelineResult:
        """Async counterpart for an injected discovery implementation."""

        retrieve = getattr(provider, "discover", None) or getattr(
            provider, "retrieve", None
        )
        if retrieve is None:
            raise TypeError(
                "Operating evidence provider must expose discover or retrieve"
            )
        evidence = retrieve(financials=financials, **kwargs)
        if hasattr(evidence, "__await__"):
            evidence = await evidence
        return self.forecast(financials, evidence, parameters, **kwargs)

    run = forecast
    compose = forecast


def _attach_operating_audit(
    forecast: FcffForecast,
    operating: CompanyOperatingForecast,
    *,
    additional_warnings: tuple[str, ...] = (),
) -> FcffForecast:
    warnings = tuple(
        dict.fromkeys((*forecast.warnings, *operating.warnings, *additional_warnings))
    )
    operating_warnings = tuple(
        dict.fromkeys((*operating.warnings, *additional_warnings))
    )
    return forecast.model_copy(
        update={
            "operating_driver_coverage": operating.driver_coverage,
            "operating_reconstruction_error": operating.reconstruction_error,
            "operating_confidence": operating.confidence,
            "operating_own_supported_years": operating.own_supported_years,
            "operating_consensus_years": operating.consensus_years,
            "operating_divergence_by_year": operating.divergence_by_year,
            "operating_divergence": operating.divergence,
            "operating_transition_start_year": operating.transition_start_year,
            "operating_warnings": operating_warnings,
            "operating_selected_revenue_by_year": operating.selected_revenue_by_year,
            "operating_source_by_year": operating.selected_source_by_year,
            "operating_confidence_by_year": operating.selected_confidence_by_year,
            "warnings": warnings,
        }
    )


def _attach_operating_plan_audit(
    plan: AdaptiveMultistagePlan,
    operating: CompanyOperatingForecast,
    *,
    additional_warnings: tuple[str, ...] = (),
) -> AdaptiveMultistagePlan:
    operating_warnings = tuple(
        dict.fromkeys((*operating.warnings, *additional_warnings))
    )
    return plan.model_copy(
        update={
            "operating_driver_coverage": operating.driver_coverage,
            "operating_reconstruction_error": operating.reconstruction_error,
            "operating_confidence": operating.confidence,
            "operating_own_supported_years": operating.own_supported_years,
            "operating_consensus_years": operating.consensus_years,
            "operating_divergence_by_year": operating.divergence_by_year,
            "operating_divergence": operating.divergence,
            "operating_transition_start_year": operating.transition_start_year,
            "operating_warnings": operating_warnings,
            "operating_selected_revenue_by_year": operating.selected_revenue_by_year,
            "operating_source_by_year": operating.selected_source_by_year,
            "operating_confidence_by_year": operating.selected_confidence_by_year,
            "warnings": tuple(dict.fromkeys((*plan.warnings, *operating.warnings))),
        }
    )


def _evidence_values(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {
        name: getattr(value, name, ())
        for name in (
            "segments",
            "definitions",
            "observations",
            "management_constraints",
            "historical_revenue",
            "warnings",
            "audit_records",
            "document_audits",
            "unusable_evidence",
            "history_audit",
            "cache_hits",
            "cache_misses",
            "filings_inspected",
            "documents_inspected",
            "raw_filings_received",
            "raw_filings_in_range",
            "candidate_filings",
            "filing_inventory_cache_bypass",
            "filing_inventory_fetched_live",
            "filing_inventory_metadata",
            "vocabulary_audit",
            "vocabulary_terms",
            "gaps_detected",
            "gaps_resolved",
            "gaps_unresolved",
            "exhibits_found",
            "gaps_resolved_sec",
            "gaps_resolved_ir",
            "ir_diagnostic",
        )
        if hasattr(value, name)
    }


def _as_items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _financial_revenue_history(financials) -> dict[int, Decimal]:
    """Use normalized annual revenue as conservative operating history."""

    result: dict[int, Decimal] = {}
    for observation in getattr(financials, "observations", ()):
        concept = getattr(getattr(observation, "concept", None), "value", "")
        granularity = getattr(getattr(observation, "granularity", None), "value", "")
        fiscal_period = getattr(
            getattr(observation, "fiscal_period", None), "value", ""
        )
        if concept != "revenue" or granularity != "annual" or fiscal_period != "FY":
            continue
        if observation.value > 0:
            result[observation.fiscal_year] = observation.value
    return result


def _anchors_by_source(
    parameters: FcffForecastParameters,
    source: ForecastAssumptionSource,
) -> dict[int, Decimal]:
    return {
        year: value
        for year, value in parameters.revenue_anchors.items()
        if parameters.revenue_anchor_sources.get(
            year, ForecastAssumptionSource.EXPLICIT
        )
        == source
    }


def _reconciliation_growth_evidence(
    reconciliation: _contracts.RevenueForecastReconciliation,
    *,
    base_revenue: Decimal,
) -> ForwardGrowthEvidence | None:
    records = reconciliation.resolved_years
    if not records:
        return None
    previous = base_revenue
    by_year: list[tuple[int, Decimal]] = []
    source_by_year: dict[int, str] = {}
    confidence_by_year: dict[int, str] = {}
    for item in records:
        if item.revenue <= 0 or previous <= 0:
            break
        growth = (item.revenue / previous - Decimal(1)) * Decimal(100)
        by_year.append((item.fiscal_year, growth))
        source_by_year[item.fiscal_year] = item.source
        confidence_by_year[item.fiscal_year] = item.confidence
        previous = item.revenue
    if not by_year:
        return None
    sources = set(source_by_year.values())
    if sources <= {"analyst_consensus"}:
        source = "analyst_consensus"
    elif sources <= {"management_guidance"}:
        source = "management_guidance"
    elif sources <= {"independent_operating", "mixed"}:
        source = "independent_operating"
    else:
        source = "operating_reconciliation"
    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    confidence = min(
        confidence_by_year.values(), key=lambda value: confidence_rank[value]
    )
    return ForwardGrowthEvidence(
        guidance="management_guidance" in sources,
        growth_path=tuple(value for _year, value in by_year),
        growth_path_by_year=tuple(by_year),
        source=source,
        confidence=confidence,
    )


def _merge_operating_growth_evidence(
    operating: ForwardGrowthEvidence | None,
    existing: ForwardGrowthEvidence | None,
    *,
    years: tuple[int, ...],
) -> ForwardGrowthEvidence | None:
    if operating is None:
        return existing
    if existing is None:
        return operating
    operating_by_year = dict(operating.growth_path_by_year)
    existing_by_year = dict(existing.growth_path_by_year)
    management_by_year = dict(existing.guidance_growth_path_by_year)
    if not management_by_year and existing.guidance:
        management_by_year = dict(existing.growth_path_by_year)
    merged: list[tuple[int, Decimal]] = []
    for year in years:
        operating_value = operating_by_year.get(year)
        existing_value = existing_by_year.get(year)
        if year in management_by_year:
            value = management_by_year[year]
        elif operating_value is not None:
            value = operating_value
        else:
            value = existing_value
        if value is not None:
            merged.append((year, value))
    prefix: list[tuple[int, Decimal]] = []
    expected = years[0] if years else None
    for year, value in merged:
        if expected is not None and year != expected:
            break
        prefix.append((year, value))
        expected = year + 1
    return operating.model_copy(
        update={
            "guidance": operating.guidance or existing.guidance,
            "guidance_growth_path": tuple(management_by_year.values()),
            "guidance_growth_path_by_year": tuple(management_by_year.items()),
            "backlog": operating.backlog or existing.backlog,
            "capacity": operating.capacity or existing.capacity,
            "growth_visibility": max(
                operating.growth_visibility, existing.growth_visibility
            ),
            "lifecycle": existing.lifecycle
            if existing.lifecycle != "unknown"
            else operating.lifecycle,
            "growth_path": tuple(value for _year, value in prefix),
            "growth_path_by_year": tuple(merged),
            "confidence": min(
                (operating.confidence or "medium", existing.confidence or "medium"),
                key={"low": 0, "medium": 1, "high": 2}.get,
            ),
            "source": (
                "management_guidance" if management_by_year else operating.source
            ),
        }
    )


merge_operating_growth_evidence = _merge_operating_growth_evidence

__all__ = [
    "OperatingForecastIntegrationService",
    "OperatingForecastPipelineService",
    "merge_operating_growth_evidence",
]


def _quality_metric_text(value: Decimal | None) -> str:
    return "unavailable" if value is None else str(value)
