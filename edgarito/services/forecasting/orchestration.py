"""Execution seam for planned FCFF forecasts.

The executor delegates all calculations to the existing FCFF service or the
existing operating pipeline.  It owns method routing only; no forecast
economics are duplicated here.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from edgarito.schemas.forecasting import (
    FcffForecast,
    FcffForecastMethod,
    FcffForecastParameters,
    ForecastOverride,
    ForecastPlan,
)
from edgarito.services.financials.availability import ObservationAvailabilityMode
from edgarito.services.forecasting._fcff.service import FcffForecastService
from edgarito.services.forecasting.plan import FcffForecastPlanService
from edgarito.services.operating.contracts import OperatingForecastQualityError


class DriverBasedForecastIncompleteError(RuntimeError):
    """Raised instead of silently falling back from the planned driver method."""

    def __init__(self, plan: ForecastPlan | None = None) -> None:
        self.plan = plan
        super().__init__(
            "FCFF forecast method 'driver_based' is planned but its execution is "
            "not implemented; no fallback forecast was used"
        )


# Name used by callers during the staged migration to the orchestration seam.
IncompleteFcffForecastMethodError = DriverBasedForecastIncompleteError


@dataclass(frozen=True)
class ForecastOrchestrationResult:
    """Canonical FCFF output together with its immutable method audit."""

    forecast: FcffForecast
    plan: ForecastPlan
    warnings: tuple[str, ...] = ()
    audit: tuple[str, ...] = ()

    @property
    def fcff_forecast(self) -> FcffForecast:
        return self.forecast

    @property
    def fcff(self) -> FcffForecast:
        return self.forecast

    @property
    def requested_method(self) -> FcffForecastMethod:
        return self.plan.requested

    @property
    def resolved_method(self) -> FcffForecastMethod:
        return self.plan.resolved

    @property
    def audit_records(self) -> tuple[str, ...]:
        return self.audit

    @property
    def method(self) -> FcffForecastMethod:
        return self.resolved_method

    @property
    def forecast_plan(self) -> ForecastPlan:
        return self.plan


class FcffForecastOrchestrationService:
    """Plan and execute normalized or hybrid FCFF forecasts."""

    def __init__(
        self,
        fcff_service: FcffForecastService | None = None,
        plan_service: FcffForecastPlanService | None = None,
        quality_gate: Any | None = None,
        operating_quality_gate: Any | None = None,
        operating_pipeline: Any | None = None,
        operating_pipeline_service: Any | None = None,
    ) -> None:
        self.fcff_service = fcff_service or FcffForecastService()
        self.plan_service = plan_service or FcffForecastPlanService(
            quality_gate=(
                quality_gate
                or operating_quality_gate
                or _default_operating_quality_gate()
            )
        )
        self.operating_pipeline = operating_pipeline or operating_pipeline_service

    def forecast(
        self,
        financials,
        parameters: FcffForecastParameters | None = None,
        method: FcffForecastMethod | str = FcffForecastMethod.NORMALIZED,
        *,
        requested_method: FcffForecastMethod | str | None = None,
        fcff_forecast_method: FcffForecastMethod | str | None = None,
        evidence: Any = None,
        operating_evidence: Any = None,
        operating_quality: Any = None,
        quality: Any = None,
        current_operating_quality: Any = None,
        operating_capability: Any = None,
        overrides: tuple[ForecastOverride, ...] | list[ForecastOverride] | dict = (),
        manual_overrides: tuple[ForecastOverride, ...]
        | list[ForecastOverride]
        | dict
        | None = None,
        pipeline_kwargs: dict[str, Any] | None = None,
        operating_pipeline_kwargs: dict[str, Any] | None = None,
        as_of=None,
        availability_mode: ObservationAvailabilityMode = (
            ObservationAvailabilityMode.POINT_IN_TIME
        ),
        **kwargs: Any,
    ) -> ForecastOrchestrationResult:
        """Execute one plan and return the existing canonical FCFF forecast.

        ``operating_evidence`` and ``operating_pipeline_kwargs`` are aliases
        retained for callers that use the valuation terminology.  Additional
        keyword arguments are forwarded only to the operating pipeline.
        """

        if requested_method is not None:
            method = requested_method
        if fcff_forecast_method is not None:
            method = fcff_forecast_method
        evidence = operating_evidence if operating_evidence is not None else evidence
        overrides = manual_overrides if manual_overrides is not None else overrides
        plan = self.plan_service.plan(
            method,
            evidence=evidence,
            operating_quality=(
                current_operating_quality
                if current_operating_quality is not None
                else operating_quality
                if operating_quality is not None
                else quality
            ),
            operating_capability=operating_capability,
            overrides=overrides,
        )
        # Revalidate custom planner output so an invalid resolved method cannot
        # bypass the dedicated driver-based execution error.
        plan = ForecastPlan.model_validate(plan, from_attributes=True)
        if plan.resolved == FcffForecastMethod.DRIVER_BASED:
            raise DriverBasedForecastIncompleteError(plan)

        if plan.resolved == FcffForecastMethod.NORMALIZED:
            forecast = self.fcff_service.forecast(
                financials,
                parameters,
                as_of=as_of,
                availability_mode=availability_mode,
            )
            pipeline_audit: tuple[str, ...] = ()
        else:
            pipeline = self.operating_pipeline or _default_operating_pipeline(
                self.fcff_service
            )
            forwarded = dict(pipeline_kwargs or {})
            forwarded.update(operating_pipeline_kwargs or {})
            forwarded.update(kwargs)
            forwarded.setdefault("evidence", evidence)
            forwarded.setdefault("parameters", parameters or FcffForecastParameters())
            forwarded.setdefault("as_of", as_of)
            forwarded.setdefault("availability_mode", availability_mode)
            try:
                result = _run_pipeline(pipeline, financials, forwarded)
            except OperatingForecastQualityError as exc:
                if plan.requested != FcffForecastMethod.AUTO:
                    raise
                plan = self.plan_service.plan(
                    plan.requested,
                    evidence=evidence,
                    operating_quality=exc.result,
                    overrides=plan.overrides,
                )
                forecast = self.fcff_service.forecast(
                    financials,
                    parameters,
                    as_of=as_of,
                    availability_mode=availability_mode,
                )
                pipeline_audit = (
                    "hybrid_pipeline_quality_rejected: " + exc.result.reason,
                )
                result = None
            else:
                forecast = getattr(result, "forecast", result)
            if not isinstance(forecast, FcffForecast):
                raise TypeError(
                    "Hybrid FCFF pipeline must return FcffForecast or a result "
                    "with a FcffForecast forecast field"
                )
            if result is not None:
                pipeline_audit = _pipeline_audit(result)
                pipeline_quality = getattr(result, "quality", None)
                if pipeline_quality is not None:
                    plan = plan.model_copy(
                        update={
                            "audit": tuple(
                                dict.fromkeys(
                                    (
                                        *plan.audit,
                                        "operating_pipeline_quality="
                                        + _quality_text(pipeline_quality),
                                    )
                                )
                            ),
                            "warnings": tuple(
                                dict.fromkeys(
                                    (
                                        *plan.warnings,
                                        *_quality_warnings(pipeline_quality),
                                    )
                                )
                            ),
                        }
                    )

        warnings = tuple(
            dict.fromkeys(
                (
                    *plan.warnings,
                    *getattr(forecast, "warnings", ()),
                )
            )
        )
        audit = tuple(dict.fromkeys((*plan.audit, *pipeline_audit)))
        return ForecastOrchestrationResult(
            forecast=forecast,
            plan=plan,
            warnings=warnings,
            audit=audit,
        )

    run = forecast
    execute = forecast
    orchestrate = forecast


def _default_operating_pipeline(fcff_service: FcffForecastService):
    # Keep the operating integration out of the module import graph for direct
    # normalized clients and to preserve the _fcff/operating boundaries.
    from edgarito.services.operating.integration import OperatingForecastPipelineService

    return OperatingForecastPipelineService(fcff_service=fcff_service)


def _default_operating_quality_gate():
    """Inject the existing activation gate without coupling the planner to it."""

    from edgarito.services.operating.integration import OperatingForecastPipelineService

    return OperatingForecastPipelineService.quality_gate


def _run_pipeline(pipeline: Any, financials: Any, kwargs: dict[str, Any]) -> Any:
    runner = getattr(pipeline, "forecast", None) or getattr(pipeline, "run", None)
    if runner is None and callable(pipeline):
        runner = pipeline
    if runner is None:
        raise TypeError("Hybrid FCFF pipeline must expose forecast or run")
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return runner(financials, **kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return runner(financials, **kwargs)
    accepted = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return runner(financials, **accepted)


def _quality_text(quality: Any) -> str:
    accepted = getattr(quality, "accepted", None)
    reason = getattr(quality, "reason", None)
    if isinstance(quality, dict):
        accepted = quality.get("accepted", accepted)
        reason = quality.get("reason", reason)
    result = "accepted" if accepted else "rejected"
    return result + (f": {reason}" if reason else "")


def _quality_warnings(quality: Any) -> tuple[str, ...]:
    warnings = (
        quality.get("warnings", ())
        if isinstance(quality, dict)
        else getattr(quality, "warnings", ())
    )
    return tuple(str(item) for item in (warnings or ()))


def _pipeline_audit(result: Any) -> tuple[str, ...]:
    quality = getattr(result, "quality", None)
    if quality is None:
        return ()
    return ("operating_pipeline_quality=" + _quality_text(quality),)


# Public spelling retained for callers that prefer a generic orchestration name.
ForecastOrchestrationService = FcffForecastOrchestrationService


__all__ = [
    "DriverBasedForecastIncompleteError",
    "FcffForecastOrchestrationService",
    "ForecastOrchestrationResult",
    "ForecastOrchestrationService",
    "IncompleteFcffForecastMethodError",
]
