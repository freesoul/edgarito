"""Opt-in economic-graph reasoning composed with DRIVER_BASED execution."""

from __future__ import annotations

import datetime
import inspect
from dataclasses import dataclass
from typing import Any

from edgarito.schemas.forecasting import FcffForecastParameters, ForecastPlan
from edgarito.schemas.normalization.financials import NormalizedCompanyFinancials
from edgarito.schemas.operating_graph import EconomicEvaluationResult, EconomicModel
from edgarito.services.financials.availability import ObservationAvailabilityMode
from edgarito.services.forecasting._fcff.driver_based import (
    DriverBasedFcffForecastService,
)
from edgarito.services.forecasting.reasoning.contracts import ForecastReasoningInput
from edgarito.services.forecasting.reasoning.reasoner import ForecastReasoner


@dataclass(frozen=True)
class EconomicGraphDriverBasedForecastResult:
    """Graph reasoning audit together with the canonical driver-based result."""

    graph_result: Any
    driver_result: Any

    @property
    def reasoning(self) -> Any:
        return self.graph_result

    @property
    def graph_reasoning(self) -> Any:
        return self.graph_result

    @property
    def economic_graph_result(self) -> Any:
        return self.graph_result

    @property
    def forecast(self):
        return self.driver_result.forecast

    @property
    def driver_forecast(self):
        return self.driver_result.forecast

    @property
    def company_operating_forecast(self):
        return self.driver_result.company_operating_forecast

    @property
    def company_operating_economics(self):
        return self.driver_result.company_operating_economics

    @property
    def readiness(self):
        return self.driver_result.readiness

    @property
    def warnings(self):
        return tuple(
            dict.fromkeys(
                (
                    *getattr(self.graph_result, "warnings", ()),
                    *getattr(self.driver_result, "warnings", ()),
                )
            )
        )

    @property
    def audit(self):
        return tuple(
            dict.fromkeys(
                (
                    *getattr(self.graph_result, "audit_identity", ()),
                    *getattr(self.driver_result, "audit", ()),
                )
            )
        )


class EconomicGraphDriverBasedForecastService:
    """Reason only graph leaves, then execute the existing DRIVER_BASED route.

    This service is intentionally not selected by ``AUTO`` or by the existing
    ``ReasonedDriverBasedForecastService``.  It is an explicit composition seam
    for callers that already own an economic graph.
    """

    def __init__(
        self,
        reasoner: ForecastReasoner | Any | None = None,
        driver_service: DriverBasedFcffForecastService | Any | None = None,
        *,
        reasoning_service: ForecastReasoner | Any | None = None,
    ) -> None:
        self.reasoner = reasoner or reasoning_service or ForecastReasoner()
        self.driver_service = driver_service or DriverBasedFcffForecastService()

    async def forecast(
        self,
        financials: NormalizedCompanyFinancials | Any,
        economic_model: EconomicModel | Any | None = None,
        parameters: FcffForecastParameters | None = None,
        *,
        economic_graph: EconomicModel | Any | None = None,
        reasoning_input: ForecastReasoningInput | Any | None = None,
        input: ForecastReasoningInput | Any | None = None,
        as_of: datetime.date | None = None,
        forecast_years: tuple[int, ...] | list[int] | None = None,
        fiscal_years: tuple[int, ...] | list[int] | None = None,
        graph_observations: Any = (),
        economic_observations: Any = (),
        economic_graph_observations: Any = (),
        compiled_graph_observations: Any = (),
        graph_evaluation: EconomicEvaluationResult | Any | None = None,
        economic_evaluation: EconomicEvaluationResult | Any | None = None,
        economic_graph_evaluation: EconomicEvaluationResult | Any | None = None,
        evaluation: EconomicEvaluationResult | Any | None = None,
        segments: Any = (),
        definitions: Any = (),
        observations: Any = (),
        management_guidance: Any = (),
        management_constraints: Any = (),
        investment_programs: Any = (),
        historical_facts: Any = None,
        research_evidence: Any = (),
        evidence_consensus: Any = (),
        manual_overrides: Any = (),
        overrides: Any | None = None,
        manual_forward_driver_observations: Any = (),
        forward_driver_observations: Any | None = None,
        plan: ForecastPlan | Any | None = None,
        forecast_plan: ForecastPlan | Any | None = None,
        availability_mode: ObservationAvailabilityMode = ObservationAvailabilityMode.POINT_IN_TIME,
        evidence: Any = None,
        operating_evidence: Any = None,
        **kwargs: Any,
    ) -> EconomicGraphDriverBasedForecastResult:
        financials = (
            financials
            if isinstance(financials, NormalizedCompanyFinancials)
            else NormalizedCompanyFinancials.model_validate(financials)
        )
        model_value = economic_model if economic_model is not None else economic_graph
        if model_value is None:
            raise TypeError("EconomicGraphDriverBasedForecastService requires an economic_model")
        model = (
            model_value
            if isinstance(model_value, EconomicModel)
            else EconomicModel.model_validate(model_value)
        )
        reasoning_input = reasoning_input or input
        if reasoning_input is None:
            years = tuple(forecast_years or fiscal_years or ())
            if not years:
                horizon = parameters.forecast_years if parameters is not None else 5
                last_year = max(item.fiscal_year for item in financials.observations)
                years = tuple(last_year + offset for offset in range(1, horizon + 1))
            reasoning_input = ForecastReasoningInput.from_artifacts(
                financials,
                as_of=as_of or datetime.date.today(),
                forecast_years=years,
                segments=segments,
                definitions=definitions,
                observations=observations,
                management_guidance=management_guidance,
                management_constraints=management_constraints,
                investment_programs=investment_programs,
                historical_facts=historical_facts,
                research_evidence=research_evidence,
                evidence_consensus=evidence_consensus,
                manual_overrides=manual_overrides,
                manual_forward_driver_observations=(
                    manual_forward_driver_observations
                    if forward_driver_observations is None
                    else forward_driver_observations
                ),
            )
        else:
            reasoning_input = (
                reasoning_input
                if isinstance(reasoning_input, ForecastReasoningInput)
                else ForecastReasoningInput.model_validate(reasoning_input)
            )
        if parameters is None:
            parameters = FcffForecastParameters(
                forecast_years=len(reasoning_input.forecast_years)
            )
        if parameters.forecast_years != len(reasoning_input.forecast_years):
            raise ValueError("FCFF parameters and reasoning input horizons must match")

        model = model.model_copy(
            update={
                "observations": (
                    *model.observations,
                    *self._graph_observations(
                        graph_observations,
                        economic_observations,
                        economic_graph_observations,
                        compiled_graph_observations,
                    ),
                )
            }
        )
        graph_evaluation = next(
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
        graph_reasoning = await self._reason_graph(
            reasoning_input,
            model,
            graph_evaluation,
        )

        driver_kwargs = dict(kwargs)
        if operating_evidence is not None or evidence is not None:
            source_evidence = (
                operating_evidence if operating_evidence is not None else evidence
            )
            driver_kwargs.update(
                {"evidence": source_evidence, "operating_evidence": source_evidence}
            )
        driver_kwargs.update(
            {
                "plan": plan if plan is not None else forecast_plan,
                "forecast_plan": plan if plan is not None else forecast_plan,
                "overrides": overrides if overrides is not None else manual_overrides,
                "forecast_overrides": overrides
                if overrides is not None
                else manual_overrides,
                "segments": reasoning_input.segments,
                "definitions": reasoning_input.definitions,
                "observations": (
                    *reasoning_input.observations,
                    *reasoning_input.manual_forward_driver_observations,
                ),
                "management_constraints": (
                    *reasoning_input.management_constraints,
                    *reasoning_input.management_guidance,
                ),
                "investment_programs": reasoning_input.investment_programs,
                "economic_model": model,
                "graph_observations": getattr(graph_reasoning, "compiled_observations", ()),
                "economic_evaluation": getattr(graph_reasoning, "evaluation", None),
                "company_id": reasoning_input.company_id,
                "as_of": reasoning_input.as_of,
                "availability_mode": availability_mode,
            }
        )
        runner = getattr(self.driver_service, "forecast", None) or getattr(
            self.driver_service, "run", None
        )
        if runner is None:
            raise TypeError("Driver service must expose forecast or run")
        driver_result = runner(financials, parameters, **driver_kwargs)
        if inspect.isawaitable(driver_result):
            driver_result = await driver_result
        return EconomicGraphDriverBasedForecastResult(
            graph_result=graph_reasoning,
            driver_result=driver_result,
        )

    async def _reason_graph(
        self,
        reasoning_input: ForecastReasoningInput,
        model: EconomicModel,
        evaluation: Any,
    ) -> Any:
        runner = getattr(self.reasoner, "reason_economic_model", None)
        if runner is not None:
            result = runner(reasoning_input, model, evaluation=evaluation)
        else:
            runner = getattr(self.reasoner, "reason", None) or getattr(
                self.reasoner, "forecast", None
            )
            if runner is None:
                raise TypeError("Graph reasoner must expose reason_economic_model or reason")
            result = runner(
                {
                    "forecast_input": reasoning_input,
                    "economic_model": model,
                    "evaluation": evaluation,
                }
            )
        if inspect.isawaitable(result):
            result = await result
        return result

    @staticmethod
    def _graph_observations(*values: Any) -> tuple[Any, ...]:
        result: list[Any] = []
        for value in values:
            if value is None:
                continue
            if hasattr(value, "compiled_observations"):
                value = value.compiled_observations
            if isinstance(value, dict):
                if "node_id" in value or "node" in value:
                    value = (value,)
                else:
                    value = value.get(
                        "observations", value.get("compiled_observations", ())
                    )
            if isinstance(value, (str, bytes)):
                value = (value,)
            try:
                result.extend(value)
            except TypeError:
                result.append(value)
        return tuple(result)

    async def run(self, *args: Any, **kwargs: Any):
        return await self.forecast(*args, **kwargs)

    async def build(self, *args: Any, **kwargs: Any):
        return await self.forecast(*args, **kwargs)

    compose = forecast


__all__ = [
    "EconomicGraphDriverBasedForecastResult",
    "EconomicGraphDriverBasedForecastService",
]
