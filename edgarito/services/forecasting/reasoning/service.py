"""Opt-in reason -> validate -> compile -> deterministic DRIVER_BASED service."""

from __future__ import annotations

import datetime
import inspect
from collections.abc import Mapping
from typing import Any

from edgarito.schemas.forecasting import FcffForecastParameters, ForecastOverride
from edgarito.schemas.guidance.management import ManagementGuidance
from edgarito.schemas.normalization.financials import NormalizedCompanyFinancials
from edgarito.schemas.operating import (
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingInvestmentProgram,
    OperatingSegment,
)
from edgarito.services.financials.availability import ObservationAvailabilityMode
from edgarito.services.forecasting._fcff.driver_based import (
    DriverBasedFcffForecastService,
)
from edgarito.services.forecasting.reasoning.compiler import ForecastReasoningCompiler
from edgarito.services.forecasting.reasoning.contracts import (
    ForecastReasoningInput,
    ForecastReasoningInputValidationError,
    ForecastReasoningResult,
    HistoricalFactSummary,
    ReasonedDriverBasedForecastResult,
    canonical_driver_id,
)
from edgarito.services.forecasting.reasoning.evidence import content_hash
from edgarito.services.forecasting.reasoning.reasoner import ForecastReasoner
from edgarito.services.forecasting.reasoning.validation import (
    ForecastReasoningValidator,
)
from edgarito.services.research.consensus import EvidenceConsensus
from edgarito.services.research.contracts import EvidenceItem, ResearchEvidence


class ReasonedDriverBasedForecastService:
    """Run ForecastReasoner v1 without changing any existing forecast route."""

    def __init__(
        self,
        reasoner: ForecastReasoner | None = None,
        validator: ForecastReasoningValidator | None = None,
        compiler: ForecastReasoningCompiler | None = None,
        driver_service: DriverBasedFcffForecastService | Any | None = None,
        *,
        reasoning_service: ForecastReasoner | None = None,
    ) -> None:
        self.reasoner = reasoner or reasoning_service or ForecastReasoner()
        self.validator = validator or ForecastReasoningValidator()
        self.compiler = compiler or ForecastReasoningCompiler()
        self.driver_service = driver_service or DriverBasedFcffForecastService()

    async def forecast(
        self,
        financials: NormalizedCompanyFinancials | Any,
        reasoning_input: ForecastReasoningInput | Any | None = None,
        parameters: FcffForecastParameters | None = None,
        *,
        input: ForecastReasoningInput | Any | None = None,
        as_of: datetime.date | None = None,
        forecast_years: tuple[int, ...] | list[int] | None = None,
        fiscal_years: tuple[int, ...] | list[int] | None = None,
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
        availability_mode: ObservationAvailabilityMode = ObservationAvailabilityMode.POINT_IN_TIME,
        evidence: Any = None,
        operating_evidence: Any = None,
        **kwargs: Any,
    ) -> ReasonedDriverBasedForecastResult:
        financials = (
            financials
            if isinstance(financials, NormalizedCompanyFinancials)
            else NormalizedCompanyFinancials.model_validate(financials)
        )
        reasoning_input = reasoning_input or input
        source_evidence = (
            operating_evidence if operating_evidence is not None else evidence
        )
        values = _evidence_values(source_evidence)
        if not segments:
            segments = values.get("segments", ())
        if not definitions:
            definitions = values.get("definitions", ())
        if not observations:
            observations = values.get("observations") or values.get(
                "eligible_records", ()
            )
        if not management_constraints:
            management_constraints = values.get("management_constraints", ())
        if not management_guidance:
            management_guidance = values.get("management_guidance") or values.get(
                "guidance", ()
            )
        if not investment_programs:
            investment_programs = values.get("investment_programs") or values.get(
                "programs", ()
            )
        if not research_evidence:
            research_evidence = values.get("research_evidence") or values.get(
                "research", ()
            )
        if not evidence_consensus:
            evidence_consensus = values.get("evidence_consensus") or values.get(
                "consensus", ()
            )
        if historical_facts is None:
            historical_facts = values.get("historical_facts")
        if overrides is not None:
            manual_overrides = overrides
        if forward_driver_observations is not None:
            manual_forward_driver_observations = forward_driver_observations
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
                manual_forward_driver_observations=manual_forward_driver_observations,
            )
        else:
            reasoning_input = (
                reasoning_input
                if isinstance(reasoning_input, ForecastReasoningInput)
                else ForecastReasoningInput.model_validate(reasoning_input)
            )
            reasoning_input = _merge_additive_input(
                reasoning_input,
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
                manual_forward_driver_observations=manual_forward_driver_observations,
            )
        if parameters is None:
            parameters = FcffForecastParameters(
                forecast_years=len(reasoning_input.forecast_years)
            )
        if parameters.forecast_years != len(reasoning_input.forecast_years):
            raise ValueError("FCFF parameters and reasoning input horizons must match")

        proposal = await self.reasoner.reason(reasoning_input)
        validation = self.validator.validate(
            proposal.response,
            reasoning_input,
            proposal.catalog,
        )
        if validation.input_issues:
            raise ForecastReasoningInputValidationError(validation.input_issues)
        compilation = self.compiler.compile(
            reasoning_input,
            validation,
            metadata=proposal.metadata,
        )

        all_observations = tuple(
            (
                *reasoning_input.observations,
                *reasoning_input.manual_forward_driver_observations,
                *compilation.observations,
            )
        )
        management = tuple(
            (
                *reasoning_input.management_constraints,
                *reasoning_input.management_guidance,
            )
        )
        driver_kwargs = dict(kwargs)
        if source_evidence is not None:
            driver_kwargs.setdefault("evidence", source_evidence)
            driver_kwargs.setdefault("operating_evidence", source_evidence)
        driver_kwargs.update(
            {
                "plan": compilation.plan,
                "forecast_plan": compilation.plan,
                "overrides": compilation.overrides,
                "forecast_overrides": compilation.overrides,
                "segments": reasoning_input.segments,
                "definitions": reasoning_input.definitions,
                "observations": all_observations,
                "management_constraints": management,
                "investment_programs": reasoning_input.investment_programs,
                "historical_revenue": _historical_revenue(reasoning_input),
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
        driver_result = _call_driver(runner, financials, parameters, driver_kwargs)
        if inspect.isawaitable(driver_result):
            driver_result = await driver_result

        rejected = tuple((*validation.rejected_assumptions, *compilation.collisions))
        warnings = tuple(
            dict.fromkeys(
                (
                    *proposal.warnings,
                    *validation.warnings,
                    *compilation.warnings,
                    *getattr(driver_result, "warnings", ()),
                )
            )
        )
        audit_identity = (
            f"proposal_cache_key={proposal.cache_key}",
            f"evidence_bundle_hash={proposal.catalog.bundle_hash}",
            f"model={proposal.metadata.model}",
            f"prompt_version={proposal.metadata.prompt_version}",
            f"schema_version={proposal.metadata.schema_version}",
            f"validator_version={proposal.metadata.validator_version}",
            "execution_method=driver_based",
            "accounting=deterministic_driver_service",
            "validation=read_only",
        )
        reasoning_result = ForecastReasoningResult(
            proposal_identity=proposal.cache_key,
            audit_identity=audit_identity,
            proposal=proposal,
            accepted_assumptions=validation.accepted_assumptions,
            accepted_decisions=validation.accepted_decisions,
            rejected_assumptions=rejected,
            rejected_decisions=validation.rejected_decisions,
            unresolved_items=validation.unresolved_items,
            warnings=warnings,
            evidence_catalog=proposal.catalog,
            metadata=proposal.metadata,
            compiled_plan=compilation.plan,
            compiled_observations=compilation.observations,
            compiled_overrides=compilation.overrides,
            retained_ranges=compilation.retained_ranges,
            collisions=compilation.collisions,
            cache_hit=proposal.cache_hit,
            driver_result=driver_result,
        )
        return ReasonedDriverBasedForecastResult(
            reasoning=reasoning_result,
            driver_result=driver_result,
        )

    async def run(self, *args: Any, **kwargs: Any):
        return await self.forecast(*args, **kwargs)

    async def build(self, *args: Any, **kwargs: Any):
        return await self.forecast(*args, **kwargs)

    aforecast = forecast


ForecastReasonedDriverBasedForecastService = ReasonedDriverBasedForecastService
ForecastReasoningInputError = ForecastReasoningInputValidationError
InvalidForecastReasoningInputError = ForecastReasoningInputValidationError


def _historical_revenue(input_value: ForecastReasoningInput) -> dict[int, Any]:
    return {
        item.fiscal_year: item.value
        for item in input_value.historical_facts
        if item.metric.casefold().replace("-", "_").replace(" ", "_")
        in {"revenue", "segment_revenue"}
        and item.scope == "company"
    }


def _evidence_values(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    names = (
        "segments",
        "definitions",
        "observations",
        "eligible_records",
        "management_guidance",
        "guidance",
        "management_constraints",
        "investment_programs",
        "programs",
        "historical_facts",
        "research_evidence",
        "research",
        "evidence_consensus",
    )
    result = {name: getattr(value, name) for name in names if hasattr(value, name)}
    if "observations" not in result:
        result["observations"] = getattr(value, "eligible_records", ())
    return result


def _coerce_records(value: Any) -> tuple[Any, ...]:
    if value is None or value == () or value == []:
        return ()
    if hasattr(value, "applications"):
        return tuple(item.guidance for item in value.applications)
    if hasattr(value, "records"):
        return tuple(value.records)
    if hasattr(value, "eligible_records"):
        return tuple(value.eligible_records)
    if isinstance(value, Mapping):
        if {"segment_id", "driver_id", "fiscal_year"}.issubset(value):
            return (value,)
        return tuple(value.values())
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _coerce_overrides(value: Any) -> tuple[ForecastOverride, ...]:
    values = _coerce_records(value)
    return tuple(
        item
        if isinstance(item, ForecastOverride)
        else ForecastOverride.model_validate(item)
        for item in values
    )


def _merge_additive_input(
    base: ForecastReasoningInput,
    *,
    segments: Any,
    definitions: Any,
    observations: Any,
    management_guidance: Any,
    management_constraints: Any,
    investment_programs: Any,
    historical_facts: Any,
    research_evidence: Any,
    evidence_consensus: Any,
    manual_overrides: Any,
    manual_forward_driver_observations: Any,
) -> ForecastReasoningInput:
    """Merge all additive service arguments before reasoning and execution."""

    updates: dict[str, Any] = {}
    collection_specs = (
        ("segments", segments, OperatingSegment, lambda item: item.segment_id),
        (
            "definitions",
            definitions,
            OperatingDriverDefinition,
            lambda item: (item.segment_id, item.driver_id),
        ),
        (
            "observations",
            observations,
            OperatingDriverObservation,
            lambda item: (
                item.segment_id,
                canonical_driver_id(item.driver_id),
                item.fiscal_year,
                item.fiscal_period,
                item.period_key,
            ),
        ),
        (
            "management_guidance",
            management_guidance,
            ManagementGuidance,
            lambda item: (
                str(getattr(item.metric, "value", item.metric)),
                item.fiscal_year,
                item.fiscal_quarter,
                item.segment_name,
                str(getattr(item.period_type, "value", item.period_type)),
            ),
        ),
        (
            "investment_programs",
            investment_programs,
            OperatingInvestmentProgram,
            lambda item: item.program_id,
        ),
        (
            "historical_facts",
            historical_facts,
            HistoricalFactSummary,
            lambda item: (
                item.scope.value,
                item.scope_id,
                item.metric,
                item.fiscal_year,
                item.fiscal_period,
            ),
        ),
        (
            "evidence_consensus",
            evidence_consensus,
            EvidenceConsensus,
            lambda item: (
                str(getattr(item.kind, "value", item.kind)),
                item.unit,
                item.context.model_dump_json(),
            ),
        ),
        (
            "manual_forward_driver_observations",
            manual_forward_driver_observations,
            OperatingDriverObservation,
            lambda item: (
                item.segment_id,
                canonical_driver_id(item.driver_id),
                item.fiscal_year,
            ),
        ),
    )
    for field_name, raw, model, key in collection_specs:
        values = (
            tuple(
                item if isinstance(item, model) else model.model_validate(item)
                for item in _coerce_records(raw)
            )
            if raw not in (None, (), [])
            else ()
        )
        merged = _merge_records(getattr(base, field_name), values, key, field_name)
        if merged != getattr(base, field_name):
            updates[field_name] = merged

    constraint_values = (
        _coerce_records(management_constraints)
        if management_constraints not in (None, (), [])
        else ()
    )
    merged_constraints = _merge_records(
        base.management_constraints,
        constraint_values,
        _constraint_key,
        "management_constraints",
    )
    if merged_constraints != base.management_constraints:
        updates["management_constraints"] = merged_constraints

    override_values = (
        _coerce_overrides(manual_overrides)
        if manual_overrides not in (None, (), [])
        else ()
    )
    merged_overrides = _merge_records(
        base.manual_overrides,
        override_values,
        lambda item: item.key,
        "manual_overrides",
    )
    if merged_overrides != base.manual_overrides:
        updates["manual_overrides"] = merged_overrides
    from pydantic import TypeAdapter

    adapter = TypeAdapter(EvidenceItem)
    research_values = (
        tuple(
            item
            if isinstance(item, ResearchEvidence)
            else adapter.validate_python(item)
            for item in _coerce_records(research_evidence)
        )
        if research_evidence not in (None, (), [])
        else ()
    )
    merged_research = _merge_records(
        base.research_evidence,
        research_values,
        lambda item: item.evidence_id or content_hash(item),
        "research_evidence",
    )
    if merged_research != base.research_evidence:
        updates["research_evidence"] = merged_research

    if not updates:
        return base
    return base.model_copy(update=updates)


def _merge_records(existing, incoming, key, label: str):
    result = []
    by_key = {}
    for item in (*existing, *incoming):
        item_key = key(item)
        previous = by_key.get(item_key)
        if previous is not None:
            if _semantic_hash(previous) != _semantic_hash(item):
                raise ValueError(f"Conflicting duplicate {label} record: {item_key}")
            continue
        by_key[item_key] = item
        result.append(item)
    return tuple(result)


def _semantic_hash(value: Any) -> str:
    if isinstance(value, OperatingDriverObservation):
        value = value.model_copy(
            update={"driver_id": canonical_driver_id(value.driver_id)}
        )
    return content_hash(value)


def _constraint_key(item: Any):
    if isinstance(item, OperatingDriverObservation):
        return (
            item.segment_id,
            canonical_driver_id(item.driver_id),
            item.fiscal_year,
            item.fiscal_period,
            item.period_key,
        )
    return ("content", content_hash(item))


def _call_driver(runner, financials, parameters, kwargs):
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return runner(financials, parameters, **kwargs)
    positional = tuple(
        item
        for item in signature.parameters.values()
        if item.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    if len(positional) < 2:
        kwargs = dict(kwargs)
        kwargs.setdefault("parameters", parameters)
        if any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        ):
            return runner(financials, **kwargs)
        accepted = {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
        return runner(financials, **accepted)
    if any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        for item in signature.parameters.values()
    ):
        return runner(financials, parameters, **kwargs)
    accepted = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return runner(financials, parameters, **accepted)


__all__ = [
    "ReasonedDriverBasedForecastService",
    "ForecastReasonedDriverBasedForecastService",
    "ForecastReasoningInputValidationError",
    "ForecastReasoningInputError",
    "InvalidForecastReasoningInputError",
]
