"""Deterministic translation of validated proposals into existing domain types."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from edgarito.schemas.forecasting import (
    FcffForecastMethod,
    ForecastOverride,
    ForecastProvenance,
    ForecastStrategy,
)
from edgarito.schemas.forecasting import (
    ForecastValueBasis as ExecutableValueBasis,
)
from edgarito.schemas.guidance.management import ManagementGuidance
from edgarito.schemas.operating import (
    OperatingDriverObservation,
    canonical_operating_segment_id,
)
from edgarito.schemas.valuation.assumptions import (
    AssumptionOrigin,
    AssumptionProvenance,
)
from edgarito.services.forecasting.plan import FcffForecastPlanService
from edgarito.services.forecasting.reasoning.contracts import (
    ForecastReasoningCompilation,
    ForecastReasoningInput,
    ForecastReasoningInputValidationError,
    ForecastReasoningValidationIssue,
    ForecastReasoningValueBasis,
    ReasonedForecastAssumption,
    canonical_driver_id,
)
from edgarito.services.forecasting.reasoning.validation import (
    ForecastReasoningValidationResult,
    _metric_key,
)

_DERIVED = frozenset({"gross_profit", "ebit", "tax", "nopat", "delta_nwc", "fcff"})
_AUTHORITATIVE_ORIGINS = frozenset(
    {"reported", "first_party_observation", "extracted_evidence", "management_guidance"}
)


class ForecastReasoningCompiler:
    """Compile only BASE values; preserve range scenarios as audit data."""

    def __init__(self, plan_service: FcffForecastPlanService | None = None) -> None:
        self.plan_service = plan_service or FcffForecastPlanService()

    def compile(
        self,
        input_value: ForecastReasoningInput,
        validation: ForecastReasoningValidationResult,
        *,
        metadata: Any | None = None,
    ) -> ForecastReasoningCompilation:
        input_value = (
            input_value
            if isinstance(input_value, ForecastReasoningInput)
            else ForecastReasoningInput.model_validate(input_value)
        )
        if validation.input_issues:
            raise ForecastReasoningInputValidationError(validation.input_issues)
        manual_overrides = tuple(input_value.manual_overrides)
        manual_override_keys = {_override_key(item) for item in manual_overrides}
        manual_driver_keys = {
            (item.segment_id, canonical_driver_id(item.driver_id), item.fiscal_year)
            for item in input_value.manual_forward_driver_observations
        }
        authoritative_driver_keys = manual_driver_keys | {
            key
            for item in input_value.observations
            if item.origin in _AUTHORITATIVE_ORIGINS
            for key in _observation_keys(item)
        }
        authoritative_driver_keys |= {
            key
            for item in input_value.management_constraints
            for key in _constraint_observation_keys(item)
        }
        authoritative_driver_keys |= {
            key
            for item in input_value.management_guidance
            for key in _guidance_observation_keys(item)
        }
        authoritative_metric_keys = _authoritative_metric_keys(input_value)
        generated_observations: list[OperatingDriverObservation] = []
        generated_overrides: list[ForecastOverride] = []
        collisions: list[ForecastReasoningValidationIssue] = []
        retained: dict[str, tuple[tuple[Decimal, Decimal, Decimal], ...]] = {}
        warnings: list[str] = list(validation.warnings)

        for assumption in validation.accepted_assumptions:
            retained[assumption.assumption_id] = tuple(
                zip(assumption.low, assumption.base, assumption.high, strict=True)
            )
            if assumption.target_type == "operating_driver":
                generated = self._compile_driver(assumption, metadata)
                blocked = [
                    item
                    for item in generated
                    if (
                        item.segment_id,
                        canonical_driver_id(item.driver_id),
                        item.fiscal_year,
                    )
                    in authoritative_driver_keys
                ]
                if blocked:
                    collisions.append(
                        ForecastReasoningValidationIssue(
                            assumption_id=assumption.assumption_id,
                            code=(
                                "MANUAL_DRIVER_PRECEDENCE"
                                if any(
                                    (
                                        item.segment_id,
                                        canonical_driver_id(item.driver_id),
                                        item.fiscal_year,
                                    )
                                    in manual_driver_keys
                                    for item in blocked
                                )
                                else "AUTHORITATIVE_DRIVER_PRECEDENCE"
                            ),
                            reason="Authoritative forward driver observation wins; reasoned observation retained but not executed",
                        )
                    )
                    generated = tuple(item for item in generated if item not in blocked)
                generated_observations.extend(generated)
            else:
                override = self._compile_metric(assumption, metadata)
                if override is None:
                    warnings.append(
                        f"{assumption.assumption_id}: unsupported financial target was retained as audit context"
                    )
                    continue
                if _override_key(override) in manual_override_keys:
                    collisions.append(
                        ForecastReasoningValidationIssue(
                            assumption_id=assumption.assumption_id,
                            code="MANUAL_OVERRIDE_PRECEDENCE",
                            reason="Manual forecast override wins; reasoned override retained but not executed",
                        )
                    )
                elif any(
                    (
                        override.scope.value,
                        override.scope_id,
                        _economic_metric_key(override.metric),
                        year,
                    )
                    in authoritative_metric_keys
                    for year in input_value.forecast_years
                ):
                    collisions.append(
                        ForecastReasoningValidationIssue(
                            assumption_id=assumption.assumption_id,
                            code="AUTHORITATIVE_METRIC_PRECEDENCE",
                            reason="Authoritative same-scope metric observation wins; reasoned override retained but not executed",
                        )
                    )
                else:
                    generated_overrides.append(override)

        all_overrides = tuple(
            sorted(
                (*manual_overrides, *generated_overrides),
                key=_override_key,
            )
        )
        base_plan = self.plan_service.plan(
            FcffForecastMethod.DRIVER_BASED,
            evidence={"segments": input_value.segments},
            overrides=all_overrides,
        )
        plan = base_plan
        if validation.accepted_decisions:
            warnings.append(
                "Proposed modeling decisions are retained for audit only and were not executed"
            )
        if collisions:
            warnings.extend(issue.reason for issue in collisions)
        return ForecastReasoningCompilation(
            plan=plan,
            observations=tuple(
                sorted(
                    generated_observations,
                    key=lambda item: (
                        item.segment_id,
                        item.driver_id,
                        item.fiscal_year,
                    ),
                )
            ),
            overrides=all_overrides,
            retained_ranges=retained,
            collisions=tuple(collisions),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    compile_response = compile
    build = compile

    def _compile_driver(
        self,
        assumption: ReasonedForecastAssumption,
        metadata: Any | None,
    ) -> tuple[OperatingDriverObservation, ...]:
        provenance = self._assumption_provenance(assumption, metadata)
        origin = (
            "reasoned_assumption" if assumption.evidence_based else "model_assumption"
        )
        return tuple(
            OperatingDriverObservation(
                segment_id=assumption.scope_id,
                driver_id=canonical_driver_id(assumption.driver_id),
                fiscal_year=year,
                value=base,
                low=low,
                high=high,
                unit=assumption.unit,
                scope="segment",
                scope_evidence="ForecastReasoner v1 compiled operating input",
                origin=origin,
                confidence=assumption.confidence,
                provenance=provenance,
            )
            for year, low, base, high in zip(
                assumption.fiscal_years,
                assumption.low,
                assumption.base,
                assumption.high,
                strict=True,
            )
        )

    def _compile_metric(
        self,
        assumption: ReasonedForecastAssumption,
        metadata: Any | None,
    ) -> ForecastOverride | None:
        metric = _metric_key(assumption.metric)
        if metric in _DERIVED:
            return None
        basis = assumption.basis
        target_metric = metric
        strategy = ForecastStrategy.EXPLICIT
        executable_basis = ExecutableValueBasis.ABSOLUTE
        if metric == "gross_margin":
            executable_basis = ExecutableValueBasis.PERCENTAGE_POINTS
        elif metric == "tax_rate":
            executable_basis = ExecutableValueBasis.PERCENTAGE_POINTS
        elif metric in {"r_and_d", "sg_and_a"}:
            if basis == ForecastReasoningValueBasis.PERCENT_OF_REVENUE:
                strategy = ForecastStrategy.RATIO
                executable_basis = ExecutableValueBasis.PERCENT_OF_REVENUE
            else:
                executable_basis = ExecutableValueBasis.ABSOLUTE
        elif metric in {
            "depreciation_and_amortization",
            "capex",
            "operating_working_capital",
        }:
            if basis == ForecastReasoningValueBasis.PERCENT_OF_REVENUE:
                target_metric = {
                    "depreciation_and_amortization": "depreciation_to_revenue",
                    "capex": "capex_to_revenue",
                    "operating_working_capital": "operating_working_capital_to_revenue",
                }[metric]
                strategy = ForecastStrategy.RATIO
                executable_basis = ExecutableValueBasis.PERCENT_OF_REVENUE
        elif metric in {
            "depreciation_to_revenue",
            "capex_to_revenue",
            "operating_working_capital_to_revenue",
        }:
            strategy = ForecastStrategy.RATIO
            executable_basis = ExecutableValueBasis.PERCENT_OF_REVENUE
        elif metric == "revenue":
            executable_basis = ExecutableValueBasis.ABSOLUTE
        elif metric == "other_operating_items":
            executable_basis = ExecutableValueBasis.ABSOLUTE
        else:
            return None
        provenance = self._forecast_provenance(assumption, metadata)
        return ForecastOverride(
            scope=assumption.scope,
            scope_id=assumption.scope_id,
            metric=target_metric,
            strategy=strategy,
            explicit_path=assumption.base,
            basis=executable_basis,
            provenance=provenance,
        )

    @staticmethod
    def _assumption_provenance(
        assumption: ReasonedForecastAssumption, metadata: Any | None
    ):
        return AssumptionProvenance(
            origin=(
                AssumptionOrigin.REASONED_ASSUMPTION
                if assumption.evidence_based
                else AssumptionOrigin.MODEL_ASSUMPTION
            ),
            provider="ForecastReasoner",
            assumption_id=assumption.assumption_id,
            evidence_ids=assumption.evidence_ids,
            model=getattr(metadata, "model", None),
            prompt_hash=getattr(metadata, "prompt_hash", None),
            prompt_version=getattr(metadata, "prompt_version", None),
            schema_version=getattr(metadata, "schema_version", None),
            validator_version=getattr(metadata, "validator_version", None),
            methodology=_reasoned_methodology(assumption),
        )

    @staticmethod
    def _forecast_provenance(
        assumption: ReasonedForecastAssumption, metadata: Any | None
    ):
        return ForecastProvenance(
            source="ForecastReasoner",
            origin=assumption.assumption_type,
            methodology=_reasoned_methodology(assumption),
            reference=(
                f"assumption_id={assumption.assumption_id};"
                f"evidence_ids={','.join(assumption.evidence_ids) or 'none'}"
            ),
            assumption_id=assumption.assumption_id,
            evidence_ids=assumption.evidence_ids,
            model=getattr(metadata, "model", None),
            prompt_hash=getattr(metadata, "prompt_hash", None),
            prompt_version=getattr(metadata, "prompt_version", None),
            schema_version=getattr(metadata, "schema_version", None),
            validator_version=getattr(metadata, "validator_version", None),
        )


def _reasoned_methodology(assumption: ReasonedForecastAssumption) -> str:
    """Return stable provenance without accepting model-supplied prose."""

    target_type = getattr(assumption.target_type, "value", assumption.target_type)
    basis = getattr(assumption.basis, "value", assumption.basis)
    return f"reasoned:{target_type}:{basis}:{assumption.assumption_type}"


def _override_key(value: ForecastOverride) -> tuple[str, str, str]:
    metric = _economic_metric_key(value.metric)
    # Legacy and canonical ratio spellings represent the same executable cell.
    return (value.scope.value, value.scope_id, metric)


def _economic_metric_key(value: Any) -> str:
    """Return one collision identity for amount and ratio economic targets."""

    metric = _metric_key(value)
    return {
        "depreciation_to_revenue": "depreciation_and_amortization",
        "capex_to_revenue": "capex",
        "operating_working_capital_to_revenue": "operating_working_capital",
    }.get(metric, metric)


def _authoritative_metric_keys(
    input_value: ForecastReasoningInput,
) -> set[tuple[str, str, str, int]]:
    keys: set[tuple[str, str, str, int]] = set()
    for observation in input_value.observations:
        if observation.origin in _AUTHORITATIVE_ORIGINS:
            keys.update(
                _metric_observation_keys(observation, input_value.forecast_years)
            )
    for observation in input_value.manual_forward_driver_observations:
        keys.update(_metric_observation_keys(observation, input_value.forecast_years))
    for item in input_value.management_constraints:
        if isinstance(item, OperatingDriverObservation):
            keys.update(_metric_observation_keys(item, input_value.forecast_years))
    for guidance in input_value.management_guidance:
        if guidance.fiscal_year in input_value.forecast_years:
            scope, scope_id = _scope_key(guidance.scope.value, guidance.segment_name)
            keys.add(
                (
                    scope,
                    scope_id,
                    _economic_metric_key(guidance.metric),
                    guidance.fiscal_year,
                )
            )
    return keys


def _metric_observation_keys(
    observation: OperatingDriverObservation,
    forecast_years: tuple[int, ...],
) -> set[tuple[str, str, str, int]]:
    if observation.fiscal_year not in forecast_years:
        return set()
    scope, scope_id = _scope_key(observation.scope, observation.segment_id)
    return {
        (
            scope,
            scope_id,
            _economic_metric_key(observation.driver_id),
            observation.fiscal_year,
        )
    }


def _scope_key(scope: str, scope_id: str | None) -> tuple[str, str]:
    normalized = str(scope).casefold()
    if normalized in {"company", "consolidated"} or scope_id in {
        None,
        "company",
        "consolidated",
    }:
        return "company", "company"
    return "segment", str(scope_id)


def _observation_keys(
    value: OperatingDriverObservation,
) -> tuple[tuple[str, str, int], ...]:
    return (
        (value.segment_id, canonical_driver_id(value.driver_id), value.fiscal_year),
    )


def _constraint_observation_keys(value: Any) -> tuple[tuple[str, str, int], ...]:
    if isinstance(value, OperatingDriverObservation):
        return _observation_keys(value)
    if isinstance(value, dict) and {"segment_id", "driver_id", "fiscal_year"}.issubset(
        value
    ):
        try:
            return _observation_keys(OperatingDriverObservation.model_validate(value))
        except ValueError:
            return ()
    return ()


def _guidance_observation_keys(
    value: ManagementGuidance,
) -> tuple[tuple[str, str, int], ...]:
    if value.fiscal_year is None:
        return ()
    metric = canonical_driver_id(_metric_key(value.metric))
    metric = {
        "revenue_growth": "growth",
        "operating_margin": "operating_margin",
        "capex": "capex",
    }.get(metric, metric)
    scope = value.scope.value
    segment_id = (
        canonical_operating_segment_id(value.segment_name)
        if scope == "segment" and value.segment_name
        else "company"
    )
    return ((segment_id, metric, value.fiscal_year),)


ReasoningCompiler = ForecastReasoningCompiler


__all__ = ["ForecastReasoningCompiler", "ReasoningCompiler"]
