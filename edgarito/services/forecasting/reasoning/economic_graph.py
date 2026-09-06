"""Opt-in, versioned reasoning for deterministic economic graphs.

This module is deliberately separate from the v1 driver-reasoning contracts.
The model may suggest values only for missing graph leaves; graph relationships,
aggregates, and all accounting arithmetic remain deterministic code.
"""

from __future__ import annotations

import datetime
import inspect
from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from edgarito.schemas.operating_graph import (
    EconomicEvaluationResult,
    EconomicModel,
    EconomicNode,
    EconomicNodeType,
    EconomicObservation,
    EconomicProvenance,
    EconomicRelationshipType,
    UnresolvedLeafRequirement,
)
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.forecasting.reasoning.contracts import (
    ForecastReasoningInput,
)
from edgarito.services.forecasting.reasoning.evidence import (
    EvidenceCatalog,
    EvidenceCatalogItem,
    build_evidence_catalog,
    canonical_json,
    compact_reasoning_input,
    compact_structured,
    content_hash,
)
from edgarito.services.openai import OpenAIClient
from edgarito.services.operating._graph.evaluator import EconomicGraphEvaluator

EconomicDecimal = Annotated[Decimal, WithJsonSchema({"type": "number"})]

ECONOMIC_GRAPH_PROMPT_VERSION = "forecast_reasoner_economic_graph_prompt_1"
ECONOMIC_GRAPH_SCHEMA_VERSION = "forecast_reasoner_economic_graph_response_1"
ECONOMIC_GRAPH_VALIDATOR_VERSION = "forecast_reasoner_economic_graph_validator_1"
ECONOMIC_GRAPH_CONTEXT_VERSION = "forecast_reasoner_economic_graph_context_1"

# Descriptive aliases make the versioned identity discoverable under the same
# vocabulary as ForecastReasoner v1 without reusing any v1 identity.
GRAPH_PROMPT_VERSION = ECONOMIC_GRAPH_PROMPT_VERSION
GRAPH_SCHEMA_VERSION = ECONOMIC_GRAPH_SCHEMA_VERSION
GRAPH_VALIDATOR_VERSION = ECONOMIC_GRAPH_VALIDATOR_VERSION
GRAPH_CONTEXT_VERSION = ECONOMIC_GRAPH_CONTEXT_VERSION


ECONOMIC_GRAPH_REASONER_INSTRUCTIONS = """You are EconomicGraphForecastReasoner v1. Return only the strict structured response schema.

The supplied economic model is a deterministic graph. You may propose values only
for explicit unresolved leaf requirements whose node is marked
forecast_assumption_allowed and whose node type is INPUT or COMPONENT. Never
propose a derived node, aggregate, revenue root, formula, method, or accounting
aggregate. Do not calculate or return any graph relationship output.

Every assumption must use the exact supplied fiscal years, exact leaf unit, and
low/base/high paths with low <= base <= high. Evidence-based assumptions must
cite one or more evidence catalog IDs. Model assumptions must be explicitly
marked model_assumption. Never browse, invent URLs, or cite an ID absent from the
as-of evidence catalog. Explain unsupported or unresolved leaves in unresolved.
"""


def _text(value: Any, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


def _tuple_records(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping, BaseModel)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


class EconomicGraphReasoningInput(BaseModel):
    """Frozen v1 context plus one deterministic economic graph evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    forecast_input: ForecastReasoningInput = Field(
        validation_alias=AliasChoices(
            "forecast_input", "reasoning_input", "input", "context"
        )
    )
    economic_model: EconomicModel = Field(
        validation_alias=AliasChoices("economic_model", "model", "graph")
    )
    evaluation: EconomicEvaluationResult = Field(
        validation_alias=AliasChoices(
            "evaluation", "economic_evaluation", "graph_evaluation"
        )
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_and_evaluate(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("Economic graph reasoning input must be a mapping")
        data = dict(value)
        forecast_input = data.get(
            "forecast_input",
            data.get("reasoning_input", data.get("input", data.get("context"))),
        )
        economic_model = data.get(
            "economic_model", data.get("model", data.get("graph"))
        )
        if forecast_input is None or economic_model is None:
            raise ValueError("Economic graph reasoning requires input and model")
        forecast_input = (
            forecast_input
            if isinstance(forecast_input, ForecastReasoningInput)
            else ForecastReasoningInput.model_validate(forecast_input)
        )
        economic_model = (
            economic_model
            if isinstance(economic_model, EconomicModel)
            else EconomicModel.model_validate(economic_model)
        )
        supplied = data.get("evaluation")
        if supplied is None:
            supplied = data.get("economic_evaluation", data.get("graph_evaluation"))
        if supplied is None:
            supplied = EconomicGraphEvaluator().evaluate(
                economic_model,
                forecast_input.forecast_years,
                as_of=forecast_input.as_of,
                fiscal_period=economic_model.fiscal_period,
            )
        elif not isinstance(supplied, EconomicEvaluationResult):
            supplied = EconomicEvaluationResult.model_validate(supplied)
        expected = EconomicGraphEvaluator().evaluate(
            economic_model,
            forecast_input.forecast_years,
            as_of=forecast_input.as_of,
            fiscal_period=economic_model.fiscal_period,
        )
        if supplied != expected:
            raise ValueError(
                "Economic graph evaluation must be the deterministic evaluation of the model"
            )
        data["forecast_input"] = forecast_input
        data["economic_model"] = economic_model
        data["evaluation"] = supplied
        return data

    @model_validator(mode="after")
    def validate_exact_evaluation(self) -> "EconomicGraphReasoningInput":
        expected_years = self.forecast_input.forecast_years
        if self.evaluation.target_years != expected_years:
            raise ValueError(
                "Economic graph evaluation years must exactly match forecast horizon"
            )
        if self.evaluation.fiscal_period != self.economic_model.fiscal_period:
            raise ValueError("Economic graph evaluation period must match model period")
        if self.evaluation.as_of != self.forecast_input.as_of:
            raise ValueError("Economic graph evaluation as_of must match forecast input")
        return self

    @property
    def input(self) -> ForecastReasoningInput:
        return self.forecast_input

    @property
    def reasoning_input(self) -> ForecastReasoningInput:
        return self.forecast_input

    @property
    def model(self) -> EconomicModel:
        return self.economic_model

    @property
    def graph(self) -> EconomicModel:
        return self.economic_model

    @property
    def unresolved_leaf_requirements(self) -> tuple[UnresolvedLeafRequirement, ...]:
        return self.evaluation.unresolved_leaf_requirements

    @property
    def unresolved_leaves(self) -> tuple[UnresolvedLeafRequirement, ...]:
        return self.unresolved_leaf_requirements

    @classmethod
    def from_model(
        cls,
        forecast_input: ForecastReasoningInput | Any,
        economic_model: EconomicModel | Any,
    ) -> "EconomicGraphReasoningInput":
        return cls(forecast_input=forecast_input, economic_model=economic_model)

    build = from_model


class EconomicLeafReasoningAssumption(BaseModel):
    """A strict proposal for one graph leaf across the complete horizon."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    fiscal_years: tuple[int, ...] = Field(
        validation_alias=AliasChoices("fiscal_years", "years", "forecast_years")
    )
    low: tuple[EconomicDecimal, ...]
    base: tuple[EconomicDecimal, ...]
    high: tuple[EconomicDecimal, ...]
    unit: str
    evidence_ids: tuple[str, ...] = ()
    rationale: str
    confidence: Literal["high", "medium", "low"]
    evidence_based: bool = False
    model_assumption: bool = False

    @field_validator("node_id", "unit", "rationale")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return _text(value, "Economic leaf assumption text")

    @field_validator("fiscal_years", mode="before")
    @classmethod
    def normalize_years(cls, value: Any) -> tuple[int, ...]:
        years = tuple(int(item) for item in (value or ()))
        if not years or tuple(sorted(years)) != years or len(years) != len(set(years)):
            raise ValueError("Economic leaf assumption years must be sorted and unique")
        return years

    @field_validator("low", "base", "high", mode="before")
    @classmethod
    def normalize_paths(cls, value: Any) -> tuple[Decimal, ...]:
        values = (
            (value,)
            if isinstance(value, (str, int, float, Decimal))
            else tuple(value or ())
        )
        return tuple(_decimal(item, "Economic leaf assumption path") for item in values)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_ids(cls, value: Any) -> tuple[str, ...]:
        values = (value,) if isinstance(value, str) else tuple(value or ())
        result = tuple(_text(item, "Evidence ID") for item in values)
        if len(result) != len(set(result)):
            raise ValueError("Economic leaf evidence IDs must be unique")
        if any(item.casefold().startswith(("http://", "https://")) for item in result):
            raise ValueError("URLs are not evidence citations; use catalog IDs")
        return result

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        normalized = str(value).strip().casefold()
        if normalized not in {"high", "medium", "low"}:
            raise ValueError("Economic leaf confidence must be high, medium, or low")
        return normalized

    @model_validator(mode="after")
    def validate_shape(self) -> "EconomicLeafReasoningAssumption":
        if not self.low or len(self.low) != len(self.fiscal_years):
            raise ValueError("Economic leaf low path must match fiscal horizon")
        if len(self.base) != len(self.fiscal_years) or len(self.high) != len(
            self.fiscal_years
        ):
            raise ValueError("Economic leaf paths must match fiscal horizon")
        for low, base, high in zip(self.low, self.base, self.high, strict=True):
            if not low.is_finite() or not base.is_finite() or not high.is_finite():
                raise ValueError("Economic leaf paths must be finite")
            if not low <= base <= high:
                raise ValueError("Economic leaf paths require low <= base <= high")
        if self.evidence_based == self.model_assumption:
            raise ValueError(
                "Economic leaf assumptions must be exactly evidence_based or model_assumption"
            )
        return self

    @property
    def target_key(self) -> str:
        return self.node_id


class EconomicGraphUnresolvedItem(BaseModel):
    """Provider response explanation for a leaf left unresolved."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str | None = None
    fiscal_years: tuple[int, ...] = Field(
        default=(), validation_alias=AliasChoices("fiscal_years", "years")
    )
    reason: str
    status: str = "unresolved"

    @field_validator("node_id")
    @classmethod
    def normalize_node(cls, value: str | None) -> str | None:
        return _text(value, "Unresolved economic node") if value is not None else None

    @field_validator("reason", "status")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        return _text(value, "Unresolved economic item text")

    @field_validator("fiscal_years", mode="before")
    @classmethod
    def normalize_unresolved_years(cls, value: Any) -> tuple[int, ...]:
        return tuple(int(item) for item in (value or ()))


class EconomicLeafReasoningResponse(BaseModel):
    """Only leaf assumptions, unresolved explanations, and warnings."""

    model_config = ConfigDict(extra="forbid")

    assumptions: tuple[EconomicLeafReasoningAssumption, ...] = ()
    unresolved_items: tuple[EconomicGraphUnresolvedItem, ...] = Field(
        default=(), validation_alias=AliasChoices("unresolved_items", "unresolved")
    )
    warnings: tuple[str, ...] = ()

    @field_validator("assumptions", "unresolved_items", mode="before")
    @classmethod
    def normalize_records(cls, value: Any) -> tuple[Any, ...]:
        return _tuple_records(value)

    @field_validator("warnings", mode="before")
    @classmethod
    def normalize_warnings(cls, value: Any) -> tuple[str, ...]:
        values = (value,) if isinstance(value, str) else tuple(value or ())
        return tuple(str(item).strip() for item in values if str(item).strip())


class EconomicGraphReasoningMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    reasoning_effort: str
    prompt_version: str
    schema_version: str
    validator_version: str
    context_version: str
    prompt_hash: str
    schema_hash: str
    validator_hash: str
    context_hash: str
    evidence_bundle_hash: str

    @property
    def prompt_identity(self) -> str:
        return self.prompt_hash

    @property
    def schema_identity(self) -> str:
        return self.schema_hash

    @property
    def validator_identity(self) -> str:
        return self.validator_hash


class EconomicGraphReasoningCacheIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    company_id: str
    as_of: datetime.date
    forecast_years: tuple[int, ...]
    economic_model_hash: str
    evaluation_hash: str
    evidence_bundle_hash: str
    model: str
    reasoning_effort: str
    prompt_version: str
    schema_version: str
    validator_version: str
    context_version: str
    prompt_hash: str
    schema_hash: str
    validator_hash: str
    context_hash: str

    @property
    def digest(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class EconomicGraphReasoningCacheEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    identity: EconomicGraphReasoningCacheIdentity
    response: EconomicLeafReasoningResponse
    metadata: EconomicGraphReasoningMetadata


class EconomicGraphReasoningCache:
    """Graph cache namespace sharing the v1 filesystem/cache storage."""

    prefix = "forecast_reasoning_economic_graph_v1"

    def __init__(
        self,
        cache: FileSystemCache | str | Path | Any | None = None,
        *,
        root_directory: str | Path | None = None,
    ) -> None:
        cache = cache if cache is not None else root_directory
        if isinstance(cache, EconomicGraphReasoningCache):
            self.cache = cache.cache
        elif isinstance(cache, FileSystemCache):
            self.cache = cache
        elif hasattr(cache, "cache") and isinstance(cache.cache, FileSystemCache):
            self.cache = cache.cache
        else:
            self.cache = FileSystemCache(cache or Path("cache"))

    def path(self, identity: EconomicGraphReasoningCacheIdentity | str) -> str:
        digest = identity if isinstance(identity, str) else identity.digest
        return f"{self.prefix}/{digest}.json"

    @staticmethod
    def identity(
        input_value: EconomicGraphReasoningInput | Any,
        *,
        model: str,
        reasoning_effort: str,
        prompt_version: str,
        schema_version: str,
        validator_version: str,
        context_version: str,
        prompt_hash: str,
        schema_hash: str,
        validator_hash: str,
        context_hash: str,
        evidence_bundle_hash: str | None = None,
    ) -> EconomicGraphReasoningCacheIdentity:
        input_value = _coerce_graph_input(input_value)
        catalog = build_evidence_catalog(input_value.forecast_input)
        return EconomicGraphReasoningCacheIdentity(
            company_id=input_value.forecast_input.company_id,
            as_of=input_value.forecast_input.as_of,
            forecast_years=input_value.forecast_input.forecast_years,
            economic_model_hash=content_hash(input_value.economic_model),
            evaluation_hash=content_hash(input_value.evaluation),
            evidence_bundle_hash=evidence_bundle_hash or catalog.bundle_hash,
            model=model,
            reasoning_effort=reasoning_effort,
            prompt_version=prompt_version,
            schema_version=schema_version,
            validator_version=validator_version,
            context_version=context_version,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            validator_hash=validator_hash,
            context_hash=context_hash,
        )

    def load(
        self, identity: EconomicGraphReasoningCacheIdentity | str
    ) -> EconomicGraphReasoningCacheEnvelope | None:
        raw = self.cache.read(self.path(identity))
        if raw is None:
            return None
        try:
            envelope = EconomicGraphReasoningCacheEnvelope.model_validate_json(raw)
        except Exception:
            return None
        if (
            isinstance(identity, EconomicGraphReasoningCacheIdentity)
            and envelope.identity != identity
        ):
            return None
        return envelope

    get = load

    def save(
        self,
        identity: EconomicGraphReasoningCacheIdentity,
        response: EconomicLeafReasoningResponse,
        metadata: EconomicGraphReasoningMetadata,
    ) -> EconomicGraphReasoningCacheEnvelope:
        envelope = EconomicGraphReasoningCacheEnvelope(
            identity=identity, response=response, metadata=metadata
        )
        self.cache.save(self.path(identity), envelope.model_dump_json())
        return envelope

    put = save


class EconomicGraphReasoningProposal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    response: EconomicLeafReasoningResponse
    catalog: EvidenceCatalog
    metadata: EconomicGraphReasoningMetadata
    cache_hit: bool = False
    cache_key: str

    @property
    def assumptions(self):
        return self.response.assumptions

    @property
    def unresolved_items(self):
        return self.response.unresolved_items

    @property
    def warnings(self):
        return self.response.warnings


class EconomicGraphReasoningValidationIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str | None = None
    fiscal_years: tuple[int, ...] = ()
    code: str
    reason: str

    @property
    def assumption_id(self) -> str | None:
        return self.node_id


class EconomicGraphReasoningValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted_assumptions: tuple[EconomicLeafReasoningAssumption, ...] = ()
    rejected_assumptions: tuple[EconomicGraphReasoningValidationIssue, ...] = ()
    unresolved_items: tuple[EconomicGraphUnresolvedItem, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def accepted(self):
        return self.accepted_assumptions

    @property
    def rejected(self):
        return self.rejected_assumptions

    @property
    def is_valid(self) -> bool:
        return not self.rejected_assumptions

    passed = is_valid


class EconomicGraphReasoningValidator:
    """Allow assumptions only at explicit, eligible unresolved leaves."""

    def validate(
        self,
        response: EconomicLeafReasoningResponse | Any,
        input_value: EconomicGraphReasoningInput | Any,
        catalog: EvidenceCatalog | None = None,
    ) -> EconomicGraphReasoningValidationResult:
        response = (
            response
            if isinstance(response, EconomicLeafReasoningResponse)
            else EconomicLeafReasoningResponse.model_validate(response)
        )
        input_value = _coerce_graph_input(input_value)
        catalog = catalog or build_evidence_catalog(input_value.forecast_input)
        node_by_id = input_value.economic_model.node_by_id
        requirements = input_value.unresolved_leaf_requirements
        requirement_nodes = {
            item.node_id
            for item in requirements
            if item.fiscal_year in input_value.forecast_input.forecast_years
        }
        duplicate_nodes = {
            node_id
            for node_id, values in _group_by_node(response.assumptions).items()
            if len(values) > 1
        }
        accepted: list[EconomicLeafReasoningAssumption] = []
        rejected: list[EconomicGraphReasoningValidationIssue] = []
        for assumption in response.assumptions:
            issues = self._issues(
                assumption,
                input_value,
                catalog,
                node_by_id,
                requirements,
                requirement_nodes,
            )
            if assumption.node_id in duplicate_nodes:
                issues.append(("DUPLICATE_TARGET", "Duplicate leaf assumption target"))
            if issues:
                rejected.extend(
                    EconomicGraphReasoningValidationIssue(
                        node_id=assumption.node_id,
                        fiscal_years=assumption.fiscal_years,
                        code=code,
                        reason=reason,
                    )
                    for code, reason in _unique_issues(issues)
                )
            else:
                accepted.append(assumption)

        warnings = list(response.warnings)
        warnings.extend(
            f"duplicate explicit evidence ID retained for audit: {item}"
            for item in catalog.duplicate_explicit_ids
        )
        warnings = list(dict.fromkeys(warnings))
        return EconomicGraphReasoningValidationResult(
            accepted_assumptions=tuple(accepted),
            rejected_assumptions=tuple(rejected),
            unresolved_items=response.unresolved_items,
            warnings=tuple(warnings),
        )

    validate_response = validate
    validate_proposal = validate
    post_validate = validate

    def _issues(
        self,
        assumption: EconomicLeafReasoningAssumption,
        input_value: EconomicGraphReasoningInput,
        catalog: EvidenceCatalog,
        node_by_id: Mapping[str, EconomicNode],
        requirements: tuple[UnresolvedLeafRequirement, ...],
        requirement_nodes: set[str],
    ) -> list[tuple[str, str]]:
        issues: list[tuple[str, str]] = []
        node = node_by_id.get(assumption.node_id)
        if node is None:
            issues.append(("UNKNOWN_NODE", f"Unknown economic node: {assumption.node_id}"))
            return issues
        if assumption.fiscal_years != input_value.forecast_input.forecast_years:
            issues.append(
                ("HORIZON_MISMATCH", "Leaf assumption years do not exactly match horizon")
            )
        if assumption.node_id not in requirement_nodes:
            issues.append(
                (
                    "NOT_UNRESOLVED_LEAF",
                    "Assumption target is not an explicit unresolved leaf requirement",
                )
            )
        else:
            for year in input_value.forecast_input.forecast_years:
                matches = tuple(
                    item
                    for item in requirements
                    if item.node_id == assumption.node_id
                    and item.fiscal_year == year
                    and (not item.path or assumption.node_id in item.path)
                )
                # A complete path may cover years already supplied by a manual
                # observation. Those cells are handled as compiler collisions.
                if year not in {
                    item.fiscal_year
                    for item in requirements
                    if item.node_id == assumption.node_id
                }:
                    continue
                if not matches:
                    issues.append(
                        (
                            "LEAF_PATH_MISMATCH",
                            f"Leaf {assumption.node_id} is not on the unresolved path for FY{year}",
                        )
                    )
        if not node.forecast_assumption_allowed:
            issues.append(
                ("ASSUMPTION_NOT_ALLOWED", "Economic node does not allow forecast assumptions")
            )
        if node.node_type not in {EconomicNodeType.INPUT, EconomicNodeType.COMPONENT}:
            issues.append(
                (
                    "UNSAFE_NODE_TYPE",
                    "Only INPUT or COMPONENT unresolved leaves may be assumed",
                )
            )
        producer = next(
            (
                relationship
                for relationship in input_value.economic_model.relationships
                if relationship.target == node.node_id
            ),
            None,
        )
        if (
            node.node_type == EconomicNodeType.COMPONENT
            and producer is not None
            and producer.relationship_type != EconomicRelationshipType.RESIDUAL
        ):
            issues.append(
                (
                    "DETERMINISTIC_OUTPUT_FORBIDDEN",
                    "A component with a deterministic producer cannot be independently assumed",
                )
            )
        if assumption.node_id == input_value.economic_model.revenue_root:
            issues.append(("REVENUE_ROOT_FORBIDDEN", "Revenue root assumptions are forbidden"))
        if assumption.unit != node.unit:
            issues.append(
                (
                    "UNIT_MISMATCH",
                    f"Leaf unit {assumption.unit!r} does not match node unit {node.unit!r}",
                )
            )
        issues.extend(self._citation_issues(assumption, node, input_value, catalog))
        return _unique_issues(issues)

    @staticmethod
    def _citation_issues(
        assumption: EconomicLeafReasoningAssumption,
        node: EconomicNode,
        input_value: EconomicGraphReasoningInput,
        catalog: EvidenceCatalog,
    ) -> list[tuple[str, str]]:
        if assumption.evidence_based and not assumption.evidence_ids:
            return [("MISSING_CITATION", "Evidence-based assumptions require catalog IDs")]
        issues: list[tuple[str, str]] = []
        for evidence_id in assumption.evidence_ids:
            item = catalog.get(evidence_id)
            if item is None:
                exclusion = catalog.exclusion(evidence_id)
                issues.append(
                    (
                        "EXCLUDED_EVIDENCE" if exclusion else "UNKNOWN_EVIDENCE",
                        exclusion.reason
                        if exclusion
                        else f"Evidence ID does not exist: {evidence_id}",
                    )
                )
                continue
            issues.extend(_evidence_scope_issues(item, node, input_value))
            if item.category != "MARKET":
                target = _key(node.metric)
                evidence_target = _key(item.metric or item.driver_id)
                if evidence_target and evidence_target != target:
                    issues.append(
                        (
                            "EVIDENCE_TARGET_MISMATCH",
                            f"Evidence {evidence_id} is for {evidence_target}, not {target}",
                        )
                    )
        return _unique_issues(issues)


class EconomicGraphReasoningCompilation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    observations: tuple[EconomicObservation, ...] = ()
    retained_ranges: dict[str, tuple[tuple[Decimal, Decimal, Decimal], ...]] = {}
    collisions: tuple[EconomicGraphReasoningValidationIssue, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def compiled_observations(self) -> tuple[EconomicObservation, ...]:
        return self.observations


class EconomicGraphReasoningCompiler:
    """Compile BASE leaf values while retaining LOW/HIGH for audit."""

    def compile(
        self,
        input_value: EconomicGraphReasoningInput | Any,
        validation: EconomicGraphReasoningValidationResult,
        *,
        metadata: EconomicGraphReasoningMetadata | Any | None = None,
    ) -> EconomicGraphReasoningCompilation:
        input_value = _coerce_graph_input(input_value)
        retained: dict[str, tuple[tuple[Decimal, Decimal, Decimal], ...]] = {}
        observations: list[EconomicObservation] = []
        collisions: list[EconomicGraphReasoningValidationIssue] = []
        warnings = list(validation.warnings)
        model = input_value.economic_model
        nodes = model.node_by_id
        for assumption in validation.accepted_assumptions:
            retained[assumption.node_id] = tuple(
                zip(assumption.low, assumption.base, assumption.high, strict=True)
            )
            node = nodes[assumption.node_id]
            for year, base in zip(assumption.fiscal_years, assumption.base, strict=True):
                existing = _existing_observation(model, node, year, input_value.forecast_input.as_of)
                if existing is not None:
                    code = (
                        "MANUAL_GRAPH_OBSERVATION_PRECEDENCE"
                        if existing.origin == "manual"
                        else "EXISTING_GRAPH_OBSERVATION_PRECEDENCE"
                    )
                    collisions.append(
                        EconomicGraphReasoningValidationIssue(
                            node_id=assumption.node_id,
                            fiscal_years=(year,),
                            code=code,
                            reason=(
                                "Existing graph observation wins; reasoned observation "
                                "was retained for audit only"
                            ),
                        )
                    )
                    continue
                observations.append(
                    EconomicObservation(
                        node_id=node.node_id,
                        fiscal_year=year,
                        fiscal_period=model.fiscal_period,
                        value=base,
                        unit=node.unit,
                        currency=node.currency,
                        origin=(
                            "reasoned_assumption"
                            if assumption.evidence_based
                            else "model_assumption"
                        ),
                        provenance=_economic_provenance(assumption, metadata),
                        available_on=input_value.forecast_input.as_of,
                        scope=node.scope,
                        scope_id=node.scope_id,
                    )
                )
        if collisions:
            warnings.extend(item.reason for item in collisions)
        return EconomicGraphReasoningCompilation(
            observations=tuple(observations),
            retained_ranges=retained,
            collisions=tuple(collisions),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    compile_response = compile
    build = compile


class EconomicGraphReasoningResult(BaseModel):
    """Immutable graph reasoning audit plus the post-compilation evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    proposal_identity: str
    audit_identity: tuple[str, ...] = ()
    proposal: EconomicGraphReasoningProposal
    accepted_assumptions: tuple[EconomicLeafReasoningAssumption, ...] = ()
    rejected_assumptions: tuple[EconomicGraphReasoningValidationIssue, ...] = ()
    unresolved_items: tuple[Any, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence_catalog: EvidenceCatalog
    metadata: EconomicGraphReasoningMetadata
    compiled_observations: tuple[EconomicObservation, ...] = ()
    retained_ranges: dict[str, tuple[tuple[Decimal, Decimal, Decimal], ...]] = {}
    collisions: tuple[EconomicGraphReasoningValidationIssue, ...] = ()
    evaluation: EconomicEvaluationResult
    cache_hit: bool = False

    @property
    def graph_result(self) -> EconomicEvaluationResult:
        return self.evaluation

    @property
    def graph_evaluation(self) -> EconomicEvaluationResult:
        return self.evaluation

    @property
    def unresolved(self) -> tuple[Any, ...]:
        return self.unresolved_items

    @property
    def accepted(self):
        return self.accepted_assumptions

    @property
    def rejected(self):
        return self.rejected_assumptions


class EconomicGraphForecastReasoner:
    """Reason, validate, compile, and re-evaluate one economic graph."""

    def __init__(
        self,
        client: OpenAIClient | Any | None = None,
        *,
        openai_client: OpenAIClient | Any | None = None,
        cache: EconomicGraphReasoningCache | Any | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        validator: EconomicGraphReasoningValidator | None = None,
        compiler: EconomicGraphReasoningCompiler | None = None,
        evaluator: EconomicGraphEvaluator | None = None,
    ) -> None:
        self.client = client or openai_client or OpenAIClient()
        self.model = model or getattr(self.client, "model", "")
        self.reasoning_effort = reasoning_effort or getattr(
            self.client, "reasoning_effort", "medium"
        )
        self.cache = (
            cache
            if isinstance(cache, EconomicGraphReasoningCache)
            else EconomicGraphReasoningCache(cache)
        )
        self.validator = validator or EconomicGraphReasoningValidator()
        self.compiler = compiler or EconomicGraphReasoningCompiler()
        self.evaluator = evaluator or EconomicGraphEvaluator()

    async def propose(
        self,
        input_value: EconomicGraphReasoningInput | Any,
        *,
        force_refresh: bool = False,
    ) -> EconomicGraphReasoningProposal:
        input_value = _coerce_graph_input(input_value)
        catalog = build_evidence_catalog(input_value.forecast_input)
        prompt = build_economic_graph_reasoning_prompt()
        schema_hash = content_hash(EconomicLeafReasoningResponse.model_json_schema())
        prompt_hash = content_hash(prompt)
        validator_hash = content_hash(ECONOMIC_GRAPH_VALIDATOR_VERSION)
        context_hash = content_hash(
            {
                "version": ECONOMIC_GRAPH_CONTEXT_VERSION,
                "company_name": input_value.forecast_input.company_name,
                "ticker": input_value.forecast_input.ticker,
                "unit": input_value.forecast_input.unit,
            }
        )
        metadata = EconomicGraphReasoningMetadata(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            prompt_version=ECONOMIC_GRAPH_PROMPT_VERSION,
            schema_version=ECONOMIC_GRAPH_SCHEMA_VERSION,
            validator_version=ECONOMIC_GRAPH_VALIDATOR_VERSION,
            context_version=ECONOMIC_GRAPH_CONTEXT_VERSION,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            validator_hash=validator_hash,
            context_hash=context_hash,
            evidence_bundle_hash=catalog.bundle_hash,
        )
        identity = EconomicGraphReasoningCacheIdentity(
            company_id=input_value.forecast_input.company_id,
            as_of=input_value.forecast_input.as_of,
            forecast_years=input_value.forecast_input.forecast_years,
            economic_model_hash=content_hash(input_value.economic_model),
            evaluation_hash=content_hash(input_value.evaluation),
            evidence_bundle_hash=catalog.bundle_hash,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            prompt_version=ECONOMIC_GRAPH_PROMPT_VERSION,
            schema_version=ECONOMIC_GRAPH_SCHEMA_VERSION,
            validator_version=ECONOMIC_GRAPH_VALIDATOR_VERSION,
            context_version=ECONOMIC_GRAPH_CONTEXT_VERSION,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            validator_hash=validator_hash,
            context_hash=context_hash,
        )
        cache_key = identity.digest
        if not force_refresh:
            envelope = self.cache.load(identity)
            if envelope is not None:
                return EconomicGraphReasoningProposal(
                    response=envelope.response,
                    catalog=catalog,
                    metadata=envelope.metadata,
                    cache_hit=True,
                    cache_key=cache_key,
                )
        raw = self.client.extract_structured(
            instructions=prompt,
            content=build_economic_graph_reasoning_content(input_value, catalog),
            response_model=EconomicLeafReasoningResponse,
            model=self.model or None,
        )
        if inspect.isawaitable(raw):
            raw = await raw
        response = (
            raw
            if isinstance(raw, EconomicLeafReasoningResponse)
            else EconomicLeafReasoningResponse.model_validate(raw)
        )
        self.cache.save(identity, response, metadata)
        return EconomicGraphReasoningProposal(
            response=response,
            catalog=catalog,
            metadata=metadata,
            cache_hit=False,
            cache_key=cache_key,
        )

    async def reason(
        self,
        input_value: EconomicGraphReasoningInput | Any,
        *,
        force_refresh: bool = False,
    ) -> EconomicGraphReasoningResult:
        input_value = _coerce_graph_input(input_value)
        proposal = await self.propose(input_value, force_refresh=force_refresh)
        validation = self.validator.validate(
            proposal.response, input_value, proposal.catalog
        )
        compilation = self.compiler.compile(
            input_value, validation, metadata=proposal.metadata
        )
        evaluated_model = input_value.economic_model.model_copy(
            update={
                "observations": (
                    *input_value.economic_model.observations,
                    *compilation.observations,
                )
            }
        )
        evaluation = self.evaluator.evaluate(
            evaluated_model,
            input_value.forecast_input.forecast_years,
            as_of=input_value.forecast_input.as_of,
            fiscal_period=input_value.economic_model.fiscal_period,
        )
        unresolved = _merge_unresolved(
            validation.unresolved_items, evaluation.unresolved_leaf_requirements
        )
        rejected = tuple(validation.rejected_assumptions) + tuple(
            compilation.collisions
        )
        warnings = tuple(
            dict.fromkeys(
                (
                    *proposal.warnings,
                    *validation.warnings,
                    *compilation.warnings,
                    *(
                        item.message
                        for item in evaluation.diagnostics.diagnostic_messages
                        if item.code != "unknown_materiality"
                    ),
                )
            )
        )
        audit_identity = (
            f"proposal_cache_key={proposal.cache_key}",
            f"economic_model_hash={content_hash(input_value.economic_model)}",
            f"evaluation_hash={content_hash(evaluation)}",
            f"model={proposal.metadata.model}",
            f"prompt_version={proposal.metadata.prompt_version}",
            f"schema_version={proposal.metadata.schema_version}",
            f"validator_version={proposal.metadata.validator_version}",
            "execution=deterministic_economic_graph",
        )
        return EconomicGraphReasoningResult(
            proposal_identity=proposal.cache_key,
            audit_identity=audit_identity,
            proposal=proposal,
            accepted_assumptions=validation.accepted_assumptions,
            rejected_assumptions=rejected,
            unresolved_items=unresolved,
            warnings=warnings,
            evidence_catalog=proposal.catalog,
            metadata=proposal.metadata,
            compiled_observations=compilation.observations,
            retained_ranges=compilation.retained_ranges,
            collisions=compilation.collisions,
            evaluation=evaluation,
            cache_hit=proposal.cache_hit,
        )

    async def reason_economic_model(
        self,
        input_value: EconomicGraphReasoningInput | Any,
        economic_model: EconomicModel | Any | None = None,
        *,
        evaluation: EconomicEvaluationResult | Any | None = None,
        force_refresh: bool = False,
    ):
        if not isinstance(input_value, EconomicGraphReasoningInput):
            if economic_model is None:
                input_value = EconomicGraphReasoningInput.model_validate(input_value)
            else:
                input_value = EconomicGraphReasoningInput(
                    forecast_input=input_value,
                    economic_model=economic_model,
                    evaluation=evaluation,
                )
        return await self.reason(input_value, force_refresh=force_refresh)

    forecast = reason
    run = reason
    areason = reason


def build_economic_graph_reasoning_prompt() -> str:
    return ECONOMIC_GRAPH_REASONER_INSTRUCTIONS


def build_economic_graph_reasoning_content(
    input_value: EconomicGraphReasoningInput | Any,
    catalog: EvidenceCatalog | None = None,
) -> str:
    input_value = _coerce_graph_input(input_value)
    catalog = catalog or build_evidence_catalog(input_value.forecast_input)
    return canonical_json(
        {
            "context_version": ECONOMIC_GRAPH_CONTEXT_VERSION,
            "input": compact_reasoning_input(input_value.forecast_input),
            "economic_model": compact_structured(input_value.economic_model),
            "evaluation": compact_structured(input_value.evaluation),
            "unresolved_leaf_requirements": compact_structured(
                input_value.unresolved_leaf_requirements
            ),
            "evidence_catalog": compact_structured(catalog.items),
        }
    )


def _coerce_graph_input(value: EconomicGraphReasoningInput | Any) -> EconomicGraphReasoningInput:
    return value if isinstance(value, EconomicGraphReasoningInput) else EconomicGraphReasoningInput.model_validate(value)


def _group_by_node(
    assumptions: Iterable[EconomicLeafReasoningAssumption],
) -> dict[str, list[EconomicLeafReasoningAssumption]]:
    result: dict[str, list[EconomicLeafReasoningAssumption]] = defaultdict(list)
    for item in assumptions:
        result[item.node_id].append(item)
    return result


def _existing_observation(
    model: EconomicModel,
    node: EconomicNode,
    year: int,
    as_of: datetime.date,
) -> EconomicObservation | None:
    candidates = [
        item
        for item in model.observations
        if item.node_id == node.node_id
        and item.fiscal_year == year
        and item.fiscal_period == model.fiscal_period
        and item.unit == node.unit
        and item.currency == node.currency
        and (item.available_on is None or item.available_on <= as_of)
    ]
    return max(
        candidates,
        key=lambda item: (item.available_on or datetime.date.min, item.origin),
        default=None,
    )


def _economic_provenance(
    assumption: EconomicLeafReasoningAssumption,
    metadata: Any | None,
) -> EconomicProvenance:
    origin = "reasoned_assumption" if assumption.evidence_based else "model_assumption"
    identity = ";".join(
        item
        for item in (
            f"assumption_node={assumption.node_id}",
            f"model={getattr(metadata, 'model', None) or 'unknown'}",
            f"prompt_hash={getattr(metadata, 'prompt_hash', None) or 'unknown'}",
            f"prompt_version={getattr(metadata, 'prompt_version', None) or 'unknown'}",
            f"schema_version={getattr(metadata, 'schema_version', None) or 'unknown'}",
            f"validator_version={getattr(metadata, 'validator_version', None) or 'unknown'}",
        )
        if item
    )
    return EconomicProvenance(
        source="ForecastReasoner",
        origin=origin,
        reference=identity,
        evidence_ids=assumption.evidence_ids,
        methodology="economic_graph_leaf_reasoning",
    )


def _evidence_scope_issues(
    item: EvidenceCatalogItem,
    node: EconomicNode,
    input_value: EconomicGraphReasoningInput,
) -> list[tuple[str, str]]:
    context = item.context_map
    issues: list[tuple[str, str]] = []
    evidence_scope = (item.scope or context.get("scope") or "").casefold()
    evidence_scope_id = item.scope_id or context.get("scope_id")
    node_is_company = node.scope.casefold() in {"company", "consolidated"}
    evidence_is_segment = evidence_scope == "segment" or (
        evidence_scope_id is not None
        and str(evidence_scope_id).casefold() not in {"company", "consolidated"}
    )
    if node_is_company and evidence_is_segment and not (item.is_total and item.exhaustive):
        issues.append(
            (
                "EVIDENCE_SCOPE_MISMATCH",
                f"Segment-specific evidence {item.evidence_id} cannot support a consolidated leaf",
            )
        )
    if not node_is_company and evidence_scope in {"company", "consolidated"}:
        issues.append(
            (
                "EVIDENCE_SCOPE_MISMATCH",
                f"Company-level evidence {item.evidence_id} cannot support a segment leaf",
            )
        )
    if (
        not node_is_company
        and evidence_scope_id
        and evidence_scope_id != node.scope_id
    ):
        issues.append(
            (
                "EVIDENCE_SCOPE_MISMATCH",
                f"Evidence {item.evidence_id} belongs to another scope",
            )
        )
    if item.source_date and item.source_date > input_value.forecast_input.as_of:
        issues.append(
            (
                "EVIDENCE_AS_OF_MISMATCH",
                f"Evidence {item.evidence_id} was not available as of input date",
            )
        )
    evidence_currency = _currency_codes(item.currency) or _currency_codes(item.unit)
    if node.currency and evidence_currency and node.currency not in evidence_currency:
        issues.append(
            (
                "EVIDENCE_CURRENCY_MISMATCH",
                f"Evidence {item.evidence_id} has incompatible currency",
            )
        )
    if item.unit and item.category != "MARKET" and item.unit not in {"unit", "unspecified"} and item.unit != node.unit:
        # Exact graph units are intentionally conservative.  A research-market
        # record is handled by its target/evidence semantics instead.
        issues.append(
            (
                "EVIDENCE_UNIT_MISMATCH",
                f"Evidence {item.evidence_id} has an incompatible unit",
            )
        )
    return issues


def _currency_codes(value: str | None) -> set[str]:
    if not value:
        return set()
    import re

    return set(re.findall(r"(?<![A-Za-z])([A-Za-z]{3})(?![A-Za-z])", value.upper()))


def _key(value: Any) -> str:
    return str(getattr(value, "value", value or "")).strip().casefold().replace("-", "_").replace(" ", "_")


def _unique_issues(
    issues: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    return list(dict.fromkeys(issues))


def _merge_unresolved(
    response_items: Iterable[EconomicGraphUnresolvedItem],
    requirements: Iterable[UnresolvedLeafRequirement],
) -> tuple[Any, ...]:
    result: list[Any] = []
    seen: set[str] = set()
    for item in (*tuple(response_items), *tuple(requirements)):
        key = canonical_json(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return tuple(result)


# Compatibility aliases for callers that use the shorter graph vocabulary.
EconomicGraphReasoningAssumption = EconomicLeafReasoningAssumption
EconomicGraphLeafReasoningAssumption = EconomicLeafReasoningAssumption
EconomicGraphReasoningResponse = EconomicLeafReasoningResponse
EconomicGraphLeafReasoningResponse = EconomicLeafReasoningResponse
EconomicGraphReasoner = EconomicGraphForecastReasoner
EconomicGraphValidator = EconomicGraphReasoningValidator
EconomicGraphCompiler = EconomicGraphReasoningCompiler
EconomicGraphReasoningInputValidation = EconomicGraphReasoningValidationResult
EconomicGraphReasoningUnresolvedItem = EconomicGraphUnresolvedItem
build_graph_reasoning_prompt = build_economic_graph_reasoning_prompt
build_graph_reasoning_content = build_economic_graph_reasoning_content


__all__ = [
    "ECONOMIC_GRAPH_CONTEXT_VERSION",
    "ECONOMIC_GRAPH_PROMPT_VERSION",
    "ECONOMIC_GRAPH_REASONER_INSTRUCTIONS",
    "ECONOMIC_GRAPH_SCHEMA_VERSION",
    "ECONOMIC_GRAPH_VALIDATOR_VERSION",
    "GRAPH_CONTEXT_VERSION",
    "GRAPH_PROMPT_VERSION",
    "GRAPH_SCHEMA_VERSION",
    "GRAPH_VALIDATOR_VERSION",
    "EconomicGraphReasoningAssumption",
    "EconomicGraphLeafReasoningAssumption",
    "EconomicGraphReasoningCache",
    "EconomicGraphReasoningCacheEnvelope",
    "EconomicGraphReasoningCacheIdentity",
    "EconomicGraphReasoningCompilation",
    "EconomicGraphReasoningCompiler",
    "EconomicGraphReasoningInput",
    "EconomicGraphReasoningMetadata",
    "EconomicGraphReasoningProposal",
    "EconomicGraphReasoningResponse",
    "EconomicGraphLeafReasoningResponse",
    "EconomicGraphReasoningResult",
    "EconomicGraphReasoningValidationIssue",
    "EconomicGraphReasoningValidationResult",
    "EconomicGraphReasoningUnresolvedItem",
    "EconomicGraphReasoner",
    "EconomicGraphForecastReasoner",
    "EconomicGraphReasoningValidator",
    "EconomicGraphUnresolvedItem",
    "EconomicLeafReasoningAssumption",
    "EconomicLeafReasoningResponse",
    "build_economic_graph_reasoning_content",
    "build_economic_graph_reasoning_prompt",
    "build_graph_reasoning_content",
    "build_graph_reasoning_prompt",
]
