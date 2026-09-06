"""Provider-neutral contracts for an economic operating graph.

The graph is deliberately small and mechanical.  Providers may populate these
contracts, but neither the contracts nor the graph evaluator know anything
about a provider's vocabulary.  A graph node describes one economic quantity;
relationships describe how quantities are reconstructed for a fiscal period.
"""

from __future__ import annotations

import datetime as _datetime
from decimal import Decimal
from enum import Enum

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class _StringEnum(str, Enum):
    """A string enum that accepts case and separator variations."""

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
            for member in cls:
                if member.value == normalized or member.name.casefold() == normalized:
                    return member
        return None


class EconomicNodeType(_StringEnum):
    INPUT = "input"
    DERIVED = "derived"
    COMPONENT = "component"
    AGGREGATE = "aggregate"


class EconomicMateriality(_StringEnum):
    MATERIAL = "material"
    IMMATERIAL = "immaterial"
    UNKNOWN = "unknown"


class EconomicComponentRole(_StringEnum):
    STANDARD = "standard"
    ADDITIVE = "additive"
    CONTRA_REVENUE = "contra_revenue"


class EconomicRelationshipType(_StringEnum):
    IDENTITY = "identity"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    RATIO = "ratio"
    LAG = "lag"
    GROWTH = "growth"
    WEIGHTED_SUM = "weighted_sum"
    RESIDUAL = "residual"


class EconomicUnitKind(_StringEnum):
    MONETARY = "monetary"
    COUNT = "count"
    RATE = "rate"
    MONETARY_PER_UNIT = "monetary_per_unit"
    OTHER = "other"


# Short names are useful to callers defining a graph interactively.
NodeType = EconomicNodeType
Materiality = EconomicMateriality
ComponentRole = EconomicComponentRole
RelationshipType = EconomicRelationshipType
UnitKind = EconomicUnitKind


def _required_text(value: str, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _finite(value: Decimal | int | float, label: str) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


def _currency(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if len(normalized) != 3 or not normalized.isalpha():
        raise ValueError("Currency must be a three-letter code")
    return normalized


class EconomicProvenance(BaseModel):
    """A provider-neutral source pointer retained in graph audits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str | None = None
    origin: str | None = None
    reference: str | None = None
    available_on: _datetime.date | None = Field(
        default=None, validation_alias=AliasChoices("available_on", "filing_date")
    )
    evidence_ids: tuple[str, ...] = ()
    methodology: str | None = None

    @field_validator("source", "origin", "reference", "methodology")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("evidence_ids")
    @classmethod
    def normalize_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(item for item in (_optional_text(v) for v in value) if item)
        )


Provenance = EconomicProvenance


class EconomicSourceEdge(BaseModel):
    """A typed, finite reference from a relationship to one source node."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    node_id: str = Field(validation_alias=AliasChoices("node_id", "source_id", "id"))
    # A sign is intentionally separate from a weight.  This prevents a
    # contra-revenue component from being silently treated as a positive sum.
    sign: Decimal = Decimal(1)
    weight: Decimal = Decimal(1)
    fiscal_lag: int = Field(default=0, ge=0)

    @field_validator("node_id")
    @classmethod
    def normalize_node_id(cls, value: str) -> str:
        return _required_text(value, "Source node ID")

    @field_validator("sign", "weight")
    @classmethod
    def validate_finite(cls, value: Decimal) -> Decimal:
        return _finite(value, "Source edge sign/weight")

    @model_validator(mode="after")
    def validate_sign(self) -> "EconomicSourceEdge":
        if self.sign == 0:
            raise ValueError("Source edge sign cannot be zero")
        return self

    @property
    def source_node_id(self) -> str:
        return self.node_id


SourceEdge = EconomicSourceEdge


class EconomicNode(BaseModel):
    """Stable metadata for one economic quantity in the graph."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    node_id: str = Field(validation_alias=AliasChoices("node_id", "stable_id", "id"))
    node_type: EconomicNodeType = Field(
        validation_alias=AliasChoices("node_type", "type")
    )
    scope: str = "consolidated"
    scope_id: str = "company"
    metric: str
    unit: str
    currency: str | None = None
    unit_kind: EconomicUnitKind = Field(
        validation_alias=AliasChoices("unit_kind", "kind")
    )
    denominator_unit: str | None = None
    provenance: EconomicProvenance | str | None = None
    confidence: str = "medium"
    materiality: EconomicMateriality = EconomicMateriality.UNKNOWN
    component_role: EconomicComponentRole = Field(
        default=EconomicComponentRole.STANDARD,
        validation_alias=AliasChoices("component_role", "role"),
    )
    forecast_assumption_allowed: bool = False

    @field_validator("node_id", "scope", "scope_id", "metric", "unit")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        return _required_text(value, "Economic node text")

    @field_validator("denominator_unit")
    @classmethod
    def normalize_denominator(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("currency")
    @classmethod
    def normalize_node_currency(cls, value: str | None) -> str | None:
        return _currency(value)

    @field_validator("confidence")
    @classmethod
    def normalize_confidence(cls, value: str) -> str:
        normalized = str(value).strip().casefold()
        if normalized not in {"high", "medium", "low"}:
            raise ValueError("Economic node confidence must be high, medium, or low")
        return normalized

    @model_validator(mode="after")
    def validate_unit_shape(self) -> "EconomicNode":
        if self.unit_kind == EconomicUnitKind.MONETARY_PER_UNIT:
            if not self.denominator_unit:
                raise ValueError("Monetary-per-unit nodes require denominator_unit")
        elif self.denominator_unit is not None:
            raise ValueError(
                "Only monetary-per-unit nodes may declare denominator_unit"
            )
        return self

    @property
    def stable_id(self) -> str:
        return self.node_id

    @property
    def id(self) -> str:
        return self.node_id


class EconomicRelationship(BaseModel):
    """One deterministic producer relationship for a target node."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    target: str
    relationship_type: EconomicRelationshipType = Field(
        validation_alias=AliasChoices("relationship_type", "type", "kind")
    )
    sources: tuple[EconomicSourceEdge, ...] = Field(min_length=1)
    relationship_id: str | None = Field(
        default=None, validation_alias=AliasChoices("relationship_id", "id")
    )
    provenance: EconomicProvenance | str | None = None
    confidence: str = "medium"
    fiscal_period: str = "FY"
    # A ratio may be declared as a historical-only way to derive an otherwise
    # forecast-assumable INPUT parameter.  The evaluator never carries this
    # relationship forward into forecast periods; the parameter itself remains
    # the assumption leaf when its reported sources are unavailable.
    historical_parameter_derivation: bool = False

    @field_validator("target")
    @classmethod
    def normalize_target(cls, value: str) -> str:
        return _required_text(value, "Relationship target")

    @field_validator("relationship_id")
    @classmethod
    def normalize_relationship_id(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("confidence")
    @classmethod
    def normalize_relationship_confidence(cls, value: str) -> str:
        normalized = str(value).strip().casefold()
        if normalized not in {"high", "medium", "low"}:
            raise ValueError(
                "Economic relationship confidence must be high, medium, or low"
            )
        return normalized

    @field_validator("fiscal_period")
    @classmethod
    def normalize_period(cls, value: str) -> str:
        normalized = str(value).strip().upper()
        if normalized in {"FY", "FQ", "YTD", "LTM"}:
            return normalized
        raise ValueError("Economic relationship fiscal_period is not supported")

    @property
    def type(self) -> EconomicRelationshipType:
        return self.relationship_type


Relationship = EconomicRelationship


class EconomicObservation(BaseModel):
    """A point-in-time observation for a graph node."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    node_id: str = Field(validation_alias=AliasChoices("node_id", "node"))
    fiscal_year: int = Field(
        validation_alias=AliasChoices("fiscal_year", "year"), ge=1900, le=3000
    )
    fiscal_period: str = Field(
        default="FY", validation_alias=AliasChoices("fiscal_period", "period")
    )
    value: Decimal
    unit: str
    currency: str | None = None
    origin: str = "reported"
    provenance: EconomicProvenance | str | None = None
    available_on: _datetime.date | None = None
    # Optional declarations make scope joins explicit when observations are
    # assembled from more than one source.  The node remains authoritative.
    scope: str | None = None
    scope_id: str | None = None

    @field_validator("node_id", "unit")
    @classmethod
    def normalize_observation_text(cls, value: str) -> str:
        return _required_text(value, "Economic observation text")

    @field_validator("fiscal_period")
    @classmethod
    def normalize_observation_period(cls, value: str) -> str:
        normalized = str(value).strip().upper()
        if normalized in {"FY", "FQ", "YTD", "LTM"}:
            return normalized
        raise ValueError("Economic observation fiscal_period is not supported")

    @field_validator("value")
    @classmethod
    def validate_observation_value(cls, value: Decimal) -> Decimal:
        return _finite(value, "Economic observation value")

    @field_validator("currency")
    @classmethod
    def normalize_observation_currency(cls, value: str | None) -> str | None:
        return _currency(value)

    @field_validator("origin")
    @classmethod
    def normalize_origin(cls, value: str) -> str:
        return _required_text(value, "Economic observation origin").casefold()

    @field_validator("scope", "scope_id")
    @classmethod
    def normalize_optional_scope(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @property
    def year(self) -> int:
        return self.fiscal_year

    @property
    def period(self) -> str:
        return self.fiscal_period

    @property
    def node(self) -> str:
        return self.node_id


Observation = EconomicObservation


class EconomicModel(BaseModel):
    """A provider-neutral economic DAG definition."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    nodes: tuple[EconomicNode, ...] = ()
    relationships: tuple[EconomicRelationship, ...] = ()
    observations: tuple[EconomicObservation, ...] = ()
    revenue_root: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "revenue_root",
            "revenue_root_node_id",
            "revenue_root_id",
            "root_node_id",
        ),
    )
    business_roots: tuple[str, ...] = ()
    fiscal_period: str = "FY"

    @field_validator("revenue_root")
    @classmethod
    def normalize_revenue_root(cls, value: str | None) -> str | None:
        return _optional_text(value)

    @field_validator("business_roots")
    @classmethod
    def normalize_business_roots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        roots = tuple(_required_text(item, "Business root") for item in value)
        if len(roots) != len(set(roots)):
            raise ValueError("Business roots must be unique")
        return roots

    @field_validator("fiscal_period")
    @classmethod
    def normalize_model_period(cls, value: str) -> str:
        normalized = str(value).strip().upper()
        if normalized in {"FY", "FQ", "YTD", "LTM"}:
            return normalized
        raise ValueError("Economic model fiscal_period is not supported")

    @property
    def revenue_root_id(self) -> str | None:
        return self.revenue_root

    @property
    def root_node_id(self) -> str | None:
        return self.revenue_root

    @property
    def node_by_id(self) -> dict[str, EconomicNode]:
        return {node.node_id: node for node in self.nodes}


EconomicGraph = EconomicModel
OperatingEconomicGraph = EconomicModel
OperatingGraphModel = EconomicModel
EconomicGraphModel = EconomicModel
EconomicDAG = EconomicModel


class EconomicValue(BaseModel):
    """One evaluated node-period cell."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    fiscal_year: int
    fiscal_period: str = "FY"
    value: Decimal | None = None
    unit: str
    currency: str | None = None
    available: bool = False
    origin: str = "unavailable"
    provenance: EconomicProvenance | str | None = None
    provenance_chain: tuple[str, ...] = ()
    dependency_chain: tuple[str, ...] = ()
    leaf_nodes: tuple[str, ...] = ()
    unresolved_reasons: tuple[str, ...] = ()

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: Decimal | None) -> Decimal | None:
        return None if value is None else _finite(value, "Evaluated value")

    @property
    def is_available(self) -> bool:
        return self.available and self.value is not None


class DependencyAudit(BaseModel):
    """Backward dependency and provenance trace for one result cell."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    fiscal_year: int
    fiscal_period: str = "FY"
    available: bool = False
    dependency_chain: tuple[str, ...] = ()
    leaf_nodes: tuple[str, ...] = ()
    provenance_chain: tuple[str, ...] = ()
    unresolved_reasons: tuple[str, ...] = ()


class UnresolvedLeafRequirement(BaseModel):
    """A missing leaf which prevented a deterministic result cell."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    fiscal_year: int
    fiscal_period: str = "FY"
    reason: str
    path: tuple[str, ...] = ()
    # These fields are additive metadata.  Defaults retain compatibility with
    # callers that construct the original four-field requirement directly;
    # evaluator-created requirements always populate them from the leaf node.
    scope: str = "consolidated"
    scope_id: str = "company"
    metric: str = ""
    unit: str = ""
    currency: str | None = None
    materiality: EconomicMateriality = EconomicMateriality.UNKNOWN
    required_by_relationship_ids: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices(
            "required_by_relationship_ids",
            "required_by_relationships",
            "required_by",
            "relationship_ids",
        ),
    )

    @property
    def node(self) -> str:
        return self.node_id

    @property
    def id(self) -> str:
        return self.node_id

    @property
    def year(self) -> int:
        return self.fiscal_year

    @property
    def period(self) -> str:
        return self.fiscal_period

    @property
    def required_by(self) -> tuple[str, ...]:
        """Compatibility-friendly shorthand for relationship provenance."""

        return self.required_by_relationship_ids


class GraphDiagnostic(BaseModel):
    """Non-fatal evaluation diagnostic."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    message: str
    severity: str = "warning"
    node_id: str | None = None
    fiscal_year: int | None = None


class GraphDiagnostics(BaseModel):
    """Coverage and reconciliation diagnostics for an evaluation run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_coverage: Decimal | None = None
    aggregate_coverage: Decimal | None = None
    component_coverage_by_node: dict[str, Decimal] = Field(default_factory=dict)
    aggregate_coverage_by_node: dict[str, Decimal] = Field(default_factory=dict)
    unresolved_count: int = 0
    historical_reconstructable_share: Decimal | None = None
    reconciliation_error: Decimal | None = None
    reconciliation_error_by_year: dict[int, Decimal] = Field(default_factory=dict)
    forecastable_share: Decimal | None = None
    diagnostic_messages: tuple[GraphDiagnostic, ...] = ()


class EconomicEvaluationResult(BaseModel):
    """Deterministic graph values plus their audit trails."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_years: tuple[int, ...] = ()
    fiscal_period: str = "FY"
    as_of: _datetime.date | None = None
    # Nested maps keep the common ``values[node_id][year]`` access compact.
    # ``cells`` and ``value()`` retain unit, origin, and provenance details.
    values: dict[str, dict[int, Decimal | None]] = Field(default_factory=dict)
    cells: tuple[EconomicValue, ...] = ()
    dependency_audits: tuple[DependencyAudit, ...] = ()
    unresolved_leaf_requirements: tuple[UnresolvedLeafRequirement, ...] = ()
    diagnostics: GraphDiagnostics = Field(default_factory=GraphDiagnostics)

    def value(
        self, node_id: str, fiscal_year: int, fiscal_period: str | None = None
    ) -> Decimal | None:
        period = fiscal_period or self.fiscal_period
        if period == self.fiscal_period:
            return self.values.get(node_id, {}).get(fiscal_year)
        for cell in self.cells:
            if (
                cell.node_id == node_id
                and cell.fiscal_year == fiscal_year
                and cell.fiscal_period == period
            ):
                return cell.value
        return None

    def cell(
        self, node_id: str, fiscal_year: int, fiscal_period: str | None = None
    ) -> EconomicValue | None:
        period = fiscal_period or self.fiscal_period
        return next(
            (
                item
                for item in self.cells
                if item.node_id == node_id
                and item.fiscal_year == fiscal_year
                and item.fiscal_period == period
            ),
            None,
        )

    @property
    def evaluated_values(self) -> tuple[EconomicValue, ...]:
        return self.cells

    def dependency_audit(
        self, node_id: str, fiscal_year: int, fiscal_period: str | None = None
    ) -> DependencyAudit | None:
        period = fiscal_period or self.fiscal_period
        return next(
            (
                item
                for item in self.dependency_audits
                if item.node_id == node_id
                and item.fiscal_year == fiscal_year
                and item.fiscal_period == period
            ),
            None,
        )

    @property
    def unresolved_leaves(self) -> tuple[UnresolvedLeafRequirement, ...]:
        return self.unresolved_leaf_requirements

    @property
    def diagnostic(self) -> GraphDiagnostics:
        return self.diagnostics


# Descriptive aliases keep the contract discoverable without creating multiple
# pydantic models that could drift apart.
GraphNode = EconomicNode
GraphRelationship = EconomicRelationship
GraphObservation = EconomicObservation
OperatingGraphNode = EconomicNode
OperatingGraphRelationship = EconomicRelationship
OperatingGraphObservation = EconomicObservation
GraphSourceEdge = EconomicSourceEdge
EconomicGraphNode = EconomicNode
EconomicGraphRelationship = EconomicRelationship
EconomicGraphSourceEdge = EconomicSourceEdge
EconomicGraphObservation = EconomicObservation
EconomicGraphModel = EconomicModel
EconomicGraphResult = EconomicEvaluationResult
EconomicGraphValue = EconomicValue
EconomicGraphNodeType = EconomicNodeType
EconomicGraphMateriality = EconomicMateriality
EconomicGraphComponentRole = EconomicComponentRole
EconomicGraphRelationshipType = EconomicRelationshipType
EconomicGraphUnitKind = EconomicUnitKind


__all__ = [
    "ComponentRole",
    "DependencyAudit",
    "EconomicComponentRole",
    "EconomicDAG",
    "EconomicEvaluationResult",
    "EconomicGraph",
    "EconomicGraphComponentRole",
    "EconomicGraphMateriality",
    "EconomicGraphModel",
    "EconomicGraphNode",
    "EconomicGraphNodeType",
    "EconomicGraphObservation",
    "EconomicGraphRelationship",
    "EconomicGraphRelationshipType",
    "EconomicGraphResult",
    "EconomicGraphSourceEdge",
    "EconomicGraphUnitKind",
    "EconomicGraphValue",
    "EconomicMateriality",
    "EconomicModel",
    "EconomicNode",
    "EconomicNodeType",
    "EconomicObservation",
    "EconomicProvenance",
    "EconomicRelationship",
    "EconomicRelationshipType",
    "EconomicSourceEdge",
    "EconomicUnitKind",
    "EconomicValue",
    "GraphDiagnostic",
    "GraphDiagnostics",
    "GraphNode",
    "GraphObservation",
    "GraphRelationship",
    "GraphSourceEdge",
    "Materiality",
    "NodeType",
    "Observation",
    "OperatingEconomicGraph",
    "OperatingGraphNode",
    "OperatingGraphObservation",
    "OperatingGraphModel",
    "OperatingGraphRelationship",
    "Provenance",
    "Relationship",
    "RelationshipType",
    "SourceEdge",
    "UnitKind",
    "UnresolvedLeafRequirement",
]
