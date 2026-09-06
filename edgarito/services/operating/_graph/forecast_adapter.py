"""Adapt evaluated economic graphs to the legacy forecast contracts.

The adapter is deliberately a boundary object: graph diagnostics and values
remain in a sidecar, while ``CompanyOperatingForecast`` and
``SegmentRevenueForecast`` retain their existing frozen schemas and serialized
shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from edgarito.schemas.operating import (
    CompanyOperatingForecast,
    EvidenceReference,
    OperatingDriverForecast,
    OperatingSegment,
    SegmentRevenueForecast,
)
from edgarito.schemas.operating_graph import (
    EconomicEvaluationResult,
    EconomicModel,
    EconomicNode,
    EconomicNodeType,
    EconomicProvenance,
    EconomicRelationship,
    EconomicRelationshipType,
    GraphDiagnostics,
)
from edgarito.schemas.valuation.assumptions import (
    AssumptionOrigin,
    AssumptionProvenance,
)
from edgarito.services.operating._graph.evaluator import EconomicGraphEvaluator

_UNAVAILABLE_SOURCE = "unavailable"
_MANAGEMENT_SOURCE = "management_guidance"
_INDEPENDENT_SOURCE = "independent_operating"
_MIXED_SOURCE = "mixed"
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


class EconomicForecastAdaptationError(ValueError):
    """Raised when a legacy forecast cannot represent an unavailable graph root."""

    def __init__(
        self, message: str, *, node_id: str | None = None, years: Iterable[int] = ()
    ):
        self.node_id = node_id
        self.years = tuple(years)
        super().__init__(message)


@dataclass(frozen=True)
class RevenueRootMetadata:
    """Explicit company-root metadata for a legacy forecast adaptation."""

    node_id: str
    company_id: str = "company"
    unit: str | None = None


@dataclass(frozen=True)
class BusinessRootMetadata:
    """Map one graph business root to a true legacy operating scope."""

    node_id: str
    segment: OperatingSegment | Mapping[str, Any] | None = None
    segment_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class EconomicForecastAdapterResult:
    """Legacy forecast plus the graph result and diagnostics sidecar."""

    company_forecast: CompanyOperatingForecast
    graph_result: EconomicEvaluationResult
    diagnostics: GraphDiagnostics

    @property
    def forecast(self) -> CompanyOperatingForecast:
        return self.company_forecast

    @property
    def evaluation(self) -> EconomicEvaluationResult:
        return self.graph_result

    @property
    def graph(self) -> EconomicEvaluationResult:
        return self.graph_result

    @property
    def result(self) -> EconomicEvaluationResult:
        return self.graph_result

    @property
    def graph_diagnostics(self) -> GraphDiagnostics:
        return self.diagnostics


@dataclass(frozen=True)
class _LegacyReconstructionAudit:
    coverage: Decimal | None
    error: Decimal | None
    error_by_year: dict[int, Decimal]
    supported_years: tuple[int, ...]
    confidence: str
    warnings: tuple[str, ...]


def _confidence(values: Iterable[str]) -> str:
    candidates = tuple(str(item).casefold() for item in values if item)
    if not candidates:
        return "medium"
    return min(candidates, key=lambda item: _CONFIDENCE_RANK.get(item, 1))


def _growth(values: Sequence[Decimal]) -> tuple[Decimal | None, ...]:
    result: list[Decimal | None] = [None]
    for previous, current in zip(values[:-1], values[1:], strict=True):
        if previous == 0:
            result.append(Decimal(0) if current == 0 else None)
        else:
            result.append((current / previous - Decimal(1)) * Decimal(100))
    return tuple(result)


def _node_map(model: EconomicModel) -> dict[str, EconomicNode]:
    return {node.node_id: node for node in model.nodes}


def _relationship_map(model: EconomicModel) -> dict[str, EconomicRelationship]:
    return {relationship.target: relationship for relationship in model.relationships}


def _legacy_provenance(
    value: object,
) -> EvidenceReference | AssumptionProvenance | None:
    if value is None:
        return None
    if isinstance(value, EvidenceReference):
        return value
    if isinstance(value, EconomicProvenance):
        if not any(
            (
                value.source,
                value.reference,
                value.methodology,
                value.evidence_ids,
                value.available_on,
            )
        ):
            return None
        if value.origin in {item.value for item in AssumptionOrigin}:
            return AssumptionProvenance(
                origin=AssumptionOrigin(value.origin),
                provider=value.source,
                series_id=value.reference,
                methodology=value.methodology,
                evidence_ids=value.evidence_ids,
            )
        provider = value.source or "economic_graph"
        return EvidenceReference(
            provider=provider,
            accession=value.reference,
            filing_date=value.available_on,
            document_name=value.methodology,
        )
    if isinstance(value, str):
        return EvidenceReference(provider=value)
    if hasattr(value, "model_dump"):
        data = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        data = value
    else:
        data = {}
    return EvidenceReference(
        provider=str(data.get("provider") or data.get("source") or "economic_graph"),
        accession=data.get("accession") or data.get("reference"),
        filing_date=data.get("filing_date") or data.get("available_on"),
        document_name=data.get("document_name") or data.get("methodology"),
    )


def _source_for(
    node_id: str,
    year: int,
    result: EconomicEvaluationResult,
    relationships: Mapping[str, EconomicRelationship],
    *,
    seen: set[tuple[str, int]] | None = None,
) -> str:
    seen = set() if seen is None else seen
    key = (node_id, year)
    if key in seen:
        return _UNAVAILABLE_SOURCE
    seen.add(key)
    cell = result.cell(node_id, year)
    if cell is None or not cell.is_available:
        return _UNAVAILABLE_SOURCE
    if not cell.origin.startswith("relationship:"):
        if cell.origin == _MANAGEMENT_SOURCE:
            return _MANAGEMENT_SOURCE
        if cell.origin == "derived_historical_parameter" and isinstance(
            cell.provenance, EconomicProvenance
        ):
            return cell.provenance.origin or "reported"
        return cell.origin
    relationship = relationships.get(node_id)
    if relationship is None:
        return _INDEPENDENT_SOURCE
    sources = tuple(
        _source_for(
            edge.node_id,
            year - edge.fiscal_lag,
            result,
            relationships,
            seen=seen.copy(),
        )
        for edge in relationship.sources
    )
    if _MANAGEMENT_SOURCE in sources:
        if relationship.relationship_type != EconomicRelationshipType.ADD:
            return _MANAGEMENT_SOURCE
        if set(sources) == {_MANAGEMENT_SOURCE}:
            return _MANAGEMENT_SOURCE
        if any(source != _INDEPENDENT_SOURCE for source in sources):
            return _MIXED_SOURCE
        return _MIXED_SOURCE
    if relationship.relationship_type in {
        EconomicRelationshipType.MULTIPLY,
        EconomicRelationshipType.GROWTH,
        EconomicRelationshipType.RATIO,
        EconomicRelationshipType.SUBTRACT,
    }:
        return _INDEPENDENT_SOURCE
    if len(set(sources)) == 1:
        source = sources[0]
        return _INDEPENDENT_SOURCE if source.startswith("relationship:") else source
    if all(
        source in {_INDEPENDENT_SOURCE, "derived_historical_parameter"}
        for source in sources
    ):
        return _INDEPENDENT_SOURCE
    return _MIXED_SOURCE


def _confidence_for(
    node_id: str,
    year: int,
    result: EconomicEvaluationResult,
    nodes: Mapping[str, EconomicNode],
    relationships: Mapping[str, EconomicRelationship],
    *,
    seen: set[tuple[str, int]] | None = None,
) -> str:
    seen = set() if seen is None else seen
    key = (node_id, year)
    if key in seen:
        return "low"
    seen.add(key)
    cell = result.cell(node_id, year)
    if cell is None or not cell.is_available:
        return "low"
    relationship = relationships.get(node_id)
    if relationship is None or not cell.origin.startswith("relationship:"):
        return nodes[node_id].confidence
    values = [
        _confidence_for(
            edge.node_id,
            year - edge.fiscal_lag,
            result,
            nodes,
            relationships,
            seen=seen.copy(),
        )
        for edge in relationship.sources
    ]
    # Legacy driver confidence is derived from selected input evidence.  The
    # relationship declaration is structural metadata, not an additional
    # evidence observation; ADD therefore naturally carries the worst child.
    return _confidence(values)


def _provenance_for(
    node_id: str,
    year: int,
    result: EconomicEvaluationResult,
    relationships: Mapping[str, EconomicRelationship],
    *,
    seen: set[tuple[str, int]] | None = None,
) -> EvidenceReference | None:
    seen = set() if seen is None else seen
    key = (node_id, year)
    if key in seen:
        return None
    seen.add(key)
    cell = result.cell(node_id, year)
    if cell is None or not cell.is_available:
        return None
    relationship = relationships.get(node_id)
    if relationship is not None and cell.origin.startswith("relationship:"):
        for edge in relationship.sources:
            provenance = _provenance_for(
                edge.node_id,
                year - edge.fiscal_lag,
                result,
                relationships,
                seen=seen.copy(),
            )
            if provenance is not None:
                return provenance
    if relationship is not None and cell.origin.startswith("relationship:"):
        return None
    return _legacy_provenance(cell.provenance)


def _missing_years(node_id: str, result: EconomicEvaluationResult) -> tuple[int, ...]:
    return tuple(
        year
        for year in result.target_years
        if (cell := result.cell(node_id, year)) is None or not cell.is_available
    )


def _raise_if_missing(node_id: str, result: EconomicEvaluationResult) -> None:
    years = _missing_years(node_id, result)
    if not years:
        return
    requirements = tuple(
        item
        for item in result.unresolved_leaf_requirements
        if item.fiscal_year in years
        and (node_id in item.path or item.node_id == node_id)
    )
    detail = "; ".join(item.reason for item in requirements[:3])
    suffix = f" ({detail})" if detail else ""
    raise EconomicForecastAdaptationError(
        f"Economic revenue root {node_id!r} is unavailable for FY{', FY'.join(map(str, years))}{suffix}",
        node_id=node_id,
        years=years,
    )


def _coerce_root_metadata(
    model: EconomicModel,
    revenue_root: str | RevenueRootMetadata | Mapping[str, Any] | None,
) -> RevenueRootMetadata:
    if isinstance(revenue_root, RevenueRootMetadata):
        return revenue_root
    if isinstance(revenue_root, Mapping):
        return RevenueRootMetadata(
            node_id=str(
                revenue_root.get("node_id")
                or revenue_root.get("root_id")
                or revenue_root.get("root_node_id")
                or ""
            ),
            company_id=str(revenue_root.get("company_id") or "company"),
            unit=revenue_root.get("unit"),
        )
    node_id = str(revenue_root or model.revenue_root or "")
    node = model.node_by_id.get(node_id)
    return RevenueRootMetadata(
        node_id=node_id,
        company_id=(
            node.scope_id
            if node is not None and node.scope == "consolidated"
            else "company"
        ),
    )


def _coerce_business_metadata(
    model: EconomicModel,
    business_roots: Mapping[str, Any] | Iterable[Any] | None,
) -> tuple[BusinessRootMetadata, ...]:
    if business_roots is None:
        raw_items: Iterable[Any] = model.business_roots
    elif isinstance(business_roots, Mapping):
        if any(key in business_roots for key in ("node_id", "root_id", "id")):
            raw_items = (business_roots,)
        else:
            raw_items = tuple(
                item
                if isinstance(item, BusinessRootMetadata)
                else BusinessRootMetadata(node_id=str(node_id), segment=item)
                for node_id, item in business_roots.items()
            )
    else:
        raw_items = business_roots
    result: list[BusinessRootMetadata] = []
    for item in raw_items:
        if isinstance(item, BusinessRootMetadata):
            result.append(item)
        elif isinstance(item, str):
            result.append(BusinessRootMetadata(node_id=item))
        elif isinstance(item, Mapping):
            result.append(
                BusinessRootMetadata(
                    node_id=str(
                        item.get("node_id")
                        or item.get("root_id")
                        or item.get("root_node_id")
                        or item.get("id")
                        or ""
                    ),
                    segment=item.get("segment"),
                    segment_id=item.get("segment_id"),
                    name=item.get("name"),
                )
            )
        else:
            raise TypeError("business_roots must contain node IDs or metadata")
    return tuple(item for item in result if item.node_id)


def _segment_for_root(
    metadata: BusinessRootMetadata,
    node: EconomicNode,
    *,
    company_id: str,
) -> OperatingSegment | None:
    if node.scope == "consolidated" or node.scope_id == company_id:
        return None
    if isinstance(metadata.segment, OperatingSegment):
        return metadata.segment
    if isinstance(metadata.segment, Mapping):
        return OperatingSegment.model_validate(metadata.segment)
    segment_id = metadata.segment_id or node.scope_id
    if not segment_id:
        return None
    return OperatingSegment(
        segment_id=segment_id,
        name=metadata.name or segment_id.replace("_", " ").title(),
        scope=node.scope,
        currency=node.currency,
        source="economic_graph",
        confidence=node.confidence,
    )


def _audit_confidence(diagnostics: GraphDiagnostics) -> str:
    coverage = diagnostics.historical_reconstructable_share
    error = diagnostics.reconciliation_error
    if coverage is None or error is None:
        return "low"
    if coverage >= Decimal("0.80") and error <= Decimal("0.05"):
        return "high"
    if coverage >= Decimal("0.50") and error <= Decimal("0.20"):
        return "medium"
    return "low"


def _reconstruction_audit(
    model: EconomicModel,
    result: EconomicEvaluationResult,
    root_id: str,
    relationships: Mapping[str, EconomicRelationship],
) -> _LegacyReconstructionAudit:
    """Recreate the legacy root-vs-formula audit from graph cells."""

    reported = {
        observation.fiscal_year: observation.value
        for observation in model.observations
        if observation.node_id == root_id
        and (
            observation.origin
            in {"reported", "first_party_observation", "extracted_evidence"}
            or (
                observation.origin == "derived_historical_parameter"
                and isinstance(observation.provenance, EconomicProvenance)
                and observation.provenance.origin
                in {"reported", "first_party_observation", "extracted_evidence"}
            )
        )
    }
    if not reported:
        return _LegacyReconstructionAudit(
            coverage=None,
            error=None,
            error_by_year={},
            supported_years=(),
            confidence="low",
            warnings=(),
        )
    root_relationship = relationships.get(root_id)
    if root_relationship is None:
        return _LegacyReconstructionAudit(
            coverage=None,
            error=None,
            error_by_year={},
            supported_years=(),
            confidence="low",
            warnings=(),
        )
    supported: list[int] = []
    errors: dict[int, Decimal] = {}
    for year, reported_value in sorted(reported.items()):
        if root_relationship is None:
            continue
        children = [
            result.cell(edge.node_id, year - edge.fiscal_lag)
            for edge in root_relationship.sources
        ]
        if any(
            cell is None or not cell.is_available or cell.value is None
            for cell in children
        ):
            continue
        reconstructed = sum(
            (
                cell.value * edge.sign * edge.weight
                for edge, cell in zip(root_relationship.sources, children, strict=True)
                if cell is not None and cell.value is not None
            ),
            Decimal(0),
        )
        supported.append(year)
        errors[year] = (
            Decimal(0)
            if reported_value == reconstructed == 0
            else Decimal(1)
            if reported_value == 0
            else abs(reconstructed - reported_value) / abs(reported_value)
        )
    coverage = Decimal(len(supported)) / Decimal(len(reported))
    error = sum(errors.values(), Decimal(0)) / Decimal(len(errors)) if errors else None
    if error is None:
        confidence = "low"
    elif coverage >= Decimal("0.80") and error <= Decimal("0.05"):
        confidence = "high"
    elif coverage >= Decimal("0.50") and error <= Decimal("0.20"):
        confidence = "medium"
    else:
        confidence = "low"
    warnings: list[str] = []
    if coverage < Decimal("0.50"):
        warnings.append(
            f"historical segment driver coverage is low ({coverage:.1%}); forward driver revenue is not fully supported"
        )
    elif coverage < Decimal("0.80"):
        warnings.append(
            f"historical segment driver coverage is partial ({coverage:.1%})"
        )
    if error is not None and error > Decimal("0.20"):
        warnings.append(
            f"historical segment driver reconstruction error is high ({error:.1%})"
        )
    elif error is not None and error > Decimal("0.05"):
        warnings.append(
            f"historical segment driver reconstruction error is elevated ({error:.1%})"
        )
    if supported:
        warnings.append(
            "historical segment driver supported years: "
            + ", ".join(f"FY{year}" for year in supported)
        )
    warnings.append(f"historical segment driver confidence: {confidence}")
    return _LegacyReconstructionAudit(
        coverage=coverage,
        error=error,
        error_by_year=errors,
        supported_years=tuple(supported),
        confidence=confidence,
        warnings=tuple(warnings),
    )


def _explicit_years(
    years: Sequence[int], sources: Mapping[int, str]
) -> tuple[int, ...]:
    return tuple(
        year
        for year in years
        if sources[year] not in {_UNAVAILABLE_SOURCE, "historical_reported"}
    )


def _driver_forecasts(
    root_id: str,
    segment: OperatingSegment,
    years: Sequence[int],
    result: EconomicEvaluationResult,
    nodes: Mapping[str, EconomicNode],
    relationships: Mapping[str, EconomicRelationship],
    warnings: list[str],
    audit_confidence: str,
) -> tuple[OperatingDriverForecast, ...]:
    root_relationship = relationships.get(root_id)
    if root_relationship is None:
        return ()
    formula_nodes = tuple(
        edge.node_id
        for edge in root_relationship.sources
        if edge.node_id in nodes
        and nodes[edge.node_id].node_type == EconomicNodeType.DERIVED
    )
    forecasts: dict[tuple[str, int], OperatingDriverForecast] = {}

    def add_forecast(forecast: OperatingDriverForecast) -> None:
        key = (forecast.driver_id, forecast.fiscal_year)
        previous = forecasts.get(key)
        if previous is not None:
            if previous.value != forecast.value:
                warnings.append(
                    f"FY{forecast.fiscal_year} {forecast.driver_id}: conflicting driver forecasts; first deterministic value retained"
                )
            return
        forecasts[key] = forecast

    def add_input(node_id: str, year: int) -> None:
        node = nodes[node_id]
        if node.metric == "previous_revenue":
            return
        cell = result.cell(node_id, year)
        if cell is None or not cell.is_available or cell.value is None:
            return
        source = _source_for(node_id, year, result, relationships)
        add_forecast(
            OperatingDriverForecast(
                segment_id=segment.segment_id,
                driver_id=node.metric,
                fiscal_year=year,
                value=cell.value,
                unit=node.unit,
                source=source,
                method="management constraint"
                if source == _MANAGEMENT_SOURCE
                else "observed input",
                confidence=_confidence_for(node_id, year, result, nodes, relationships),
                provenance=_provenance_for(node_id, year, result, relationships),
            )
        )

    def add_formula(node_id: str, year: int) -> None:
        node = nodes[node_id]
        cell = result.cell(node_id, year)
        if cell is None or not cell.is_available or cell.value is None:
            return
        relationship = relationships.get(node_id)
        if relationship is None:
            return
        for edge in relationship.sources:
            source_node = nodes.get(edge.node_id)
            if source_node is None:
                continue
            if source_node.node_type == EconomicNodeType.INPUT:
                add_input(edge.node_id, year - edge.fiscal_lag)
            elif source_node.node_id in relationships:
                add_formula_inputs(source_node.node_id, year - edge.fiscal_lag)
        formula_source = _source_for(node_id, year, result, relationships)
        add_forecast(
            OperatingDriverForecast(
                segment_id=segment.segment_id,
                driver_id=node.metric,
                fiscal_year=year,
                value=cell.value,
                unit=node.unit,
                source=formula_source,
                method=(
                    f"formula:{relationship.relationship_id.rsplit(':formula:', 1)[1]}"
                    if relationship.relationship_id
                    and ":formula:" in relationship.relationship_id
                    else f"formula:{relationship.relationship_id or node.metric}"
                ),
                confidence=(
                    _confidence(
                        (
                            _confidence_for(
                                node_id, year, result, nodes, relationships
                            ),
                            audit_confidence,
                        )
                    )
                    if formula_source == _INDEPENDENT_SOURCE
                    else _confidence_for(node_id, year, result, nodes, relationships)
                ),
                provenance=_provenance_for(node_id, year, result, relationships),
            )
        )

    def add_formula_inputs(node_id: str, year: int) -> None:
        relationship = relationships.get(node_id)
        if relationship is None:
            return
        for edge in relationship.sources:
            source_node = nodes.get(edge.node_id)
            if source_node is None:
                continue
            if source_node.node_type == EconomicNodeType.INPUT:
                add_input(edge.node_id, year - edge.fiscal_lag)
            elif edge.node_id in relationships:
                add_formula_inputs(edge.node_id, year - edge.fiscal_lag)

    for year in years:
        for formula_node in formula_nodes:
            add_formula(formula_node, year)
    return tuple(
        sorted(forecasts.values(), key=lambda item: (item.fiscal_year, item.driver_id))
    )


def adapt_economic_forecast(
    model: EconomicModel,
    result: EconomicEvaluationResult,
    *,
    revenue_root: str | RevenueRootMetadata | Mapping[str, Any] | None = None,
    business_roots: Mapping[str, Any] | Iterable[Any] | None = None,
    company_id: str | None = None,
    revenue_root_metadata: str | RevenueRootMetadata | Mapping[str, Any] | None = None,
    business_root_metadata: Mapping[str, Any] | Iterable[Any] | None = None,
) -> EconomicForecastAdapterResult:
    """Adapt one graph evaluation without ever replacing unavailable values.

    The legacy schemas require concrete non-negative revenue paths.  Therefore
    an unavailable root is reported as ``EconomicForecastAdaptationError``
    rather than being silently converted to a zero.
    """

    root_metadata = _coerce_root_metadata(
        model,
        revenue_root if revenue_root is not None else revenue_root_metadata,
    )
    root_id = root_metadata.node_id
    if not root_id:
        raise EconomicForecastAdaptationError(
            "An explicit economic revenue root is required"
        )
    nodes = _node_map(model)
    relationships = _relationship_map(model)
    if root_id not in nodes:
        raise EconomicForecastAdaptationError(
            f"Unknown economic revenue root: {root_id}", node_id=root_id
        )
    _raise_if_missing(root_id, result)
    resolved_company_id = (
        company_id or root_metadata.company_id or nodes[root_id].scope_id or "company"
    )
    metadata = _coerce_business_metadata(
        model,
        business_roots if business_roots is not None else business_root_metadata,
    )
    for item in metadata:
        if item.node_id not in nodes:
            raise EconomicForecastAdaptationError(
                f"Unknown economic business root: {item.node_id}", node_id=item.node_id
            )
        _raise_if_missing(item.node_id, result)

    years = result.target_years
    warnings = [
        item.message
        for item in result.diagnostics.diagnostic_messages
        if item.code != "unknown_materiality"
    ]
    segment_forecasts: list[SegmentRevenueForecast] = []
    audits = {
        item.node_id: _reconstruction_audit(model, result, item.node_id, relationships)
        for item in metadata
    }
    root_audit = _reconstruction_audit(model, result, root_id, relationships)
    segment_audits = tuple(
        audit for item, audit in audits.items() if audit.coverage is not None
    )
    if root_audit.coverage is not None:
        company_audit = root_audit
    elif segment_audits:
        coverage = min(
            audit.coverage for audit in segment_audits if audit.coverage is not None
        )
        errors = {
            year: error
            for audit in segment_audits
            for year, error in audit.error_by_year.items()
        }
        error = (
            sum(errors.values(), Decimal(0)) / Decimal(len(errors)) if errors else None
        )
        aggregate_confidence = (
            "high"
            if error is not None
            and coverage >= Decimal("0.80")
            and error <= Decimal("0.05")
            else "medium"
            if error is not None
            and coverage >= Decimal("0.50")
            and error <= Decimal("0.20")
            else "low"
        )
        company_warnings = (
            [
                "historical company driver supported years: "
                + ", ".join(f"FY{year}" for year in sorted(errors))
            ]
            if errors
            else []
        )
        company_warnings.append(
            f"historical company driver confidence: {aggregate_confidence}"
        )
        company_audit = _LegacyReconstructionAudit(
            coverage=coverage,
            error=error,
            error_by_year=errors,
            supported_years=tuple(sorted(errors)),
            confidence=aggregate_confidence,
            warnings=tuple(company_warnings),
        )
    else:
        company_audit = _LegacyReconstructionAudit(
            coverage=None,
            error=None,
            error_by_year={},
            supported_years=(),
            confidence=_audit_confidence(result.diagnostics),
            warnings=(),
        )
    audit_confidence = company_audit.confidence
    for item in metadata:
        segment = _segment_for_root(
            item, nodes[item.node_id], company_id=resolved_company_id
        )
        if segment is None:
            # A consolidated company root is represented by the company
            # forecast itself, never as a synthetic reporting segment.
            continue
        values = tuple(result.value(item.node_id, year) for year in years)
        if any(value is None for value in values):
            _raise_if_missing(item.node_id, result)
        revenue = tuple(value for value in values if value is not None)
        sources = {
            year: _source_for(item.node_id, year, result, relationships)
            for year in years
        }
        confidences = {
            year: _confidence(
                (
                    _confidence_for(item.node_id, year, result, nodes, relationships),
                    audits[item.node_id].confidence,
                )
            )
            if _source_for(item.node_id, year, result, relationships)
            == _INDEPENDENT_SOURCE
            else _confidence_for(item.node_id, year, result, nodes, relationships)
            for year in years
        }
        segment_warnings = list(warnings)
        segment_audit = audits[item.node_id]
        if segment_audit.coverage is None and item.node_id in relationships:
            segment_warnings.append(
                "historical driver reconstruction unavailable: no reported segment revenue history"
            )
        else:
            segment_warnings.extend(segment_audit.warnings)
        drivers = _driver_forecasts(
            item.node_id,
            segment,
            years,
            result,
            nodes,
            relationships,
            segment_warnings,
            segment_audit.confidence,
        )
        segment_forecasts.append(
            SegmentRevenueForecast(
                segment=segment,
                fiscal_years=years,
                revenue=revenue,
                revenue_growth=_growth(revenue),
                driver_forecasts=drivers,
                explicit_years=_explicit_years(years, sources),
                source_by_year=sources,
                confidence_by_year=confidences,
                warnings=tuple(dict.fromkeys(segment_warnings)),
                unit=nodes[item.node_id].unit,
                driver_coverage=segment_audit.coverage,
                modeled_revenue_share=Decimal(1),
                genuine_coverage=segment_audit.coverage,
                reconstruction_error=segment_audit.error,
                reconstruction_error_by_year=segment_audit.error_by_year,
                supported_years=segment_audit.supported_years,
                own_supported_years=tuple(
                    year
                    for year in years
                    if sources[year] == _INDEPENDENT_SOURCE
                    and _CONFIDENCE_RANK[confidences[year]]
                    >= _CONFIDENCE_RANK["medium"]
                ),
                confidence=segment_audit.confidence,
            )
        )

    revenue = tuple(
        value for year in years if (value := result.value(root_id, year)) is not None
    )
    if len(revenue) != len(
        years
    ):  # defensive; the check above is the contract boundary
        _raise_if_missing(root_id, result)
    company_sources = {
        year: _source_for(root_id, year, result, relationships) for year in years
    }
    company_confidences = {
        year: _confidence(
            (
                _confidence_for(root_id, year, result, nodes, relationships),
                audit_confidence,
            )
        )
        if company_sources[year] == _INDEPENDENT_SOURCE
        else _confidence_for(root_id, year, result, nodes, relationships)
        for year in years
    }
    warnings.extend(
        f"{item.segment.segment_id}: {warning}"
        for item in segment_forecasts
        for warning in item.warnings
        if warning not in warnings
    )
    if company_audit.coverage is None and root_id in relationships:
        warnings.append(
            "company: historical company driver reconstruction unavailable: no reported company revenue history"
        )
    else:
        warnings.extend(f"company: {warning}" for warning in company_audit.warnings)
    explicit_years = _explicit_years(years, company_sources)
    company = CompanyOperatingForecast(
        company_id=resolved_company_id,
        fiscal_years=years,
        segment_forecasts=tuple(segment_forecasts),
        consolidated_revenue=revenue,
        consolidated_growth=_growth(revenue),
        explicit_years=explicit_years,
        transition_start_year=explicit_years[-1] + 1 if explicit_years else None,
        source_by_year=company_sources,
        confidence_by_year=company_confidences,
        warnings=tuple(dict.fromkeys(warnings)),
        unit=root_metadata.unit or nodes[root_id].unit,
        driver_coverage=company_audit.coverage,
        modeled_revenue_share=Decimal(1),
        genuine_coverage=company_audit.coverage,
        reconstruction_error=company_audit.error,
        reconstruction_error_by_year=company_audit.error_by_year,
        supported_years=company_audit.supported_years,
        own_supported_years=tuple(
            year
            for year in years
            if company_sources[year] == _INDEPENDENT_SOURCE
            and _CONFIDENCE_RANK[company_confidences[year]]
            >= _CONFIDENCE_RANK["medium"]
        ),
        confidence=company_audit.confidence,
    )
    return EconomicForecastAdapterResult(
        company_forecast=company,
        graph_result=result,
        diagnostics=result.diagnostics,
    )


class EconomicForecastAdapterService:
    """Evaluate an economic model and adapt the result in one operation."""

    def __init__(self, evaluator: EconomicGraphEvaluator | None = None) -> None:
        self.evaluator = evaluator or EconomicGraphEvaluator()

    def forecast(
        self,
        model: EconomicModel,
        target_years: int | Iterable[int],
        *,
        as_of=None,
        fiscal_period: str | None = None,
        revenue_root=None,
        business_roots=None,
        company_id: str | None = None,
        revenue_root_metadata=None,
        business_root_metadata=None,
    ) -> EconomicForecastAdapterResult:
        result = self.evaluator.evaluate(
            model,
            target_years,
            as_of=as_of,
            fiscal_period=fiscal_period,
        )
        return adapt_economic_forecast(
            model,
            result,
            revenue_root=revenue_root,
            business_roots=business_roots,
            company_id=company_id,
            revenue_root_metadata=revenue_root_metadata,
            business_root_metadata=business_root_metadata,
        )

    evaluate_and_adapt = forecast
    build = forecast


EconomicForecastService = EconomicForecastAdapterService
OperatingGraphForecastService = EconomicForecastAdapterService
adapt_economic_evaluation = adapt_economic_forecast
adapt_graph_forecast = adapt_economic_forecast
adapt_economic_model_forecast = adapt_economic_forecast
economic_model_to_operating_forecast = adapt_economic_forecast
OperatingForecastAdapter = EconomicForecastAdapterService


__all__ = [
    "BusinessRootMetadata",
    "EconomicForecastAdaptationError",
    "EconomicForecastAdapterResult",
    "EconomicForecastAdapterService",
    "EconomicForecastService",
    "OperatingForecastAdapter",
    "OperatingGraphForecastService",
    "RevenueRootMetadata",
    "adapt_economic_evaluation",
    "adapt_economic_forecast",
    "adapt_graph_forecast",
    "adapt_economic_model_forecast",
    "economic_model_to_operating_forecast",
]
