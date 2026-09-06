"""Recursive, memoized factor resolution without production integration."""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    SerializationInfo,
    model_serializer,
    model_validator,
)

from edgarito.services.valuation.factors.cache import SemanticFactorCache
from edgarito.services.valuation.factors.contracts import (
    CacheState,
    FactorEstimate,
    FactorKey,
    FactorRange,
    FactorRequest,
    FactorResolutionStatus,
    StopReason,
)
from edgarito.services.valuation.factors.decomposers.base import (
    FactorDecomposition,
    StaticMappingDecomposer,
)
from edgarito.services.valuation.factors.freshness import FactorFreshnessPolicy
from edgarito.services.valuation.factors.graph import (
    FactorGraphEdge,
    FactorGraphNode,
    FactorGraphSnapshot,
    ValuationFactorGraph,
)
from edgarito.services.valuation.factors.policy import (
    FactorExpansionDecision,
    FactorExpansionPolicy,
    FactorResolutionMetrics,
    _Usage,
)
from edgarito.services.valuation.factors.registry import (
    FactorDecomposerRegistry,
    FactorResolverRegistry,
)
from edgarito.services.valuation.factors.resolvers.base import ResolverResult

_PRIORITY_RANK = {
    "low": 0,
    "normal": 1,
    "high": 2,
    "urgent": 3,
}
_MATERIALITY_RANK = {
    "unknown": -1,
    "immaterial": 0,
    "low": 1,
    "material": 2,
    "medium": 3,
    "high": 4,
    "critical": 5,
}
_LIMIT_STOP_REASONS = frozenset(
    {
        StopReason.MAX_DEPTH,
        StopReason.MAX_NODES,
        StopReason.BUDGET_EXHAUSTED,
        StopReason.MAX_RESOLUTION_COST,
        StopReason.MAX_EXTERNAL_CALLS,
    }
)


def _request_order_key(request: FactorRequest):
    return (
        -_PRIORITY_RANK[request.priority.value],
        -_MATERIALITY_RANK[request.materiality.value],
        request.key.digest,
    )


def _proposal_order_key(proposal):
    return (
        -_PRIORITY_RANK[proposal.priority.value],
        -_MATERIALITY_RANK[proposal.materiality.value],
        proposal.child_key.digest,
    )


@dataclass(frozen=True)
class FactorResolutionContext:
    evaluated_at: dt.date | dt.datetime
    values: Mapping[str, Any] = None
    decomposition: Optional[FactorDecomposition] = None
    children: Mapping[Any, FactorEstimate] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values or {})))
        object.__setattr__(self, "children", MappingProxyType(dict(self.children or {})))

    def with_data(self, **updates: Any) -> "FactorResolutionContext":
        return replace(self, **updates)

    def get(self, name: str, default=None):
        return self.values.get(name, default)

    def __getitem__(self, name: str):
        return self.values[name]


class _EstimateMap(dict):
    def __getitem__(self, key):
        if isinstance(key, FactorKey):
            for candidate in (key, key.digest, key.semantic_id):
                if dict.__contains__(self, candidate):
                    return dict.__getitem__(self, candidate)
            raise KeyError(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class FactorResolutionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    roots: tuple[FactorKey, ...]
    estimates: Mapping[Any, FactorEstimate]
    graph: FactorGraphSnapshot
    metrics: FactorResolutionMetrics

    @model_validator(mode="before")
    @classmethod
    def restore_serialized_views(cls, value: Any) -> Any:
        """Rebuild immutable graph and estimate views from audit JSON."""

        if not isinstance(value, Mapping):
            return value
        values = dict(value)
        if "estimates" in values:
            values["estimates"] = _restore_estimates(values["estimates"])
        if "graph" in values:
            values["graph"] = _restore_graph(values["graph"])
        return values

    @model_validator(mode="after")
    def normalize_estimate_map(self) -> "FactorResolutionResult":
        object.__setattr__(self, "estimates", _EstimateMap(self.estimates))
        return self

    @model_serializer(mode="plain")
    def serialize_audit_result(self, info: SerializationInfo) -> dict[str, Any]:
        """Return a deterministic, JSON-safe audit contract.

        ``estimates`` and ``graph.nodes`` are intentionally represented as
        sorted records rather than mappings.  Their in-memory mappings remain
        immutable/publicly compatible while the JSON contract never exposes a
        ``MappingProxyType`` or requester-owned object.
        """

        mode = info.mode
        dump_mode = "json" if mode == "json" else "python"
        return {
            "roots": [
                root.model_dump(mode=dump_mode, by_alias=False) for root in self.roots
            ],
            "estimates": [
                {
                    "digest": estimate.key.digest,
                    "estimate": estimate.model_dump(mode=dump_mode, by_alias=False),
                }
                for estimate in sorted(
                    self.estimates.values(), key=lambda item: item.key.digest
                )
            ],
            "graph": _serialize_graph(self.graph, mode=dump_mode),
            "metrics": self.metrics.model_dump(mode=dump_mode, by_alias=False),
        }

    @property
    def estimate(self) -> Optional[FactorEstimate]:
        return self.estimates.get(self.roots[0]) if self.roots else None

    @property
    def resolved(self) -> bool:
        return all(self.estimates.get(root) is not None for root in self.roots)

    @property
    def root_estimates(self) -> Mapping[Any, FactorEstimate]:
        return MappingProxyType(
            {
                root: self.estimates[root]
                for root in self.roots
                if self.estimates.get(root) is not None
            }
        )

    @property
    def statuses(self) -> Mapping[str, FactorResolutionStatus]:
        return MappingProxyType(
            {
                digest: node.status for digest, node in self.graph.nodes.items()
            }
        )

    def for_key(self, key: FactorKey) -> Optional[FactorEstimate]:
        return self.estimates.get(key)


def _restore_estimates(value: Any) -> Any:
    if isinstance(value, _EstimateMap):
        return value
    if isinstance(value, Mapping):
        records = [
            {"digest": digest, "estimate": estimate}
            for digest, estimate in value.items()
        ]
    else:
        records = value
    if not isinstance(records, (list, tuple)):
        return value

    estimates = _EstimateMap()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("serialized estimates must be records")
        raw_estimate = record.get("estimate", record)
        estimate = FactorEstimate.model_validate(raw_estimate)
        digest = record.get("digest")
        if digest is not None and str(digest) != estimate.key.digest:
            raise ValueError("serialized estimate digest does not match its key")
        estimates[estimate.key] = estimate
    return estimates


def _serialize_graph(graph: FactorGraphSnapshot, *, mode: str) -> dict[str, Any]:
    nodes = []
    for node in sorted(graph.nodes.values(), key=lambda item: item.key.digest):
        nodes.append(
            {
                "digest": node.key.digest,
                "node": {
                    "key": node.key.model_dump(mode=mode, by_alias=False),
                    "status": node.status.value,
                    "estimate": (
                        node.estimate.model_dump(mode=mode, by_alias=False)
                        if node.estimate is not None
                        else None
                    ),
                    "depth": node.depth,
                    "priority": node.priority.value,
                    "materiality": node.materiality.value,
                    "uncertainty": (
                        node.uncertainty.model_dump(mode=mode, by_alias=False)
                        if node.uncertainty is not None
                        else None
                    ),
                    "stop_reason": (
                        node.stop_reason.value if node.stop_reason is not None else None
                    ),
                    "resolver": node.resolver,
                    "cache_state": (
                        node.cache_state.value if node.cache_state is not None else None
                    ),
                    "warnings": list(node.warnings),
                },
            }
        )
    edges = [
        {
            "dependency": edge.dependency.model_dump(mode=mode, by_alias=False),
            "consumer": edge.consumer.model_dump(mode=mode, by_alias=False),
            "role": edge.role,
            "rationale": edge.rationale,
            "materiality": edge.materiality.value,
            "weight": str(edge.weight),
        }
        for edge in sorted(
            graph.edges,
            key=lambda item: (
                item.dependency.digest,
                item.consumer.digest,
                item.role,
                item.rationale,
                item.materiality.value,
                str(item.weight),
            ),
        )
    ]
    return {"nodes": nodes, "edges": edges}


def _restore_graph(value: Any) -> Any:
    if isinstance(value, FactorGraphSnapshot):
        return value
    if not isinstance(value, Mapping):
        return value

    raw_nodes = value.get("nodes", ())
    if isinstance(raw_nodes, Mapping):
        node_records = [
            {"digest": digest, "node": node} for digest, node in raw_nodes.items()
        ]
    else:
        node_records = raw_nodes
    nodes = {}
    for record in node_records:
        if not isinstance(record, Mapping):
            raise ValueError("serialized graph nodes must be records")
        raw_node = record.get("node", record)
        if not isinstance(raw_node, Mapping):
            raise ValueError("serialized graph node must be an object")
        node_values = dict(raw_node)
        node_values["key"] = FactorKey.model_validate(node_values["key"])
        if node_values.get("estimate") is not None:
            node_values["estimate"] = FactorEstimate.model_validate(
                node_values["estimate"]
            )
        if node_values.get("uncertainty") is not None:
            node_values["uncertainty"] = FactorRange.model_validate(
                node_values["uncertainty"]
            )
        node = FactorGraphNode(**node_values)
        digest = record.get("digest")
        if digest is not None and str(digest) != node.key.digest:
            raise ValueError("serialized graph node digest does not match its key")
        nodes[node.key.digest] = node

    edges = []
    for raw_edge in value.get("edges", ()):
        if not isinstance(raw_edge, Mapping):
            raise ValueError("serialized graph edges must be objects")
        edge_values = dict(raw_edge)
        edge_values["dependency"] = FactorKey.model_validate(
            edge_values["dependency"]
        )
        edge_values["consumer"] = FactorKey.model_validate(edge_values["consumer"])
        edges.append(FactorGraphEdge(**edge_values))
    return FactorGraphSnapshot(nodes=nodes, edges=tuple(edges))


FactorResolution = FactorResolutionResult
ResolutionResult = FactorResolutionResult
ResolutionContext = FactorResolutionContext


def _as_requests(
    requests: FactorRequest | FactorKey | Iterable[FactorRequest | FactorKey],
    evaluated_at: Optional[dt.date | dt.datetime],
) -> tuple[FactorRequest, ...]:
    if isinstance(requests, (FactorRequest, FactorKey)):
        requests = (requests,)
    result = []
    for item in requests:
        if isinstance(item, FactorRequest):
            result.append(item)
        else:
            if evaluated_at is None:
                raise ValueError(
                    "bare FactorKey roots require an explicit evaluated_at"
                )
            result.append(FactorRequest(key=item, information_as_of=evaluated_at))
    return tuple(result)


class RecursiveFactorResolutionService:
    """Resolve each canonical key once while preserving its dependency DAG."""

    def __init__(
        self,
        resolvers: Iterable[object] = (),
        decomposers: Iterable[object] = (),
        *,
        resolver_registry: Optional[FactorResolverRegistry] = None,
        decomposer_registry: Optional[FactorDecomposerRegistry] = None,
        cache: Optional[SemanticFactorCache] = None,
        policy: Optional[FactorExpansionPolicy] = None,
        freshness_policy: Optional[FactorFreshnessPolicy] = None,
        freshness: Optional[FactorFreshnessPolicy] = None,
        expansion_policy: Optional[FactorExpansionPolicy] = None,
        evidence: Iterable[Any] = (),
        registry=None,
    ) -> None:
        if registry is not None:
            resolver_registry = resolver_registry or registry.resolvers
            decomposer_registry = decomposer_registry or registry.decomposers
        if isinstance(resolvers, (str, bytes)) or not isinstance(resolvers, Iterable):
            resolvers = (resolvers,)
        if isinstance(decomposers, Mapping):
            decomposers = (StaticMappingDecomposer(decomposers),)
        elif isinstance(decomposers, (str, bytes)) or not isinstance(decomposers, Iterable):
            decomposers = (decomposers,)
        self.resolvers = resolver_registry or FactorResolverRegistry(resolvers)
        self.decomposers = decomposer_registry or FactorDecomposerRegistry(decomposers)
        self.cache = cache or SemanticFactorCache()
        self.policy = policy or expansion_policy or FactorExpansionPolicy()
        self.freshness_policy = (
            freshness_policy or freshness or FactorFreshnessPolicy()
        )
        if evidence:
            from edgarito.services.valuation.factors.resolvers.direct_evidence import (
                DirectEvidenceResolver,
            )

            self.resolvers.register(DirectEvidenceResolver(evidence))
        self._reset()

    def _reset(self) -> None:
        self.graph = ValuationFactorGraph()
        self._memo: dict[str, Optional[FactorEstimate]] = {}
        self._memo_strength: dict[str, tuple[int, int]] = {}
        self._usage = _Usage()
        self._stack: list[str] = []
        self._started = 0.0
        self._context: Optional[FactorResolutionContext] = None

    def resolve(
        self,
        requests: FactorRequest | FactorKey | Iterable[FactorRequest | FactorKey] | None = None,
        context: Optional[FactorResolutionContext | Mapping[str, Any]] = None,
        *,
        evaluated_at: Optional[dt.date | dt.datetime] = None,
        roots=None,
    ) -> FactorResolutionResult:
        if requests is None:
            requests = roots
        if requests is None:
            raise ValueError("at least one factor request is required")
        root_requests = _as_requests(requests, evaluated_at)
        if not root_requests:
            raise ValueError("at least one factor request is required")
        information_as_of = root_requests[0].information_as_of
        if any(
            request.information_as_of != information_as_of
            for request in root_requests[1:]
        ):
            raise ValueError(
                "all root requests in one resolution must use the same "
                "information_as_of"
            )
        root_requests = tuple(sorted(root_requests, key=_request_order_key))
        self._reset()
        if context is None:
            context = FactorResolutionContext(
                evaluated_at=evaluated_at or root_requests[0].information_as_of
            )
        elif not isinstance(context, FactorResolutionContext):
            values = dict(context)
            context = FactorResolutionContext(
                evaluated_at=values.pop("evaluated_at", None)
                or evaluated_at
                or root_requests[0].information_as_of,
                values=values,
            )
        elif evaluated_at is not None and context.evaluated_at != evaluated_at:
            context = context.with_data(evaluated_at=evaluated_at)
        self._context = context
        self._started = time.monotonic()
        for request in root_requests:
            self._resolve_request(request, depth=0)
        estimates = _EstimateMap()
        for digest, estimate in self._memo.items():
            if estimate is not None:
                node = self.graph.nodes.get(digest)
                if node is not None:
                    estimates[node.key] = estimate
        elapsed = time.monotonic() - self._started
        return FactorResolutionResult(
            roots=tuple(request.key for request in root_requests),
            estimates=estimates,
            graph=self.graph.snapshot(),
            metrics=self._usage.metrics(elapsed),
        )

    def _resolve_request(self, request: FactorRequest, *, depth: int) -> Optional[FactorEstimate]:
        digest = request.key.digest
        if digest in self._memo:
            memoized = self._memo[digest]
            if memoized is not None:
                return memoized
            current_strength = (
                _PRIORITY_RANK[request.priority.value],
                _MATERIALITY_RANK[request.materiality.value],
            )
            previous_strength = self._memo_strength.get(digest, current_strength)
            if not (
                current_strength[0] > previous_strength[0]
                or current_strength[1] > previous_strength[1]
            ):
                return None
            del self._memo[digest]
            self._memo_strength.pop(digest, None)
        self.graph.add_node(
            request.key,
            depth=depth,
            priority=request.priority,
            materiality=request.materiality,
        )
        if digest in self._stack:
            self._set_node(request.key, status=FactorResolutionStatus.FAILED, stop_reason=StopReason.FAILED, warnings=("cycle detected",))
            self._memoize(request, None)
            return None
        self._stack.append(digest)
        try:
            cached = self._try_cache(request, depth)
            if cached is not None:
                self._memoize(request, cached)
                return cached

            direct = self._resolve_direct(request)
            if direct is not None:
                self._memoize(request, direct)
                return direct

            decision = self.policy.decision(
                request,
                depth=depth,
                nodes=len(self.graph),
                resolution_cost=self._usage.resolution_cost,
                external_calls=self._usage.external_calls,
                model_calls=self._usage.model_calls,
                resolver_calls=self._usage.resolver_calls,
            )
            if not decision.allowed:
                self._stop(request.key, decision)
                self._memoize(request, None)
                return None

            decomposition = self._decompose(request)
            if decomposition is None:
                self._stop(request.key, FactorExpansionDecision(allowed=False, reason=StopReason.NO_DECOMPOSITION))
                self._memoize(request, None)
                return None

            children: dict[str, FactorEstimate] = {}
            missing_required = False
            blocked_reason: Optional[StopReason] = None
            proposals = sorted(decomposition.proposals, key=_proposal_order_key)
            for proposal in proposals:
                child_key = proposal.child_key
                try:
                    self.graph.add_edge(
                        child_key,
                        request.key,
                        role=proposal.relationship_role,
                        rationale=proposal.rationale,
                        materiality=proposal.materiality,
                        weight=proposal.weight,
                    )
                except ValueError:
                    self._set_node(
                        request.key,
                        status=FactorResolutionStatus.FAILED,
                        stop_reason=StopReason.FAILED,
                        warnings=("cycle detected",),
                    )
                    self._memoize(request, None)
                    return None
                child_request = request.model_copy(
                    update={
                        "key": child_key,
                        "priority": proposal.priority,
                        "materiality": proposal.materiality,
                        "max_depth": request.max_depth,
                        "remaining_depth": max(request.max_depth - depth - 1, 0),
                    }
                )
                child = self._resolve_request(child_request, depth=depth + 1)
                if child is None:
                    missing_required = missing_required or proposal.required
                    child_node = self.graph.get_node(child_key)
                    if child_node is not None and child_node.stop_reason in (
                        _LIMIT_STOP_REASONS | {StopReason.BELOW_MATERIALITY}
                    ):
                        blocked_reason = child_node.stop_reason
                else:
                    children[child_key.digest] = child
            if missing_required:
                self._stop(
                    request.key,
                    FactorExpansionDecision(
                        allowed=False,
                        reason=blocked_reason or StopReason.UNRESOLVED_DEPENDENCIES,
                    ),
                )
                self._memoize(request, None)
                return None

            derived = self._resolve_derived(request, decomposition, children)
            if derived is None:
                node = self.graph.get_node(request.key)
                if node is None or node.stop_reason not in _LIMIT_STOP_REASONS:
                    self._stop(
                        request.key,
                        FactorExpansionDecision(
                            allowed=False, reason=StopReason.NO_RESOLVER
                        ),
                    )
            else:
                self._memoize(request, derived)
            return derived
        except Exception as exc:  # resolver failures are represented in the graph
            self._set_node(
                request.key,
                status=FactorResolutionStatus.FAILED,
                stop_reason=StopReason.FAILED,
                warnings=(str(exc),),
            )
            self._memoize(request, None)
            return None
        finally:
            self._stack.pop()

    def _try_cache(self, request: FactorRequest, depth: int) -> Optional[FactorEstimate]:
        history = tuple(reversed(self.cache.history(request.key)))
        if not history:
            self._set_node(request.key, cache_state=CacheState.MISS)
            return None
        for candidate in history:
            current = {}
            if candidate.dependencies:
                complete = True
                for dependency in candidate.dependencies:
                    try:
                        self.graph.add_edge(
                            dependency,
                            request.key,
                            role="cached_dependency",
                            rationale="recorded cached dependency",
                        )
                    except ValueError:
                        complete = False
                        break
                    child_request = request.model_copy(update={"key": dependency})
                    child = self._resolve_request(child_request, depth=depth + 1)
                    if child is None:
                        complete = False
                        break
                    current[dependency.digest] = child.fingerprint
                if not complete:
                    continue
            lookup = self.cache.lookup(
                request,
                self.freshness_policy,
                self._context.evaluated_at,
                current,
            )
            if lookup.hit and lookup.estimate is not None:
                self._set_node(
                    request.key,
                    status=FactorResolutionStatus.CACHE_HIT,
                    estimate=lookup.estimate,
                    resolver=lookup.estimate.resolver,
                    cache_state=CacheState.HIT,
                    stop_reason=StopReason.FRESH_CACHE,
                )
                return lookup.estimate
            # A direct candidate cannot need dependency recursion. Avoid
            # repeatedly resolving it when a newer version is stale.
            if not candidate.dependencies:
                break
        self._set_node(request.key, cache_state=CacheState.STALE)
        return None

    def _resolve_direct(self, request: FactorRequest) -> Optional[FactorEstimate]:
        candidates = self.resolvers.candidates(request, self._context, derived=False)
        for resolver in candidates:
            attempt = self.policy.attempt_decision(
                request,
                resolver_calls=self._usage.resolver_calls,
            )
            if not attempt.allowed:
                self._stop(request.key, attempt)
                return None
            self._usage.record_resolver_call()
            try:
                raw = resolver.resolve(request, self._context)
            except TypeError:
                raw = resolver.resolve(request)
            result = (
                raw
                if isinstance(raw, ResolverResult)
                else ResolverResult(estimate=raw)
            )
            self._usage.add(
                result.cost,
                external=result.external_calls,
                model=result.model_calls,
                expected_currency=self._cost_currency(request),
                count_resolver=False,
            )
            if not result.resolved:
                continue
            estimate = result.estimate
            if estimate is None or estimate.key != request.key:
                continue
            if not self._within_limits(request):
                self._stop(
                    request.key,
                    self._limit_decision(request),
                )
                return None
            reason = (
                StopReason.SUFFICIENT_CONFIDENCE
                if estimate.confidence.rank >= request.min_confidence.rank
                else None
            )
            self._set_node(
                request.key,
                status=FactorResolutionStatus.DIRECTLY_RESOLVED,
                estimate=estimate,
                resolver=estimate.resolver,
                cache_state=CacheState.MISS,
                stop_reason=reason,
                warnings=result.warnings,
            )
            self.cache.put(estimate)
            return estimate
        return None

    def _decompose(self, request: FactorRequest) -> Optional[FactorDecomposition]:
        for decomposer in self.decomposers.candidates(request, self._context):
            raw = decomposer.decompose(request, self._context)
            if isinstance(raw, FactorDecomposition):
                result = raw
            else:
                result = FactorDecomposition(
                    parent_key=request.key,
                    proposals=tuple(raw),
                    decomposer=getattr(decomposer, "decomposer_id", type(decomposer).__name__),
                )
            if result.parent_key != request.key:
                result = result.model_copy(update={"parent_key": request.key})
            self._usage.add_cost(
                result.cost,
                expected_currency=self._cost_currency(request),
            )
            self._usage.external_calls += result.external_calls
            self._usage.model_calls += result.model_calls
            return result
        return None

    def _resolve_derived(self, request, decomposition, children):
        context = self._context.with_data(
            decomposition=decomposition,
            children=children,
        )
        for resolver in self.resolvers.candidates(request, context, derived=True):
            attempt = self.policy.attempt_decision(
                request,
                resolver_calls=self._usage.resolver_calls,
            )
            if not attempt.allowed:
                self._stop(request.key, attempt)
                return None
            self._usage.record_resolver_call()
            try:
                raw = resolver.resolve(
                    request,
                    context,
                    children=children,
                    decomposition=decomposition,
                )
            except TypeError:
                raw = resolver.resolve(request, context)
            result = raw if isinstance(raw, ResolverResult) else ResolverResult(estimate=raw)
            self._usage.add(
                result.cost,
                external=result.external_calls,
                model=result.model_calls,
                expected_currency=self._cost_currency(request),
                count_resolver=False,
            )
            if not result.resolved or result.estimate is None:
                continue
            if result.estimate.key != request.key:
                continue
            if not self._within_limits(request):
                self._stop(request.key, self._limit_decision(request))
                return None
            self._set_node(
                request.key,
                status=FactorResolutionStatus.DERIVED,
                estimate=result.estimate,
                resolver=result.estimate.resolver,
                cache_state=CacheState.MISS,
                stop_reason=StopReason.SUFFICIENT_CONFIDENCE,
                warnings=result.warnings,
            )
            self.cache.put(result.estimate)
            return result.estimate
        return None

    def _within_limits(self, request) -> bool:
        budget = request.budget
        max_cost = self.policy.max_resolution_cost
        max_external = self.policy.max_external_calls
        max_model = self.policy.max_model_calls
        if budget is not None:
            if budget.max_resolution_cost is not None:
                max_cost = (
                    budget.max_resolution_cost
                    if max_cost is None
                    else min(max_cost, budget.max_resolution_cost)
                )
            elif budget.max_cost is not None:
                max_cost = budget.max_cost if max_cost is None else min(max_cost, budget.max_cost)
            if budget.max_external_calls is not None:
                max_external = (
                    budget.max_external_calls
                    if max_external is None
                    else min(max_external, budget.max_external_calls)
                )
            if budget.max_model_calls is not None:
                max_model = (
                    budget.max_model_calls
                    if max_model is None
                    else min(max_model, budget.max_model_calls)
                )
        return not (
            (max_cost is not None and self._usage.resolution_cost > max_cost)
            or (
                max_external is not None
                and self._usage.external_calls > max_external
            )
            or (max_model is not None and self._usage.model_calls > max_model)
        )

    def _limit_decision(self, request):
        return self.policy.decision(
            request,
            depth=0,
            nodes=len(self.graph),
            resolution_cost=self._usage.resolution_cost,
            external_calls=self._usage.external_calls,
            model_calls=self._usage.model_calls,
            resolver_calls=self._usage.resolver_calls,
        )

    def _cost_currency(self, request: FactorRequest) -> str:
        """Use the request budget currency when one is supplied.

        Resolution costs are not factor values, so they default to the
        policy's configured currency rather than inheriting the factor key's
        currency. A request budget is the most specific caller constraint.
        """

        if request.budget is not None:
            return request.budget.currency
        return self.policy.cost_currency

    def _stop(self, key, decision: FactorExpansionDecision) -> None:
        status = (
            FactorResolutionStatus.UNRESOLVED
            if decision.reason
            in {
                StopReason.NO_RESOLVER,
                StopReason.NO_DECOMPOSITION,
                StopReason.UNRESOLVED_DEPENDENCIES,
            }
            else FactorResolutionStatus.STOPPED
        )
        self._set_node(
            key,
            status=status,
            stop_reason=decision.reason,
            warnings=(decision.detail,) if decision.detail else (),
        )

    def _set_node(self, key, **updates) -> None:
        self.graph.update_node(key, **updates)

    def _memoize(self, request: FactorRequest, estimate: Optional[FactorEstimate]) -> None:
        digest = request.key.digest
        self._memo[digest] = estimate
        if estimate is None:
            self._memo_strength[digest] = (
                _PRIORITY_RANK[request.priority.value],
                _MATERIALITY_RANK[request.materiality.value],
            )
        else:
            self._memo_strength.pop(digest, None)


__all__ = [
    "FactorResolution",
    "FactorResolutionContext",
    "FactorResolutionResult",
    "ResolutionContext",
    "ResolutionResult",
    "RecursiveFactorResolutionService",
]
