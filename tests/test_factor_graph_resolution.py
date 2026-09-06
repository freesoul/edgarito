import datetime as dt
from decimal import Decimal

import pytest

from edgarito.services.valuation.factors import (
    CacheState,
    DerivedFactorResolver,
    DirectEvidenceResolver,
    DirectFinancialEvidenceResolver,
    ExistingResearchEvidenceResolver,
    FactorBudget,
    FactorDecomposerRegistry,
    FactorDecompositionProposal,
    FactorEvidence,
    FactorExpansionPolicy,
    FactorKey,
    FactorPeriod,
    FactorRequest,
    FactorResolutionResult,
    FactorResolutionStatus,
    FactorResolverRegistry,
    ManagementGuidanceFactorResolver,
    RecursiveFactorResolutionService,
    ResolverResult,
    StaticMappingDecomposer,
    StopReason,
    ValuationFactorGraph,
)


def _period(year=2025):
    return FactorPeriod(target_year=year, period_type="FY", period_key=f"FY {year}")


def _key(subject, metric, *, domain="company", geography=None):
    return FactorKey(
        domain=domain,
        subject_type="company" if domain == "company" else "commodity",
        subject_id=subject,
        metric=metric,
        geography=geography,
        period=_period(),
        unit="percent",
        currency=None,
    )


def _evidence(key, value, *, available=dt.date(2024, 1, 1), source="test"):
    return FactorEvidence(
        key=key,
        point=Decimal(value),
        information_available_on=available,
        observed_on=available,
        source=source,
        confidence="high",
    )


def _request(key, **changes):
    values = {"key": key, "information_as_of": dt.date(2025, 1, 1)}
    values.update(changes)
    return FactorRequest(**values)


def test_graph_deduplicates_and_rejects_cycles_before_commit():
    first, second, third = (_key(item, "metric") for item in "abc")
    graph = ValuationFactorGraph()
    graph.add_edge(first, second)
    graph.add_edge(second, third)
    with pytest.raises(ValueError):
        graph.add_edge(third, first)
    assert len(graph.edges) == 2
    assert set(graph.ancestors(third)) == {first, second}
    assert len(graph.snapshot().nodes) == 3


def test_shared_dependencies_are_resolved_once_and_provenance_is_stable():
    lithium = _key("lithium", "price", domain="commodity", geography="global")
    battery = _key("ev", "battery")
    vehicle = _key("ev", "vehicle")
    storage = _key("ev", "storage")
    mapping = {
        battery: [FactorDecompositionProposal(child_key=lithium)],
        vehicle: [FactorDecompositionProposal(child_key=battery)],
        storage: [FactorDecompositionProposal(child_key=battery)],
    }
    direct = DirectEvidenceResolver([_evidence(lithium, "10")])
    service = RecursiveFactorResolutionService(
        resolvers=[direct, DerivedFactorResolver()],
        decomposers=[StaticMappingDecomposer(mapping)],
    )
    result = service.resolve([_request(vehicle), _request(storage)])
    assert result.resolved
    assert result.metrics.resolver_calls == 4
    assert result.graph.node(lithium).status is FactorResolutionStatus.DIRECTLY_RESOLVED
    assert result.graph.dependency_paths(vehicle) == result.graph.dependency_paths(vehicle)
    assert result.graph.node(battery).estimate.dependencies == (lithium,)


@pytest.mark.parametrize("operation", ("IDENTITY", "ADD", "SUBTRACT", "WEIGHTED_SUM"))
def test_derived_operations_fail_closed_on_unit_and_period_mismatch(operation):
    parent = _key("acme", "total").model_copy(
        update={
            "unit": "usd_per_metric_tonne",
            "currency": "USD",
            "period": _period(2027),
        }
    )
    compatible = parent.model_copy(update={"subject_id": "compatible"})
    incompatible = parent.model_copy(
        update={
            "subject_id": "incompatible",
            "unit": "usd_per_kwh",
            "period": _period(2028),
        }
    )
    proposals = [
        FactorDecompositionProposal(child_key=incompatible)
    ] if operation == "IDENTITY" else [
        FactorDecompositionProposal(child_key=compatible),
        FactorDecompositionProposal(child_key=incompatible, weight="0.5"),
    ]
    service = RecursiveFactorResolutionService(
        resolvers=[
            DirectEvidenceResolver(
                [_evidence(compatible, "10"), _evidence(incompatible, "20")]
            ),
            DerivedFactorResolver(),
        ],
        decomposers=[
            StaticMappingDecomposer(
                {parent: {"operation": operation, "proposals": proposals}}
            )
        ],
    )

    result = service.resolve(_request(parent))

    assert not result.resolved
    assert result.graph.node(parent).estimate is None


def test_derived_identity_rejects_currency_mismatch():
    parent = _key("acme", "total").model_copy(update={"currency": "USD"})
    child = parent.model_copy(update={"subject_id": "child", "currency": "EUR"})
    service = RecursiveFactorResolutionService(
        resolvers=[DirectEvidenceResolver([_evidence(child, "10")]), DerivedFactorResolver()],
        decomposers=[
            StaticMappingDecomposer(
                {parent: [FactorDecompositionProposal(child_key=child)]}
            )
        ],
    )

    assert not service.resolve(_request(parent)).resolved


def test_planner_expands_high_materiality_branch_when_low_branch_is_listed_first():
    root = _key("acme", "total")
    low_branch = _key("acme", "low_branch")
    high_branch = _key("acme", "high_branch")
    shared = _key("acme", "shared")
    leaf = _key("acme", "leaf")
    mapping = {
        root: {
            "operation": "ADD",
            "proposals": [
                FactorDecompositionProposal(
                    child_key=low_branch,
                    priority="low",
                    materiality="low",
                    required=False,
                ),
                FactorDecompositionProposal(
                    child_key=high_branch,
                    priority="high",
                    materiality="high",
                ),
            ],
        },
        low_branch: {
            "operation": "IDENTITY",
            "proposals": [
                FactorDecompositionProposal(
                    child_key=shared,
                    priority="low",
                    materiality="low",
                )
            ],
        },
        high_branch: {
            "operation": "IDENTITY",
            "proposals": [
                FactorDecompositionProposal(
                    child_key=shared,
                    priority="high",
                    materiality="high",
                )
            ],
        },
        shared: [FactorDecompositionProposal(child_key=leaf, priority="high")],
    }
    service = RecursiveFactorResolutionService(
        resolvers=[
            DirectEvidenceResolver([_evidence(leaf, "10")]),
            DerivedFactorResolver(),
        ],
        decomposers=[StaticMappingDecomposer(mapping)],
        policy=FactorExpansionPolicy(min_materiality="medium"),
    )

    result = service.resolve(_request(root))

    assert result.resolved
    assert result.graph.node(shared).estimate is not None
    assert result.graph.node(high_branch).estimate is not None


def test_expired_commodity_cache_reruns_direct_resolution():
    commodity = _key("lithium", "price", domain="commodity", geography="global")
    service = RecursiveFactorResolutionService(
        resolvers=[DirectEvidenceResolver([_evidence(commodity, "10")])]
    )
    service.resolve(
        _request(commodity, information_as_of=dt.date(2025, 1, 1)),
        evaluated_at=dt.date(2025, 1, 1),
    )

    refreshed = service.resolve(
        _request(commodity, information_as_of=dt.date(2025, 1, 20)),
        evaluated_at=dt.date(2025, 1, 20),
    )

    assert refreshed.resolved
    assert refreshed.metrics.resolver_calls == 1
    assert refreshed.graph.node(commodity).status is FactorResolutionStatus.DIRECTLY_RESOLVED


def test_bare_factor_key_roots_require_an_explicit_evaluation_date():
    with pytest.raises(ValueError, match="bare FactorKey"):
        RecursiveFactorResolutionService().resolve(_key("acme", "metric"))


def test_roots_with_different_information_dates_are_rejected():
    first = _request(_key("a", "metric"), information_as_of=dt.date(2025, 1, 1))
    second = _request(_key("b", "metric"), information_as_of=dt.date(2025, 2, 1))
    with pytest.raises(ValueError, match="same information_as_of"):
        RecursiveFactorResolutionService().resolve([first, second])


class _MissingDirectResolver:
    resolver_id = "a_missing"
    requires_dependencies = False

    def can_resolve(self, request, context=None):
        return True

    def resolve(self, request, context=None):
        return ResolverResult.missing(
            "not available", cost="2", external_calls=1
        )


def test_direct_resolution_falls_through_unresolved_resolvers_and_counts_usage():
    factor_key = _key("acme", "metric")
    evidence = _evidence(factor_key, "10")
    service = RecursiveFactorResolutionService(
        resolvers=[_MissingDirectResolver(), DirectEvidenceResolver([evidence])]
    )
    result = service.resolve(_request(factor_key))
    assert result.resolved
    assert result.metrics.resolver_calls == 2
    assert result.metrics.external_calls == 1
    assert result.metrics.resolution_cost == Decimal("2")


def test_registries_sort_by_explicit_priority_then_stable_id():
    class RegisteredResolver:
        def __init__(self, resolver_id, priority=0):
            self.resolver_id = resolver_id
            self.priority = priority

    class RegisteredDecomposer:
        def __init__(self, decomposer_id, priority=0):
            self.decomposer_id = decomposer_id
            self.priority = priority

    resolvers = FactorResolverRegistry(
        [RegisteredResolver("z"), RegisteredResolver("a"), RegisteredResolver("high", 2)]
    )
    decomposers = FactorDecomposerRegistry(
        [RegisteredDecomposer("z"), RegisteredDecomposer("a"), RegisteredDecomposer("high", 2)]
    )

    assert [item.resolver_id for item in resolvers.resolvers] == ["high", "a", "z"]
    assert [item.decomposer_id for item in decomposers.decomposers] == [
        "high",
        "a",
        "z",
    ]
    assert (
        ManagementGuidanceFactorResolver.priority
        > DirectFinancialEvidenceResolver.priority
        > ExistingResearchEvidenceResolver.priority
        > DirectEvidenceResolver.priority
    )


def test_unresolved_higher_priority_resolver_falls_through():
    factor_key = _key("acme", "metric")

    class MissingHighPriorityResolver:
        resolver_id = "high_priority_missing"
        priority = 10
        requires_dependencies = False

        def can_resolve(self, request, context=None):
            return True

        def resolve(self, request, context=None):
            return ResolverResult.missing("not available")

    service = RecursiveFactorResolutionService(
        resolvers=[
            DirectEvidenceResolver([_evidence(factor_key, "10")]),
            MissingHighPriorityResolver(),
        ]
    )
    result = service.resolve(_request(factor_key))

    assert result.resolved
    assert result.estimate.range.base == Decimal("10")
    assert result.metrics.resolver_calls == 2


def test_max_attempts_stops_before_another_resolver_call():
    factor_key = _key("acme", "metric")
    service = RecursiveFactorResolutionService(
        resolvers=[
            _MissingDirectResolver(),
            DirectEvidenceResolver([_evidence(factor_key, "10")]),
        ]
    )
    result = service.resolve(
        _request(factor_key, budget=FactorBudget(max_attempts=1))
    )

    assert not result.resolved
    assert result.metrics.resolver_calls == 1
    assert result.graph.node(factor_key).stop_reason is StopReason.BUDGET_EXHAUSTED


def test_direct_evidence_is_point_in_time_and_cached_derived_values_track_fingerprints():
    lithium = _key("lithium", "price", domain="commodity", geography="global")
    battery = _key("ev", "battery")
    old = _evidence(lithium, "10", available=dt.date(2024, 1, 1))
    new = _evidence(lithium, "20", available=dt.date(2025, 1, 1))
    direct = DirectEvidenceResolver([old, new])
    service = RecursiveFactorResolutionService(
        resolvers=[direct, DerivedFactorResolver()],
        decomposers=[
            StaticMappingDecomposer(
                {battery: [FactorDecompositionProposal(child_key=lithium)]}
            )
        ],
    )
    assert service.resolve(_request(lithium, information_as_of=dt.date(2024, 12, 31))).estimate.range.base == Decimal("10")
    first = service.resolve(_request(battery))
    assert first.estimate.range.base == Decimal("10")
    newer = direct.resolve(
        _request(lithium, information_as_of=dt.date(2025, 12, 31))
    )
    service.cache.put(newer.estimate)
    second = service.resolve(_request(battery))
    assert second.estimate.range.base == Decimal("20")
    older = service.resolve(
        _request(battery, information_as_of=dt.date(2024, 12, 31))
    )
    assert older.estimate.range.base == Decimal("10")


def test_global_cache_is_shared_across_requesters_but_company_keys_are_not():
    lithium = _key("lithium", "price", domain="commodity", geography="global")
    battery_a = _key("company-a", "battery")
    battery_b = _key("company-b", "battery")
    mapping = {
        battery_a: [FactorDecompositionProposal(child_key=lithium)],
        battery_b: [FactorDecompositionProposal(child_key=lithium)],
    }
    service = RecursiveFactorResolutionService(
        resolvers=[DirectEvidenceResolver([_evidence(lithium, "10")]), DerivedFactorResolver()],
        decomposers=[StaticMappingDecomposer(mapping)],
    )
    first = service.resolve(
        _request(lithium, requester="company-a", audit_context={"company": "a"})
    )
    second = service.resolve(
        _request(lithium, requester="company-b", audit_context={"company": "b"})
    )
    assert first.graph.node(lithium).status is FactorResolutionStatus.DIRECTLY_RESOLVED
    assert second.graph.node(lithium).status is FactorResolutionStatus.CACHE_HIT
    assert second.metrics.resolver_calls == 0

    resolved_a = service.resolve(_request(battery_a))
    resolved_b = service.resolve(_request(battery_b))
    assert resolved_a.resolved and resolved_b.resolved
    assert battery_a != battery_b
    assert resolved_a.estimate.key == battery_a
    assert resolved_b.estimate.key == battery_b


def test_direct_resolution_result_round_trips_its_immutable_audit_views():
    factor_key = _key("acme", "metric")
    result = RecursiveFactorResolutionService(
        resolvers=[DirectEvidenceResolver([_evidence(factor_key, "10.25")])]
    ).resolve(_request(factor_key))

    payload = result.model_dump_json()
    restored = FactorResolutionResult.model_validate_json(payload)

    assert restored == result
    assert restored.model_dump_json() == payload
    assert restored.estimate == result.estimate
    assert restored.for_key(factor_key) == result.for_key(factor_key)
    assert restored.graph.node(factor_key) == result.graph.node(factor_key)
    assert restored.graph.node(factor_key).cache_state is CacheState.MISS


def test_recursive_result_round_trip_preserves_graph_explanation_traversal():
    leaf = _key("leaf", "metric")
    middle = _key("middle", "metric")
    root = _key("root", "metric")
    service = RecursiveFactorResolutionService(
        resolvers=[DirectEvidenceResolver([_evidence(leaf, "10")]), DerivedFactorResolver()],
        decomposers=[
            StaticMappingDecomposer(
                {
                    root: [FactorDecompositionProposal(child_key=middle)],
                    middle: [FactorDecompositionProposal(child_key=leaf)],
                }
            )
        ],
    )
    result = service.resolve(_request(root))
    restored = FactorResolutionResult.model_validate_json(result.model_dump_json())

    assert restored.resolved
    assert restored.graph.ancestors(root) == result.graph.ancestors(root)
    assert restored.graph.dependency_paths(root) == result.graph.dependency_paths(root)
    assert restored.graph.explain(root) == result.graph.explain(root)


def test_cache_hit_result_round_trip_retains_status_stop_reason_and_cache_state():
    factor_key = _key("acme", "metric")
    service = RecursiveFactorResolutionService(
        resolvers=[DirectEvidenceResolver([_evidence(factor_key, "10")])]
    )
    service.resolve(_request(factor_key))
    result = service.resolve(_request(factor_key))
    restored = FactorResolutionResult.model_validate_json(result.model_dump_json())

    node = restored.graph.node(factor_key)
    assert node.status is FactorResolutionStatus.CACHE_HIT
    assert node.stop_reason is StopReason.FRESH_CACHE
    assert node.cache_state is CacheState.HIT


def test_result_json_order_is_independent_of_root_input_order():
    first = _key("a", "metric")
    second = _key("b", "metric")
    evidence = [_evidence(first, "1"), _evidence(second, "2")]

    def resolve(roots):
        result = RecursiveFactorResolutionService(
            resolvers=[DirectEvidenceResolver(evidence)]
        ).resolve([_request(root) for root in roots])
        return result.model_copy(
            update={
                "metrics": result.metrics.model_copy(update={"elapsed_seconds": None})
            }
        )

    assert resolve([first, second]).model_dump_json() == resolve(
        [second, first]
    ).model_dump_json()


def test_policy_stops_depth_nodes_and_materiality_expansion():
    leaf = _key("leaf", "metric")
    middle = _key("middle", "metric")
    root = _key("root", "metric")
    mapping = {
        root: [FactorDecompositionProposal(child_key=middle)],
        middle: [FactorDecompositionProposal(child_key=leaf)],
    }
    service = RecursiveFactorResolutionService(
        decomposers=[StaticMappingDecomposer(mapping)],
        policy=FactorExpansionPolicy(max_depth=1, max_nodes=10),
    )
    result = service.resolve(_request(root))
    assert result.graph.node(root).stop_reason is StopReason.MAX_DEPTH

    low = RecursiveFactorResolutionService(
        decomposers=[StaticMappingDecomposer(mapping)],
        policy=FactorExpansionPolicy(min_materiality="medium"),
    ).resolve(_request(root, materiality="low"))
    assert low.graph.node(root).stop_reason is StopReason.BELOW_MATERIALITY
