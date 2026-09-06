"""Focused synthetic coverage for the remaining factor foundation seams."""

import datetime as dt
from decimal import Decimal

from edgarito.schemas.operating_graph import UnresolvedLeafRequirement
from edgarito.services.forecasting.reasoning import ForecastReasoningInput
from edgarito.services.forecasting.reasoning.factor_context import (
    build_factor_evidence_catalog,
)
from edgarito.services.valuation.factors import (
    DerivedFactorResolver,
    DirectEvidenceResolver,
    EconomicLeafFactorAdapter,
    FactorBudget,
    FactorConfidence,
    FactorCost,
    FactorDecompositionProposal,
    FactorDomain,
    FactorEstimate,
    FactorEvidence,
    FactorExpansionPolicy,
    FactorFreshnessMode,
    FactorFreshnessPolicy,
    FactorFreshnessRule,
    FactorKey,
    FactorPeriod,
    FactorRange,
    FactorRequest,
    FactorResolutionStatus,
    RecursiveFactorResolutionService,
    ResolvedFactorReasoningAdapter,
    ResolverResult,
    SemanticFactorCache,
    StaticMappingDecomposer,
    StopReason,
    valuation_root_keys,
)

D = Decimal
AS_OF = dt.date(2025, 1, 1)


def _period(year=2028):
    return FactorPeriod(
        target_year=year,
        period_type="FY",
        period_key=f"FY {year}",
    )


def _key(
    subject,
    metric,
    *,
    domain=FactorDomain.COMPANY,
    unit="USD",
    currency="USD",
    period=None,
):
    domain = FactorDomain(domain)
    return FactorKey(
        domain=domain,
        subject_type="commodity" if domain is FactorDomain.COMMODITY else "company",
        subject_id=subject,
        metric=metric,
        period=period or _period(),
        unit=unit,
        currency=currency,
    )


def _evidence(key, low, base=None, high=None, *, available=AS_OF, source="synthetic"):
    base = D(base if base is not None else low)
    high = D(high if high is not None else base)
    return FactorEvidence(
        key=key,
        low=D(low),
        base=base,
        high=high,
        information_available_on=available,
        observed_on=available,
        source=source,
        evidence_id=f"evidence-{key.metric}",
        provenance=source,
        confidence=FactorConfidence.HIGH,
    )


def _estimate(
    key,
    factor_range,
    *,
    dependencies=(),
    source="synthetic",
    evidence_refs=(),
    info_as_of=AS_OF,
    created_at=AS_OF,
    immutable=False,
    expires_at=None,
    version=1,
    dependency_fingerprints=None,
):
    dependencies = tuple(dependencies)
    dependency_fingerprints = dependency_fingerprints or {}
    return FactorEstimate(
        key=key,
        range=factor_range,
        unit=key.unit,
        currency=key.currency,
        info_as_of=info_as_of,
        target_period=key.period,
        confidence=FactorConfidence.HIGH,
        methodology="synthetic deterministic synthesis",
        resolver=source,
        evidence_refs=tuple(evidence_refs),
        dependencies=dependencies,
        dependency_fingerprints=dependency_fingerprints,
        all_availability_dates=(info_as_of,),
        created_at=created_at,
        immutable=immutable,
        expires_at=expires_at,
        source=source,
        version=version,
    )


class _CountingDirectEvidenceResolver(DirectEvidenceResolver):
    resolver_id = "counting_direct_evidence"

    def __init__(self, evidence=()):
        super().__init__(evidence)
        self.calls = []

    def resolve(self, request, context=None, **kwargs):
        self.calls.append(request.key)
        return super().resolve(request, context, **kwargs)


class _DimensionalSynthesisResolver:
    """Use explicit domain formulas instead of pretending units are fungible."""

    resolver_id = "dimensional_synthesis"
    requires_dependencies = True

    def can_resolve(self, request, context=None):
        return request.key.metric in {
            "battery_cost",
            "automotive_gross_margin",
            "storage_gross_margin",
        }

    def resolve(self, request, context=None, *, children=None, **kwargs):
        children = children or getattr(context, "children", {})
        decomposition = kwargs.get("decomposition") or getattr(
            context, "decomposition", None
        )
        proposals = decomposition.proposals
        resolved = [children[proposal.child_key.digest] for proposal in proposals]
        dependencies = tuple(estimate.key for estimate in resolved)
        fingerprints = {
            estimate.key.digest: estimate.fingerprint for estimate in resolved
        }
        if request.key.metric == "battery_cost":
            lithium = resolved[0]
            cell = resolved[1] if len(resolved) > 1 else None
            # Lithium is USD/tonne and cell cost is USD/kWh.  The conversion
            # factor is intentionally explicit and belongs to this resolver.
            value_range = FactorRange(
                low=(cell.range.low if cell else D("0"))
                + lithium.range.low * D("0.01"),
                base=(cell.range.base if cell else D("0"))
                + lithium.range.base * D("0.01"),
                high=(cell.range.high if cell else D("0"))
                + lithium.range.high * D("0.01"),
            )
        else:
            (battery,) = resolved
            # The margin formula uses a separate explicit battery-cost burden;
            # generic derived arithmetic must not cross these dimensions.
            burden = D("0.5")
            value_range = FactorRange(
                low=D("100") - battery.range.high * burden,
                base=D("100") - battery.range.base * burden,
                high=D("100") - battery.range.low * burden,
            )
        return _estimate(
            request.key,
            value_range,
            dependencies=dependencies,
            dependency_fingerprints=fingerprints,
            source="dimensional_synthesis",
            evidence_refs=(f"synthesis-{request.key.metric}",),
            info_as_of=max(estimate.info_as_of for estimate in resolved),
            created_at=getattr(context, "evaluated_at", request.information_as_of),
        )


def _evco_graph():
    lithium = _key(
        "lithium",
        "lithium_carbonate_price",
        domain=FactorDomain.COMMODITY,
        unit="USD / tonne",
    )
    cell = _key("evco", "cell_manufacturing_cost", unit="USD / kWh")
    battery = _key("evco", "battery_cost", unit="USD / kWh")
    automotive = _key("evco", "automotive_gross_margin", unit="percent", currency=None)
    storage = _key("evco", "storage_gross_margin", unit="percent", currency=None)
    mapping = {
        battery: {
            "operation": "IDENTITY",
            "proposals": [
                FactorDecompositionProposal(
                    child_key=lithium, rationale="lithium input"
                ),
                FactorDecompositionProposal(child_key=cell, rationale="cell input"),
            ],
        },
        automotive: [FactorDecompositionProposal(child_key=battery)],
        storage: [FactorDecompositionProposal(child_key=battery)],
    }
    direct = _CountingDirectEvidenceResolver(
        [
            _evidence(
                lithium,
                "90",
                "100",
                "110",
                source="global-lithium-series",
            ),
            _evidence(
                cell,
                "60",
                "70",
                "80",
                source="evco-cell-series",
            ),
        ]
    )
    service = RecursiveFactorResolutionService(
        resolvers=[direct, _DimensionalSynthesisResolver()],
        decomposers=[StaticMappingDecomposer(mapping)],
    )
    return service, direct, lithium, cell, battery, automotive, storage


def test_shared_battery_synthesis_is_dimensional_and_cached_for_two_roots():
    service, direct, lithium, cell, battery, automotive, storage = _evco_graph()
    result = service.resolve(
        [
            FactorRequest(key=automotive, information_as_of=AS_OF),
            FactorRequest(key=storage, information_as_of=AS_OF),
        ],
        evaluated_at=AS_OF,
    )

    assert result.resolved
    assert sum(key == lithium for key in direct.calls) == 1
    assert sum(key == cell for key in direct.calls) == 1
    assert (
        len([node for node in result.graph.nodes.values() if node.key == lithium]) == 1
    )
    assert (
        len([node for node in result.graph.nodes.values() if node.key == battery]) == 1
    )
    assert result.graph.node(battery).estimate.range == FactorRange(
        low=D("60.9"), base=D("71"), high=D("81.1")
    )
    for root in (automotive, storage):
        estimate = result.for_key(root)
        assert estimate is not None
        assert estimate.range == FactorRange(
            low=D("59.45"), base=D("64.5"), high=D("69.55")
        )
        assert estimate.dependencies == (battery,)
        assert estimate.dependency_fingerprint_map[
            battery.digest
        ] == battery_estimate_fingerprint(result, battery)
        assert estimate.source == "dimensional_synthesis"
        assert estimate.evidence_refs == (f"synthesis-{root.metric}",)
        assert len(result.graph.dependency_paths(root)) == 2
    assert len(service.cache.history(battery)) == 1
    assert len(service.cache.history(automotive)) == 1
    assert len(service.cache.history(storage)) == 1
    cached = service.resolve(
        [
            FactorRequest(key=automotive, information_as_of=AS_OF),
            FactorRequest(key=storage, information_as_of=AS_OF),
        ],
        evaluated_at=AS_OF,
    )
    assert all(
        cached.graph.node(root).status is FactorResolutionStatus.CACHE_HIT
        for root in (automotive, storage)
    )
    assert all(
        len(cached.graph.dependency_paths(root)) == 2 for root in (automotive, storage)
    )
    assert sum(key == lithium for key in direct.calls) == 1


def battery_estimate_fingerprint(result, key):
    estimate = result.graph.node(key).estimate
    assert estimate is not None
    return estimate.fingerprint


def test_second_company_reuses_global_lithium_cache_but_not_battery_identity():
    lithium = _key(
        "lithium",
        "lithium_carbonate_price",
        domain=FactorDomain.COMMODITY,
        unit="USD / tonne",
    )
    battery_a = _key("company-a", "battery_cost", unit="USD / kWh")
    battery_b = _key("company-b", "battery_cost", unit="USD / kWh")
    mapping = {
        battery_a: [FactorDecompositionProposal(child_key=lithium)],
        battery_b: [FactorDecompositionProposal(child_key=lithium)],
    }
    cache = SemanticFactorCache()
    first_direct = _CountingDirectEvidenceResolver([_evidence(lithium, "100")])
    first = RecursiveFactorResolutionService(
        resolvers=[first_direct, _DimensionalSynthesisResolver()],
        decomposers=[StaticMappingDecomposer(mapping)],
        cache=cache,
    )
    first_result = first.resolve(FactorRequest(key=battery_a, information_as_of=AS_OF))
    assert first_result.resolved
    assert first_direct.calls == [lithium]

    second_direct = _CountingDirectEvidenceResolver([])
    second = RecursiveFactorResolutionService(
        resolvers=[second_direct, _DimensionalSynthesisResolver()],
        decomposers=[StaticMappingDecomposer(mapping)],
        cache=cache,
    )
    second_result = second.resolve(
        FactorRequest(key=battery_b, information_as_of=AS_OF),
        evaluated_at=AS_OF,
    )
    assert second_result.resolved
    assert second_direct.calls == []
    assert second_result.graph.node(lithium).status is FactorResolutionStatus.CACHE_HIT
    assert battery_a != battery_b
    assert first_result.estimate.key == battery_a
    assert second_result.estimate.key == battery_b
    assert len(cache.history(lithium)) == 1


def _chain(prefix="chain"):
    values = tuple(_key(prefix, letter) for letter in "abcde")
    mapping = {
        parent: [FactorDecompositionProposal(child_key=child)]
        for parent, child in zip(values[:-1], values[1:], strict=True)
    }
    return values, mapping


def test_chain_limits_mark_the_exact_stopped_node_and_reason():
    values, mapping = _chain()
    root, _, _, _, _ = values
    depth = RecursiveFactorResolutionService(
        decomposers=[StaticMappingDecomposer(mapping)],
        policy=FactorExpansionPolicy(max_depth=3),
    ).resolve(FactorRequest(key=root, information_as_of=AS_OF))
    assert depth.graph.node(values[3]).stop_reason is StopReason.MAX_DEPTH
    assert depth.graph.node(values[4]) is None

    nodes = RecursiveFactorResolutionService(
        decomposers=[StaticMappingDecomposer(mapping)],
        policy=FactorExpansionPolicy(max_nodes=3),
    ).resolve(FactorRequest(key=root, information_as_of=AS_OF))
    assert nodes.graph.node(values[2]).stop_reason is StopReason.MAX_NODES

    class CostlyDecomposer(StaticMappingDecomposer):
        def decompose(self, request, context=None):
            result = super().decompose(request, context)
            return result.model_copy(
                update={"cost": FactorCost(amount="2", currency="USD")}
            )

    cost = RecursiveFactorResolutionService(
        decomposers=[CostlyDecomposer(mapping)],
        policy=FactorExpansionPolicy(max_resolution_cost=D("1")),
    ).resolve(FactorRequest(key=root, information_as_of=AS_OF))
    assert cost.graph.node(values[1]).stop_reason is StopReason.MAX_RESOLUTION_COST

    class ExternalDecomposer(StaticMappingDecomposer):
        def decompose(self, request, context=None):
            result = super().decompose(request, context)
            return result.model_copy(update={"external_calls": 1})

    external = RecursiveFactorResolutionService(
        decomposers=[ExternalDecomposer(mapping)],
        policy=FactorExpansionPolicy(max_external_calls=1),
    ).resolve(FactorRequest(key=root, information_as_of=AS_OF))
    assert external.graph.node(values[1]).stop_reason is StopReason.MAX_EXTERNAL_CALLS


def test_low_materiality_branch_stops_while_high_branch_expands():
    root = _key("branches", "total")
    low = _key("branches", "low")
    high = _key("branches", "high")
    mapping = {
        root: {
            "operation": "IDENTITY",
            "proposals": [
                FactorDecompositionProposal(
                    child_key=low,
                    materiality="low",
                    required=False,
                ),
                FactorDecompositionProposal(child_key=high, materiality="high"),
            ],
        }
    }
    result = RecursiveFactorResolutionService(
        resolvers=[
            DirectEvidenceResolver([_evidence(high, "10")]),
            DerivedFactorResolver(),
        ],
        decomposers=[StaticMappingDecomposer(mapping)],
        policy=FactorExpansionPolicy(min_materiality="medium"),
    ).resolve(FactorRequest(key=root, information_as_of=AS_OF))
    assert result.resolved
    assert result.graph.node(low).stop_reason is StopReason.BELOW_MATERIALITY
    assert result.graph.node(high).estimate is not None


def test_derived_resolver_budget_failures_keep_the_limit_reason():
    _, _, _, _, battery, _, _ = _evco_graph()

    class MeteredSynthesis(_DimensionalSynthesisResolver):
        def __init__(self, *, cost=0, external_calls=0):
            self.cost = cost
            self.external_calls = external_calls

        def resolve(self, request, context=None, **kwargs):
            estimate = super().resolve(request, context, **kwargs)
            return ResolverResult.success(
                estimate,
                cost=self.cost,
                external_calls=self.external_calls,
            )

    mapping = {
        battery: [
            FactorDecompositionProposal(
                child_key=_key(
                    "lithium",
                    "lithium_carbonate_price",
                    domain=FactorDomain.COMMODITY,
                    unit="USD / tonne",
                )
            )
        ]
    }
    evidence_key = mapping[battery][0].child_key
    cost = RecursiveFactorResolutionService(
        resolvers=[
            DirectEvidenceResolver([_evidence(evidence_key, "100")]),
            MeteredSynthesis(cost="2"),
        ],
        decomposers=[StaticMappingDecomposer(mapping)],
        policy=FactorExpansionPolicy(max_resolution_cost=D("1")),
    ).resolve(FactorRequest(key=battery, information_as_of=AS_OF))
    assert cost.graph.node(battery).stop_reason is StopReason.MAX_RESOLUTION_COST

    external = RecursiveFactorResolutionService(
        resolvers=[
            DirectEvidenceResolver([_evidence(evidence_key, "100")]),
            MeteredSynthesis(external_calls=2),
        ],
        decomposers=[StaticMappingDecomposer(mapping)],
        policy=FactorExpansionPolicy(max_external_calls=1),
    ).resolve(FactorRequest(key=battery, information_as_of=AS_OF))
    assert external.graph.node(battery).stop_reason is StopReason.MAX_EXTERNAL_CALLS


def test_mismatched_resolver_cost_currency_fails_without_summing():
    factor_key = _key("cost", "factor")
    evidence = _evidence(factor_key, "10")
    estimate = DirectEvidenceResolver([evidence]).resolve(
        FactorRequest(key=factor_key, information_as_of=AS_OF)
    ).estimate

    class EuroCostResolver:
        resolver_id = "euro_cost"
        requires_dependencies = False

        def can_resolve(self, request, context=None):
            return True

        def resolve(self, request, context=None):
            return ResolverResult.success(
                estimate,
                cost=FactorCost(amount="1", currency="EUR"),
            )

    result = RecursiveFactorResolutionService(
        resolvers=[EuroCostResolver()]
    ).resolve(
        FactorRequest(
            key=factor_key,
            information_as_of=AS_OF,
            budget=FactorBudget(max_cost=D("2"), currency="USD"),
        )
    )

    node = result.graph.node(factor_key)
    assert not result.resolved
    assert result.metrics.resolver_calls == 1
    assert result.metrics.resolution_cost == D("0")
    assert node.stop_reason is StopReason.FAILED
    assert "currency" in node.warnings[0]


def test_matching_resolver_cost_currency_is_tracked():
    factor_key = _key("cost", "factor")
    evidence = _evidence(factor_key, "10")
    estimate = DirectEvidenceResolver([evidence]).resolve(
        FactorRequest(key=factor_key, information_as_of=AS_OF)
    ).estimate

    class UsdCostResolver:
        resolver_id = "usd_cost"
        requires_dependencies = False

        def can_resolve(self, request, context=None):
            return True

        def resolve(self, request, context=None):
            return ResolverResult.success(
                estimate,
                cost=FactorCost(amount="1", currency="USD"),
            )

    result = RecursiveFactorResolutionService(
        resolvers=[UsdCostResolver()]
    ).resolve(
        FactorRequest(
            key=factor_key,
            information_as_of=AS_OF,
            budget=FactorBudget(max_cost=D("2"), currency="USD"),
        )
    )

    assert result.resolved
    assert result.metrics.resolution_cost == D("1")


def test_cache_history_is_point_in_time_and_dependency_versions_are_immutable():
    commodity = _key(
        "lithium",
        "lithium_carbonate_price",
        domain=FactorDomain.COMMODITY,
        unit="USD / tonne",
    )
    cache = SemanticFactorCache()
    old = _estimate(
        commodity,
        FactorRange.from_point(D("100")),
        info_as_of=dt.date(2024, 1, 1),
        created_at=dt.date(2024, 1, 1),
        version=2,
    )
    new = _estimate(
        commodity,
        FactorRange.from_point(D("200")),
        info_as_of=dt.date(2025, 1, 1),
        created_at=dt.date(2025, 1, 1),
        version=1,
    )
    cache.put(old)
    cache.put(new)
    policy = FactorFreshnessPolicy(
        rules=(
            FactorFreshnessRule(
                domain=FactorDomain.COMMODITY,
                mode=FactorFreshnessMode.NONE,
                ttl=None,
            ),
        )
    )
    historical = cache.lookup(
        FactorRequest(key=commodity, information_as_of=dt.date(2024, 12, 31)),
        policy,
        dt.date(2025, 2, 1),
    )
    assert historical.hit and historical.estimate.fingerprint == old.fingerprint
    assert cache.lookup(
        FactorRequest(key=commodity, information_as_of=dt.date(2024, 12, 31)),
        policy,
        dt.date(2025, 2, 1),
    ).estimate.range.base == D("100")
    assert cache.history(commodity) == (old, new)

    expiring_direct = _CountingDirectEvidenceResolver([_evidence(commodity, "100")])
    expiring = RecursiveFactorResolutionService(
        resolvers=[expiring_direct], cache=cache
    )
    # The cached synthetic values are not direct-evidence values, so use a
    # fresh key for the TTL rerun assertion.
    spot = commodity.model_copy(update={"subject_id": "spot-lithium"})
    expiring_direct.evidence = (_evidence(spot, "100"),)
    expiring.resolve(
        FactorRequest(key=spot, information_as_of=dt.date(2025, 1, 1)),
        evaluated_at=dt.date(2025, 1, 1),
    )
    expiring.resolve(
        FactorRequest(key=spot, information_as_of=dt.date(2025, 1, 20)),
        evaluated_at=dt.date(2025, 1, 20),
    )
    assert expiring_direct.calls == [spot, spot]

    immutable = _estimate(
        commodity,
        FactorRange.from_point(D("50")),
        info_as_of=dt.date(2024, 1, 1),
        created_at=dt.date(2024, 1, 1),
        immutable=True,
    )
    immutable_cache = SemanticFactorCache()
    immutable_cache.put(immutable)
    assert immutable_cache.lookup(
        FactorRequest(key=commodity, information_as_of=dt.date(2024, 12, 31)),
        evaluated_at=dt.date(2030, 1, 1),
    ).hit

    dependency = _key("dependency", "input")
    parent = _key("parent", "output")
    service = RecursiveFactorResolutionService(
        resolvers=[
            DirectEvidenceResolver(
                [
                    _evidence(dependency, "10", available=dt.date(2024, 1, 1)),
                    _evidence(dependency, "20", available=dt.date(2025, 1, 1)),
                ]
            ),
            DerivedFactorResolver(),
        ],
        decomposers=[
            StaticMappingDecomposer(
                {parent: [FactorDecompositionProposal(child_key=dependency)]}
            )
        ],
    )
    first = service.resolve(
        FactorRequest(key=parent, information_as_of=dt.date(2024, 12, 31)),
        evaluated_at=dt.date(2024, 12, 31),
    )
    second = service.resolve(
        FactorRequest(key=parent, information_as_of=dt.date(2025, 12, 31)),
        evaluated_at=dt.date(2025, 12, 31),
    )
    older = service.resolve(
        FactorRequest(key=parent, information_as_of=dt.date(2024, 12, 31)),
        evaluated_at=dt.date(2024, 12, 31),
    )
    assert first.estimate.range.base == D("10")
    assert second.estimate.range.base == D("20")
    assert older.graph.node(parent).status is FactorResolutionStatus.CACHE_HIT
    assert older.estimate.range.base == D("10")
    assert len(service.cache.history(parent)) == 2


def test_economic_leaf_resolves_into_combined_factor_catalog_without_overrides():
    requirement = UnresolvedLeafRequirement(
        node_id="price_per_call",
        fiscal_year=2026,
        reason="missing pricing leaf",
        path=("revenue", "price_per_call"),
        metric="price_per_call",
        unit="USD / call",
        currency="USD",
        required_by_relationship_ids=("revenue-identity",),
    )
    request = EconomicLeafFactorAdapter("ACME", information_as_of=AS_OF).adapt(
        requirement
    )
    evidence = _evidence(request.key, "8", "10", "12", source="pricing-survey")
    resolution = RecursiveFactorResolutionService(
        resolvers=[DirectEvidenceResolver([evidence])]
    ).resolve(request)
    estimate = resolution.estimate
    assert estimate is not None
    assert estimate.key == request.key
    assert estimate.source == "pricing_survey"
    context = ResolvedFactorReasoningAdapter().to_context(estimate)
    base = ForecastReasoningInput(
        company_id="ACME",
        unit="USD",
        as_of=AS_OF,
        forecast_years=(2026,),
    )
    augmented = ResolvedFactorReasoningAdapter().augment(base, context)
    catalog = build_factor_evidence_catalog(augmented)
    item = next(item for item in catalog.items if item.category == "FACTOR")
    assert item.evidence_id == f"FACTOR-{estimate.fingerprint}"
    assert item.low == D("8") and item.base == D("10") and item.high == D("12")
    assert item.payload_identity == estimate.fingerprint
    assert item.provenance.source == "pricing_survey"
    assert augmented.manual_overrides == ()
    assert context.items[0].key == request.key
    assert context.items[0].fingerprint == estimate.fingerprint


def test_explain_exposes_stable_ancestors_descendants_and_dependency_paths():
    service, _, lithium, cell, battery, automotive, storage = _evco_graph()
    result = service.resolve(
        [
            FactorRequest(key=automotive, information_as_of=AS_OF),
            FactorRequest(key=storage, information_as_of=AS_OF),
        ]
    )
    explanation = result.graph.explain(battery)
    assert {item.metric for item in explanation.ancestors} == {
        "lithium_carbonate_price",
        "cell_manufacturing_cost",
    }
    assert {item.metric for item in explanation.descendants} == {
        "automotive_gross_margin",
        "storage_gross_margin",
    }
    assert explanation.dependency_paths == explanation.paths
    assert {
        tuple(item.metric for item in path) for path in explanation.dependency_paths
    } == {
        ("battery_cost", "lithium_carbonate_price"),
        ("battery_cost", "cell_manufacturing_cost"),
    }
    assert {edge.dependency for edge in explanation.edges} >= {lithium, cell}


def test_valuation_root_keys_are_distinct_and_round_trip_serializable():
    roots = valuation_root_keys(company_id="ACME", period=_period(), currency="USD")
    assert len(roots) == 7
    assert len({root.digest for root in roots}) == 7
    assert {root.metric for root in roots} == {
        "risk_free_rate",
        "equity_risk_premium",
        "beta",
        "debt_spread",
        "target_capital_structure",
        "terminal_growth",
        "terminal_roic",
    }
    assert roots[0].subject_id == roots[1].subject_id == "global"
    assert all(
        FactorKey.model_validate_json(root.model_dump_json()) == root for root in roots
    )


def test_factor_request_budget_contract_round_trips_limits():
    key = _key("budget", "factor")
    request = FactorRequest(
        key=key,
        information_as_of=AS_OF,
        budget=FactorBudget(
            max_depth=3,
            max_nodes=5,
            max_resolution_cost=D("2"),
            max_external_calls=1,
        ),
    )
    assert request.budget.max_depth == 3
    assert request.budget.max_nodes == 5
