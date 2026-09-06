import datetime as dt
from datetime import timedelta
from decimal import Decimal

import pytest

from edgarito.services.valuation.factors import (
    CacheState,
    FactorConfidence,
    FactorDomain,
    FactorEstimate,
    FactorEvidence,
    FactorFreshnessMode,
    FactorFreshnessPolicy,
    FactorFreshnessRule,
    FactorGraphNode,
    FactorKey,
    FactorPeriod,
    FactorRange,
    FactorRequest,
    FreshnessReason,
    SemanticFactorCache,
)


def period(period_type="FY", year=2025):
    return FactorPeriod(
        target_year=year,
        period_type=period_type,
        period_key=f"{period_type} {year}",
        start=dt.date(year, 1, 1) if year is not None else None,
        end=dt.date(year, 12, 31) if year is not None else None,
    )


def key(**changes):
    values = {
        "domain": FactorDomain.COMPANY,
        "subject_type": "Company",
        "subject_id": " ACME ",
        "metric": "Operating Margin",
        "period": period(),
        "unit": "percent",
    }
    values.update(changes)
    return FactorKey(**values)


def estimate(factor_key, *, info=dt.date(2025, 1, 1), value="1", **changes):
    values = {
        "key": factor_key,
        "range": FactorRange.from_point(Decimal(value)),
        "unit": factor_key.unit,
        "currency": factor_key.currency,
        "info_as_of": info,
        "target_period": factor_key.period,
        "confidence": FactorConfidence.HIGH,
        "methodology": "test methodology",
        "resolver": "test resolver",
        "all_availability_dates": (info,),
        "created_at": info,
    }
    values.update(changes)
    return FactorEstimate(**values)


def request(factor_key, *, info=dt.date(2025, 12, 31), **changes):
    values = {
        "key": factor_key,
        "information_as_of": info,
        "min_confidence": FactorConfidence.MEDIUM,
    }
    values.update(changes)
    return FactorRequest(**values)


def test_key_canonicalization_preserves_economic_distinctions():
    first = key(geography="Worldwide", unit="%")
    second = key(geography=" global ", unit="percentage")
    assert first == second
    assert key(product="lithium carbonate") != key(product="lithium hydroxide")
    assert key(geography="United States") == key(geography="USA")
    assert key(geography="United Kingdom") != key(geography="UK")


def test_ambiguous_ton_units_do_not_collapse_into_metric_tonnes():
    carbonate_per_ton = key(
        domain=FactorDomain.COMMODITY,
        subject_type="commodity",
        subject_id="lithium",
        metric="price",
        product="lithium carbonate",
        unit="USD/ton",
        currency="USD",
    )
    carbonate_per_tonne = FactorKey.model_validate(
        {**carbonate_per_ton.model_dump(mode="python"), "unit": "USD/tonne"}
    )
    hydroxide_per_ton = FactorKey.model_validate(
        {
            **carbonate_per_ton.model_dump(mode="python"),
            "product": "lithium hydroxide",
        }
    )

    assert carbonate_per_ton.unit == "usd_per_ton"
    assert carbonate_per_tonne.unit == "usd_per_metric_tonne"
    assert carbonate_per_ton != carbonate_per_tonne
    assert carbonate_per_ton != hydroxide_per_ton


def test_graph_node_uncertainty_accepts_only_factor_ranges():
    factor_key = key()
    uncertainty = FactorRange.from_point(Decimal("0.25"))
    assert FactorGraphNode(key=factor_key, uncertainty=uncertainty).uncertainty == uncertainty


def test_period_type_is_part_of_identity():
    assert key(period=period("FY")) != key(period=period("FQ"))
    assert key(period=period("current_spot", year=None)) != key(
        period=FactorPeriod(period_type="long_term", period_key="long term")
    )


def test_global_factor_identity_has_no_requester_scope():
    global_key = key(
        domain=FactorDomain.COMMODITY,
        subject_type="commodity",
        subject_id="lithium",
        metric="price",
        geography="worldwide",
        unit="USD / tonne",
        currency="usd",
    )
    first = request(global_key, requester="company-a")
    second = request(global_key, requester="company-b")
    assert first.key == second.key
    assert first.key.digest == second.key.digest


def test_evidence_point_normalizes_to_ordered_range():
    evidence = FactorEvidence(
        key=key(),
        point=Decimal("3.25"),
        information_available_on=dt.date(2025, 2, 1),
        observed_on=dt.date(2025, 1, 31),
        source="test source",
        confidence="high",
    )
    assert evidence.range == FactorRange(
        low=Decimal("3.25"), base=Decimal("3.25"), high=Decimal("3.25")
    )


def test_point_in_time_selects_old_version_until_new_version_is_available():
    factor_key = key()
    cache = SemanticFactorCache()
    cache.put(estimate(factor_key, info=dt.date(2025, 1, 1), value="1", version=1))
    cache.put(estimate(factor_key, info=dt.date(2025, 2, 1), value="2", version=2))
    policy = FactorFreshnessPolicy()

    old = cache.lookup(
        request(factor_key, info=dt.date(2025, 1, 15)), policy, dt.date(2025, 3, 1)
    )
    new = cache.lookup(
        request(factor_key, info=dt.date(2025, 2, 15)), policy, dt.date(2025, 3, 1)
    )
    assert old.state is CacheState.HIT and old.estimate.range.base == Decimal("1")
    assert new.state is CacheState.HIT and new.estimate.range.base == Decimal("2")


def test_cache_orders_knowledge_before_arbitrary_version_numbers():
    factor_key = key()
    cache = SemanticFactorCache()
    cache.put(estimate(factor_key, info=dt.date(2025, 1, 1), value="1", version=99))
    cache.put(estimate(factor_key, info=dt.date(2025, 2, 1), value="2", version=1))
    policy = FactorFreshnessPolicy()

    latest = cache.lookup(
        request(factor_key, info=dt.date(2025, 12, 31)), policy, dt.date(2025, 3, 1)
    )
    historical = cache.lookup(
        request(factor_key, info=dt.date(2025, 1, 15)), policy, dt.date(2025, 3, 1)
    )
    assert latest.estimate.range.base == Decimal("2")
    assert historical.estimate.range.base == Decimal("1")


def test_immutable_estimate_is_reusable_beyond_ttl():
    factor_key = key()
    cache = SemanticFactorCache()
    cache.put(estimate(factor_key, immutable=True))
    policy = FactorFreshnessPolicy(
        rules=(
            FactorFreshnessRule(
                domain=FactorDomain.COMPANY,
                mode=FactorFreshnessMode.TTL,
                ttl=dt.timedelta(days=1),
            ),
        )
    )
    result = cache.lookup(request(factor_key), policy, dt.date(2025, 2, 1))
    assert result.state is CacheState.HIT


def test_default_freshness_bounds_nonimmutable_estimates_by_domain():
    policy = FactorFreshnessPolicy()
    for domain in FactorDomain:
        factor_key = key(domain=domain)
        rule = policy.rule_for(estimate(factor_key))
        assert rule.ttl is not None
        assert timedelta(0) < rule.ttl <= timedelta(days=180)
        result = policy.check(
            estimate(factor_key),
            request(factor_key),
            dt.date(2025, 1, 1) + rule.ttl,
        )
        assert not result.eligible
        assert result.reason is FreshnessReason.TTL_EXPIRED


@pytest.mark.parametrize("domain", tuple(FactorDomain))
def test_default_freshness_reuses_immutable_estimates(domain):
    factor_key = key(domain=domain)
    cached = estimate(factor_key, immutable=True)
    result = FactorFreshnessPolicy().check(
        cached,
        request(factor_key),
        dt.date(2030, 1, 1),
    )
    assert result.eligible


def test_ttl_expiry_is_evaluated_at_supplied_time():
    factor_key = key()
    cache = SemanticFactorCache()
    cache.put(estimate(factor_key, immutable=False))
    policy = FactorFreshnessPolicy(
        rules=(FactorFreshnessRule(mode="ttl", ttl=dt.timedelta(days=30)),)
    )
    result = cache.lookup(request(factor_key), policy, dt.date(2025, 2, 1))
    assert result.state is CacheState.STALE
    assert result.reason is FreshnessReason.TTL_EXPIRED


def test_versions_are_retained_and_put_is_idempotent():
    factor_key = key()
    cache = SemanticFactorCache()
    first = estimate(factor_key, version=1)
    assert cache.put(first)
    assert not cache.put(first)
    assert cache.put(estimate(factor_key, version=2, value="2"))
    assert len(cache.history(factor_key)) == 2
    cache.invalidate(factor_key, fingerprint=first.fingerprint)
    assert len(cache.history(factor_key)) == 2
    assert (
        cache.lookup(request(factor_key), evaluated_at=dt.date(2025, 2, 1)).state
        is CacheState.HIT
    )


def test_estimate_fingerprint_ignores_cache_timing():
    factor_key = key()
    first = estimate(
        factor_key,
        created_at=dt.date(2025, 1, 2),
        expires_at=dt.date(2025, 2, 1),
    )
    second = estimate(
        factor_key,
        created_at=dt.date(2030, 1, 2),
        expires_at=dt.date(2030, 2, 1),
    )
    assert first.fingerprint == second.fingerprint


def test_dependency_fingerprint_mismatch_is_stale():
    factor_key = key()
    dependency = key(metric="revenue")
    cached = estimate(
        factor_key,
        dependencies=(dependency,),
        dependency_fingerprints={dependency.digest: "old"},
    )
    cache = SemanticFactorCache()
    cache.put(cached)
    result = cache.lookup(
        request(factor_key),
        FactorFreshnessPolicy(),
        dt.date(2025, 2, 1),
        {dependency.digest: "new"},
    )
    assert result.state is CacheState.STALE
    assert result.reason is FreshnessReason.DEPENDENCY_FINGERPRINT_MISMATCH


def test_persistence_round_trip_and_corrupt_files_fail_safe(tmp_path):
    factor_key = key()
    first = SemanticFactorCache(tmp_path)
    first.put(estimate(factor_key))
    second = SemanticFactorCache(tmp_path)
    result = second.lookup(request(factor_key), evaluated_at=dt.date(2025, 2, 1))
    assert result.state is CacheState.HIT

    (tmp_path / f"{factor_key.digest}.json").write_text("not json", encoding="utf-8")
    assert (
        SemanticFactorCache(tmp_path)
        .lookup(request(factor_key), evaluated_at=dt.date(2025, 2, 1))
        .state
        is CacheState.MISS
    )


def test_contracts_forbid_unknown_fields():
    with pytest.raises(ValueError):
        key(requester="must not be identity")
