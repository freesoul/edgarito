import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from edgarito.services.research.competitors import CompetitorEvidenceCollection
from edgarito.services.research.consensus import (
    reconcile_evidence,
    reconcile_market_size,
)
from edgarito.services.research.contracts import (
    CompetitorObservation,
    EvidenceConfidence,
    EvidenceContext,
    EvidenceKind,
    EvidenceSourceType,
    MarketGrowthEvidence,
    MarketShareEvidence,
    MarketSizeEvidence,
    PricingObservation,
    ProductionCapacityEvidence,
)
from edgarito.services.research.markets import MarketEvidenceCollection

DATE = datetime.date(2026, 8, 29)


def _common(source_type: EvidenceSourceType, source: str, **values: object) -> dict:
    return {
        "source_date": DATE,
        "source_type": source_type,
        "low": Decimal("80"),
        "base": Decimal("100"),
        "high": Decimal("120"),
        "provenance": source,
        "confidence": EvidenceConfidence.HIGH,
        "unit": "USD million",
        **values,
    }


def test_contracts_cover_all_evidence_kinds_and_are_immutable():
    share_values = _common(
        EvidenceSourceType.ANALYST_ESTIMATE,
        "share",
        unit="percent",
        low=Decimal("20"),
        base=Decimal("30"),
        high=Decimal("40"),
    )
    values = (
        MarketSizeEvidence(
            **_common(EvidenceSourceType.ANALYST_ESTIMATE, "size"),
            market="EV charging",
            market_scope="tam",
        ),
        MarketGrowthEvidence(
            **_common(EvidenceSourceType.ANALYST_ESTIMATE, "growth", unit="percent"),
            market="EV charging",
        ),
        MarketShareEvidence(**share_values, company="Acme", market="EV charging"),
        CompetitorObservation(
            **_common(EvidenceSourceType.ANALYST_ESTIMATE, "competitor"),
            competitor="Peer",
            market="EV charging",
        ),
        ProductionCapacityEvidence(
            **_common(EvidenceSourceType.ANALYST_ESTIMATE, "capacity"),
            company="Acme",
            market="EV charging",
        ),
        PricingObservation(
            **_common(EvidenceSourceType.ANALYST_ESTIMATE, "pricing"),
            product="Charger",
            market="EV charging",
        ),
    )

    assert tuple(value.kind for value in values) == (
        EvidenceKind.MARKET_SIZE,
        EvidenceKind.MARKET_GROWTH,
        EvidenceKind.MARKET_SHARE,
        EvidenceKind.COMPETITOR,
        EvidenceKind.PRODUCTION_CAPACITY,
        EvidenceKind.PRICING,
    )
    assert values[0].estimate.base == Decimal("100")
    assert values[0].context.market == "EV charging"
    with pytest.raises(ValidationError):
        values[0].market = "other"


def test_range_validation_rejects_malformed_and_nonfinite_ranges():
    malformed = _common(EvidenceSourceType.ANALYST_ESTIMATE, "bad")
    malformed["low"] = Decimal("101")
    with pytest.raises(ValidationError, match="low <= base <= high"):
        MarketSizeEvidence(
            **malformed,
            market="EV charging",
        )
    nonfinite = _common(EvidenceSourceType.ANALYST_ESTIMATE, "bad")
    nonfinite["high"] = Decimal("NaN")
    with pytest.raises(ValidationError, match="finite"):
        MarketSizeEvidence(
            **nonfinite,
            market="EV charging",
        )
    negative = _common(EvidenceSourceType.ANALYST_ESTIMATE, "bad")
    negative.update(low=Decimal("-1"), base=Decimal("0"), high=Decimal("1"))
    with pytest.raises(ValidationError, match="cannot be negative"):
        MarketSizeEvidence(
            **negative,
            market="EV charging",
        )


def test_context_and_collections_are_typed_without_forecast_behavior():
    size = MarketSizeEvidence(
        **_common(EvidenceSourceType.ANALYST_ESTIMATE, "size"), market="EV charging"
    )
    growth = MarketGrowthEvidence(
        **_common(EvidenceSourceType.ANALYST_ESTIMATE, "growth", unit="percent"),
        market="EV charging",
    )
    competitor = CompetitorObservation(
        **_common(EvidenceSourceType.ANALYST_ESTIMATE, "peer"),
        competitor="Peer",
        market="EV charging",
    )
    markets = MarketEvidenceCollection.from_items((size, growth))
    competitors = CompetitorEvidenceCollection.from_items((competitor,))
    assert markets.market_names == ("EV charging",)
    assert markets.for_market("ev charging") == (size, growth)
    assert competitors.competitor_names == ("Peer",)
    assert competitors.for_competitor("peer") == (competitor,)
    assert EvidenceContext(market="EV charging").market == "EV charging"


def test_precedence_uses_highest_tier_and_preserves_lower_tier_sources():
    first_party = MarketSizeEvidence(
        **_common(
            EvidenceSourceType.REPORTED_FIRST_PARTY_FACT,
            "annual report",
            low=Decimal("90"),
            base=Decimal("100"),
            high=Decimal("110"),
        ),
        market="EV charging",
    )
    analyst = MarketSizeEvidence(
        **_common(
            EvidenceSourceType.ANALYST_ESTIMATE,
            "broker note",
            low=Decimal("900"),
            base=Decimal("1000"),
            high=Decimal("1100"),
        ),
        market="EV charging",
    )
    result = reconcile_market_size((analyst, first_party))
    assert (result.low, result.base, result.high) == (
        Decimal("90"),
        Decimal("100"),
        Decimal("110"),
    )
    assert result.governing_source_type == EvidenceSourceType.REPORTED_FIRST_PARTY_FACT
    assert result.number_sources == 2
    assert result.governing_number_sources == 1
    assert result.sources == ("annual report", "broker note")
    assert result.lower_priority_contributors[0].source == "broker note"


def test_same_tier_conflicts_use_coordinate_medians_and_are_deterministic():
    left = MarketSizeEvidence(
        **_common(
            EvidenceSourceType.INDEPENDENT_INDUSTRY_ESTIMATE,
            "industry A",
            low=Decimal("80"),
            base=Decimal("100"),
            high=Decimal("120"),
        ),
        market="EV charging",
    )
    right = MarketSizeEvidence(
        **_common(
            EvidenceSourceType.INDEPENDENT_INDUSTRY_ESTIMATE,
            "industry B",
            low=Decimal("100"),
            base=Decimal("140"),
            high=Decimal("180"),
        ),
        market="EV charging",
    )
    first = reconcile_evidence((right, left))
    second = reconcile_evidence((left, right))
    assert first == second
    assert (first.low, first.base, first.high) == (
        Decimal("90"),
        Decimal("120"),
        Decimal("150"),
    )
    assert first.dispersion == Decimal("40")
    assert tuple(item.source for item in first.contributors) == (
        "industry A",
        "industry B",
    )
    assert tuple(item.governing for item in first.contributors) == (True, True)


def test_empty_input_and_noncomparable_inputs_are_rejected():
    with pytest.raises(ValueError, match="empty evidence"):
        reconcile_evidence(())
    one = MarketSizeEvidence(
        **_common(EvidenceSourceType.ANALYST_ESTIMATE, "one"), market="EV charging"
    )
    with pytest.raises(ValueError, match="different units"):
        reconcile_evidence(
            (
                one,
                one.model_copy(update={"unit": "percent"}),
            )
        )
