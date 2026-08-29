"""Standalone provider-neutral market-intelligence evidence contracts."""

from edgarito.services.research.competitors import CompetitorEvidenceCollection
from edgarito.services.research.consensus import (
    EvidenceConsensus,
    EvidenceContributor,
    EvidenceReconciler,
    reconcile_evidence,
    reconcile_market_growth,
    reconcile_market_share,
    reconcile_market_size,
    reconcile_pricing_observations,
    reconcile_production_capacity,
)
from edgarito.services.research.contracts import (
    CompetitorObservation,
    EstimateRange,
    EvidenceConfidence,
    EvidenceContext,
    EvidenceItem,
    EvidenceKind,
    EvidenceProvenance,
    EvidenceSourceType,
    MarketGrowthEvidence,
    MarketShareEvidence,
    MarketSizeEvidence,
    PricingObservation,
    ProductionCapacityEvidence,
)
from edgarito.services.research.markets import MarketEvidenceCollection

__all__ = [
    "CompetitorEvidenceCollection",
    "CompetitorObservation",
    "EvidenceConfidence",
    "EvidenceConsensus",
    "EvidenceContext",
    "EvidenceContributor",
    "EvidenceItem",
    "EvidenceKind",
    "EvidenceProvenance",
    "EvidenceReconciler",
    "EvidenceSourceType",
    "EstimateRange",
    "MarketEvidenceCollection",
    "MarketGrowthEvidence",
    "MarketShareEvidence",
    "MarketSizeEvidence",
    "PricingObservation",
    "ProductionCapacityEvidence",
    "reconcile_evidence",
    "reconcile_market_growth",
    "reconcile_market_share",
    "reconcile_market_size",
    "reconcile_pricing_observations",
    "reconcile_production_capacity",
]
