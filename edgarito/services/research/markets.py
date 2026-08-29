"""Typed collections and non-forecast helpers for market evidence."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgarito.services.research.contracts import (
    EvidenceItem,
    MarketGrowthEvidence,
    MarketShareEvidence,
    MarketSizeEvidence,
)


class MarketEvidenceCollection(BaseModel):
    """Immutable category buckets for market-size, growth, and share evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    market_sizes: tuple[MarketSizeEvidence, ...] = Field(default_factory=tuple)
    market_growth: tuple[MarketGrowthEvidence, ...] = Field(default_factory=tuple)
    market_shares: tuple[MarketShareEvidence, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def reject_duplicate_ids(self) -> "MarketEvidenceCollection":
        identifiers = [
            item.evidence_id
            for item in self.all_evidence
            if item.evidence_id is not None
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Market evidence IDs cannot repeat")
        return self

    @classmethod
    def from_items(cls, items: Iterable[EvidenceItem]) -> "MarketEvidenceCollection":
        """Bucket market evidence without selecting or projecting values."""

        sizes: list[MarketSizeEvidence] = []
        growth: list[MarketGrowthEvidence] = []
        shares: list[MarketShareEvidence] = []
        for item in items:
            if isinstance(item, MarketSizeEvidence):
                sizes.append(item)
            elif isinstance(item, MarketGrowthEvidence):
                growth.append(item)
            elif isinstance(item, MarketShareEvidence):
                shares.append(item)
            else:
                raise TypeError(f"Unsupported market evidence kind: {item.kind.value}")
        return cls(
            market_sizes=tuple(sizes),
            market_growth=tuple(growth),
            market_shares=tuple(shares),
        )

    @property
    def all_evidence(self) -> tuple[EvidenceItem, ...]:
        """Return all buckets in stable category order."""

        return (*self.market_sizes, *self.market_growth, *self.market_shares)

    @property
    def market_size(self) -> tuple[MarketSizeEvidence, ...]:
        return self.market_sizes

    @property
    def market_growths(self) -> tuple[MarketGrowthEvidence, ...]:
        return self.market_growth

    @property
    def market_share(self) -> tuple[MarketShareEvidence, ...]:
        return self.market_shares

    @property
    def market_names(self) -> tuple[str, ...]:
        """Return unique market labels without inferring a market outlook."""

        names = {
            item.market
            for item in self.all_evidence
            if hasattr(item, "market") and item.market is not None
        }
        return tuple(sorted(names, key=str.casefold))

    def for_market(self, market: str) -> tuple[EvidenceItem, ...]:
        """Return observations explicitly attached to ``market``."""

        normalized = market.strip().casefold()
        if not normalized:
            raise ValueError("Market lookup cannot be blank")
        return tuple(
            item
            for item in self.all_evidence
            if getattr(item, "market", None) is not None
            and item.market.casefold() == normalized
        )


MarketEvidenceSet = MarketEvidenceCollection


__all__ = ["MarketEvidenceCollection", "MarketEvidenceSet"]
