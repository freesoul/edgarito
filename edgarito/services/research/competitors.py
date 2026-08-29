"""Typed collections and non-forecast helpers for competitor evidence."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from edgarito.services.research.contracts import (
    CompetitorObservation,
    EvidenceItem,
    PricingObservation,
    ProductionCapacityEvidence,
)


class CompetitorEvidenceCollection(BaseModel):
    """Immutable buckets for competitor, capacity, and pricing observations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    competitor_observations: tuple[CompetitorObservation, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("competitor_observations", "observations"),
    )
    production_capacity_constraints: tuple[ProductionCapacityEvidence, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices(
            "production_capacity_constraints", "capacity_constraints", "capacity"
        ),
    )
    pricing_observations: tuple[PricingObservation, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("pricing_observations", "pricing"),
    )

    @model_validator(mode="after")
    def reject_duplicate_ids(self) -> "CompetitorEvidenceCollection":
        identifiers = [
            item.evidence_id
            for item in self.all_evidence
            if item.evidence_id is not None
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Competitor evidence IDs cannot repeat")
        return self

    @classmethod
    def from_items(
        cls, items: Iterable[EvidenceItem]
    ) -> "CompetitorEvidenceCollection":
        """Bucket competitor evidence without choosing or forecasting anything."""

        observations: list[CompetitorObservation] = []
        capacities: list[ProductionCapacityEvidence] = []
        pricing: list[PricingObservation] = []
        for item in items:
            if isinstance(item, CompetitorObservation):
                observations.append(item)
            elif isinstance(item, ProductionCapacityEvidence):
                capacities.append(item)
            elif isinstance(item, PricingObservation):
                pricing.append(item)
            else:
                raise TypeError(
                    f"Unsupported competitor evidence kind: {item.kind.value}"
                )
        return cls(
            competitor_observations=tuple(observations),
            production_capacity_constraints=tuple(capacities),
            pricing_observations=tuple(pricing),
        )

    @property
    def all_evidence(self) -> tuple[EvidenceItem, ...]:
        """Return all buckets in stable category order."""

        return (
            *self.competitor_observations,
            *self.production_capacity_constraints,
            *self.pricing_observations,
        )

    @property
    def observations(self) -> tuple[CompetitorObservation, ...]:
        return self.competitor_observations

    @property
    def capacity_constraints(self) -> tuple[ProductionCapacityEvidence, ...]:
        return self.production_capacity_constraints

    @property
    def pricing(self) -> tuple[PricingObservation, ...]:
        return self.pricing_observations

    @property
    def competitor_names(self) -> tuple[str, ...]:
        """Return unique competitor labels present in observations."""

        names = {
            item.competitor
            for item in self.competitor_observations
            if item.competitor is not None
        }
        return tuple(sorted(names, key=str.casefold))

    def for_competitor(self, competitor: str) -> tuple[EvidenceItem, ...]:
        """Return observations explicitly attached to ``competitor``."""

        normalized = competitor.strip().casefold()
        if not normalized:
            raise ValueError("Competitor lookup cannot be blank")
        return tuple(
            item
            for item in self.all_evidence
            if getattr(item, "competitor", None) is not None
            and item.competitor.casefold() == normalized
        )


CompetitorEvidenceSet = CompetitorEvidenceCollection


__all__ = ["CompetitorEvidenceCollection", "CompetitorEvidenceSet"]
