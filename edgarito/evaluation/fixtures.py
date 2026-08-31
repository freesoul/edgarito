"""Frozen, offline evaluation fixture manifests and loader."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.granularity import Granularity
from edgarito.schemas.normalization.financials import (
    FinancialConcept,
    FinancialObservation,
    NormalizedCompanyFinancials,
)
from edgarito.services.forecasting.reasoning.contracts import ForecastReasoningInput

from .contracts import (
    ActualOutcomeData,
    ForecastBacktestCase,
    InformationAvailabilityRecord,
)
from .leakage import (
    LeakageError,
    audit_case,
    canonical_information_content_hash,
    canonical_information_identity,
)


class FixtureLoader:
    def load(self, name: str, directory: str | Path | None = None) -> "EvaluationFixture":
        return load_fixture(name, directory=directory)


class FixtureFinancialRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fiscal_year: int
    fiscal_period: str = "FY"
    period_start: datetime.date | None = None
    period_end: datetime.date
    concept: str
    value: str
    unit: str
    provider: str = "fixture"
    taxonomy: str = "fixture"
    source_concept: str
    source_id: str | None = None
    filed: datetime.date | None = None
    scale: str = "1"
    currency: str | None = None
    provenance: str = "manually_curated_fixture"

    @field_validator("concept", "unit", "provider", "taxonomy", "source_concept", "provenance")
    @classmethod
    def require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Fixture financial row text cannot be blank")
        return value


class FixtureEvidenceMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str
    category: str
    source: str
    source_date: datetime.date
    as_of: datetime.date
    provenance: str
    forward: bool = False


class EvaluationFixtureManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fixture_id: str
    ticker: str
    company: str
    as_of: datetime.date
    fiscal_calendar: str
    historical_cutoff: datetime.date
    exact_fiscal_years: tuple[int, ...]
    expected_archetypes: tuple[str, ...] = ()
    completeness: Literal["partial", "complete"] = "partial"
    financial_rows: tuple[FixtureFinancialRow, ...]
    reasoning_input: dict[str, Any]
    evidence_metadata: tuple[FixtureEvidenceMetadata, ...]
    availability_records: tuple[InformationAvailabilityRecord, ...] = ()
    actual_outcomes: ActualOutcomeData

    @model_validator(mode="before")
    @classmethod
    def require_raw_scored_actual_provenance(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        actuals = value.get("actual_outcomes") or {}
        observations = actuals.get("observations", ()) if isinstance(actuals, dict) else ()
        required = {
            "source_id",
            "source_concept",
            "source_date",
            "fiscal_period",
            "period_end",
            "unit",
            "scale",
            "currency",
            "source",
            "value_kind",
            "reconstruction_provenance",
        }
        for observation in observations:
            if not isinstance(observation, dict) or observation.get("value") is None:
                continue
            missing = sorted(required - set(observation))
            if "reconstruction_provenance" not in observation:
                missing.append("reconstruction_provenance")
            if observation.get("value_kind") == "derived" and "reconstruction_provenance" not in observation:
                missing.append("reconstruction_provenance")
            if missing:
                raise ValueError(
                    "Scored fixture actuals require explicit fields: "
                    + ", ".join(sorted(set(missing)))
                )
        return value

    @field_validator("exact_fiscal_years", mode="before")
    @classmethod
    def normalize_years(cls, value: Any) -> tuple[int, ...]:
        years = tuple(int(item) for item in value)
        if tuple(sorted(years)) != years or len(years) != len(set(years)):
            raise ValueError("Fixture target years must be sorted and unique")
        return years

    @property
    def source_provenance(self) -> tuple[str, ...]:
        return tuple(item.provenance for item in self.evidence_metadata)

    @model_validator(mode="after")
    def require_scored_actual_provenance(self) -> "EvaluationFixtureManifest":
        for item in self.actual_outcomes.observations:
            if item.value is None:
                continue
            missing = [
                name
                for name, value in (
                    ("source_id", item.source_id),
                    ("source_concept", item.source_concept),
                    ("source_date", item.source_date),
                    ("provenance", item.reconstruction_provenance or item.provenance),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "Scored fixture actuals require source_id, source_concept, "
                    "source_date, and reconstruction provenance: "
                    + ", ".join(missing)
                )
        identities = {item.identity for item in self.availability_records}
        unlinked = [
            item.evidence_id
            for item in self.evidence_metadata
            if item.evidence_id not in identities
        ]
        if unlinked:
            raise ValueError(
                "Fixture evidence metadata must link to availability records: "
                + ", ".join(unlinked)
            )
        return self


class EvaluationFixture(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    manifest: EvaluationFixtureManifest
    case: ForecastBacktestCase
    actual_outcomes: ActualOutcomeData


def _fixture_directory() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "tests" / "fixtures" / "evaluation"


def _financials(manifest: EvaluationFixtureManifest) -> NormalizedCompanyFinancials:
    observations = []
    for row in manifest.financial_rows:
        concept = FinancialConcept(row.concept)
        observations.append(
            FinancialObservation(
                concept=concept,
                statement=concept.statement,
                value=row.value,
                unit=row.unit,
                granularity=Granularity.ANNUAL,
                fiscal_year=row.fiscal_year,
                fiscal_period=FiscalPeriod(row.fiscal_period),
                period_start=row.period_start,
                period_end=row.period_end,
                provider=row.provider,
                taxonomy=row.taxonomy,
                source_concept=row.source_concept,
                accession_number=row.source_id,
                filed=row.filed,
                derivation=(
                    f"{row.provenance}; source={row.provider}; scale={row.scale}"
                ),
            )
        )
    return NormalizedCompanyFinancials(
        provider="fixture",
        company_id=manifest.ticker,
        company_name=manifest.company,
        ticker=manifest.ticker,
        retrieved_at=datetime.datetime.combine(manifest.as_of, datetime.time(12), tzinfo=datetime.timezone.utc),
        observations=observations,
    )


def _financials_from_rows(
    *,
    ticker: str,
    company: str,
    as_of: datetime.date,
    rows: tuple[FixtureFinancialRow, ...],
) -> NormalizedCompanyFinancials:
    # Avoid validating this partial manifest; _financials only needs the
    # fields it reads and the preflight object never leaves this function.
    observations = []
    for row in rows:
        concept = FinancialConcept(row.concept)
        observations.append(
            FinancialObservation(
                concept=concept,
                statement=concept.statement,
                value=row.value,
                unit=row.unit,
                granularity=Granularity.ANNUAL,
                fiscal_year=row.fiscal_year,
                fiscal_period=FiscalPeriod(row.fiscal_period),
                period_start=row.period_start,
                period_end=row.period_end,
                provider=row.provider,
                taxonomy=row.taxonomy,
                source_concept=row.source_concept,
                accession_number=row.source_id,
                filed=row.filed,
                derivation=f"{row.provenance}; source={row.provider}; scale={row.scale}",
            )
        )
    return NormalizedCompanyFinancials(
        provider="fixture",
        company_id=ticker,
        company_name=company,
        ticker=ticker,
        retrieved_at=datetime.datetime.combine(
            as_of, datetime.time(12), tzinfo=datetime.timezone.utc
        ),
        observations=observations,
    )


def _availability_hash_candidates(payload: dict[str, Any]) -> dict[str, str]:
    rows = tuple(
        FixtureFinancialRow.model_validate(item)
        for item in payload.get("financial_rows", ())
    )
    financials = _financials_from_rows(
        ticker=payload["ticker"],
        company=payload["company"],
        as_of=datetime.date.fromisoformat(str(payload["as_of"])[:10]),
        rows=rows,
    )
    candidates: dict[str, str] = {}
    for item in financials.observations:
        candidates[canonical_information_identity(item, "normalized_fact")] = canonical_information_content_hash(item)
    reasoning = ForecastReasoningInput.model_validate(payload["reasoning_input"])
    for field_name, category in (
        ("segments", "operating"),
        ("definitions", "operating"),
        ("observations", "operating"),
        ("manual_forward_driver_observations", "forward"),
        ("management_guidance", "guidance"),
        ("management_constraints", "constraint"),
        ("investment_programs", "program"),
        ("historical_facts", "historical_summary"),
        ("research_evidence", "research"),
        ("evidence_consensus", "consensus"),
        ("manual_overrides", "manual"),
    ):
        for item in getattr(reasoning, field_name, ()):
            identity = canonical_information_identity(item, category)
            candidates[identity] = canonical_information_content_hash(item)
            if field_name == "evidence_consensus":
                for contributor in getattr(item, "contributors", ()):
                    identity = canonical_information_identity(contributor, "consensus")
                    candidates[identity] = canonical_information_content_hash(contributor)
    return candidates


def load_fixture(name: str, directory: str | Path | None = None) -> EvaluationFixture:
    """Load one committed JSON manifest without network access."""

    normalized = str(name).strip().casefold()
    if normalized.endswith(".json"):
        normalized = normalized[:-5]
    path = Path(directory) if directory is not None else _fixture_directory()
    manifest_path = path / f"{normalized}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Unknown evaluation fixture: {name}")
    raw_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = _availability_hash_candidates(raw_payload)
    for record in raw_payload.get("availability_records", ()):
        identity = record.get("identity", record.get("item_identity"))
        expected_hash = candidates.get(identity)
        if expected_hash is None:
            raise ValueError(f"Fixture availability record is not linked: {identity}")
        supplied_hash = record.get("content_hash", record.get("payload_hash"))
        if supplied_hash is not None and supplied_hash != expected_hash:
            raise ValueError(f"Stale fixture availability hash for {identity}")
        record["content_hash"] = expected_hash
        record.pop("payload_hash", None)
        record.pop("canonical_content_hash", None)
    manifest = EvaluationFixtureManifest.model_validate(raw_payload)
    if manifest.fixture_id.casefold() != normalized:
        raise ValueError("Fixture filename and fixture_id do not match")
    financials = _financials(manifest)
    if any(row.period_end > manifest.historical_cutoff for row in manifest.financial_rows):
        raise ValueError("Fixture financial rows exceed the historical cutoff")
    case = ForecastBacktestCase(
        ticker=manifest.ticker,
        company=manifest.company,
        as_of=manifest.as_of,
        fiscal_years=manifest.exact_fiscal_years,
        point_in_time_financials=financials,
        reasoning_input=manifest.reasoning_input,
        # Availability metadata is intentionally kept on the case manifest,
        # not forwarded as economic evidence to a collaborator.
        evidence_snapshot=None,
        availability_manifest=manifest.availability_records,
        expected_archetypes=manifest.expected_archetypes,
    )
    audit = audit_case(case)
    if not audit.valid:
        raise LeakageError(audit)
    return EvaluationFixture(manifest=manifest, case=case, actual_outcomes=manifest.actual_outcomes)


load_evaluation_fixture = load_fixture


__all__ = [
    "FixtureFinancialRow",
    "FixtureEvidenceMetadata",
    "EvaluationFixtureManifest",
    "EvaluationFixture",
    "FixtureLoader",
    "load_fixture",
    "load_evaluation_fixture",
]
