"""Fail-closed point-in-time leakage auditing for evaluation inputs."""

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from edgarito.schemas.forecasting import ForecastOverride
from edgarito.schemas.guidance.management import (
    ExtractedGuidanceItem,
    ExtractedGuidanceResponse,
    GuidanceApplication,
    GuidanceDocumentAudit,
    GuidanceExtractionCacheEntry,
    GuidanceOverlayResult,
    GuidanceRejection,
    ManagementGuidance,
    MonetaryForecastConstraint,
)
from edgarito.schemas.normalization.financials import NormalizedCompanyFinancials
from edgarito.schemas.operating import (
    EvidenceReference,
    ExtractedOperatingDriverDefinition,
    ExtractedOperatingEvidenceResponse,
    ExtractedOperatingInvestmentProgram,
    ExtractedOperatingObservation,
    ExtractedOperatingSegment,
    OperatingDocumentAudit,
    OperatingDriverDefinition,
    OperatingDriverForecast,
    OperatingDriverObservation,
    OperatingEvidenceAuditRecord,
    OperatingEvidenceExtractionResult,
    OperatingEvidenceRejection,
    OperatingExtractionCacheEntry,
    OperatingInvestmentProgram,
    OperatingSegment,
)
from edgarito.services.financials.availability import ObservationAvailabilityMode
from edgarito.services.forecasting.reasoning.contracts import HistoricalFactSummary
from edgarito.services.forecasting.reasoning.evidence import content_hash
from edgarito.services.research.consensus import EvidenceConsensus, EvidenceContributor
from edgarito.services.research.contracts import (
    EstimateRange,
    EvidenceContext,
    EvidenceItem,
    EvidenceProvenance,
    ResearchEvidence,
)

from .contracts import (
    ActualOutcomeData,
    EvidenceSnapshot,
    ForecastBacktestCase,
    ImmutableNormalizedCompanyFinancials,
    InformationAvailabilityRecord,
    LeakageAudit,
    LeakageEvidenceRecord,
    LeakageStatus,
    _contains_actual,
    _stable_value,
)


class LeakageError(ValueError):
    """Raised before a reasoner, cache, or baseline can see a leaking case."""

    def __init__(self, audit: LeakageAudit):
        self.audit = audit
        details = "; ".join(audit.issues) or "point-in-time leakage audit failed"
        super().__init__(details)


class LeakageAuditor:
    """Reusable fail-closed auditor with a fixed availability policy."""

    def __init__(
        self,
        availability_mode: ObservationAvailabilityMode = ObservationAvailabilityMode.POINT_IN_TIME,
    ) -> None:
        self.availability_mode = ObservationAvailabilityMode(availability_mode)

    def audit(self, case: ForecastBacktestCase | Any) -> LeakageAudit:
        return audit_case(case, availability_mode=self.availability_mode)

    def validate(self, case: ForecastBacktestCase | Any) -> LeakageAudit:
        audit = self.audit(case)
        if not audit.valid:
            raise LeakageError(audit)
        return audit


class ActualOutcomeAudit(BaseModel):
    """Separate audit trail for outcome dates; it is never information evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: datetime.date
    dates: tuple[tuple[str, datetime.date], ...] = ()
    subsequent_dates: tuple[tuple[str, datetime.date], ...] = ()
    pre_cutoff_dates: tuple[tuple[str, datetime.date], ...] = ()
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Seen:
    evidence_id: str
    category: str
    date: datetime.date | None
    dates: tuple[datetime.date, ...]
    required_date: bool = False
    available: bool = True
    reason: str | None = None
    payload: Any = None
    manifest_error: str | None = None


_DATE_KEYS = {
    "source_date",
    "filing_date",
    "filed",
    "observed_on",
    "published_on",
    "as_of",
    "retrieved_at",
    "date",
}


def _as_date(value: Any) -> datetime.date | None:
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        try:
            return datetime.date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _iter_dates(value: Any, *, include_period_end: bool = False) -> tuple[datetime.date, ...]:
    """Find publication/availability dates in an arbitrary typed evidence tree."""

    found: list[datetime.date] = []
    seen: set[int] = set()

    def visit(item: Any) -> None:
        if item is None or id(item) in seen or isinstance(item, (str, bytes, int, float, bool, Decimal)):
            return
        seen.add(id(item))
        if isinstance(item, datetime.datetime):
            found.append(item.date())
            return
        if isinstance(item, datetime.date):
            found.append(item)
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                key_text = str(key).casefold()
                if key_text in _DATE_KEYS or (include_period_end and key_text == "period_end"):
                    date = _as_date(nested)
                    if date is not None:
                        found.append(date)
                elif isinstance(nested, (Mapping, Sequence)) or hasattr(nested, "__dict__"):
                    visit(nested)
            return
        if isinstance(item, Sequence | set | frozenset):
            for nested in item:
                visit(nested)
            return
        # Pydantic models and small project result objects are intentionally
        # traversed by attributes rather than by model_dump: this preserves
        # nested EvidenceReference and contributor boundaries.
        for name in _DATE_KEYS | ({"period_end"} if include_period_end else set()):
            nested = getattr(item, name, None)
            date = _as_date(nested)
            if date is not None:
                found.append(date)
        values = getattr(item, "__dict__", None)
        if isinstance(values, dict):
            for name, nested in values.items():
                if name.casefold() not in _DATE_KEYS and not (
                    include_period_end and name.casefold() == "period_end"
                ):
                    visit(nested)

    visit(value)
    return tuple(sorted(set(found)))


_REQUIRED_CONTEXT = {
    "operating",
    "forward",
    "forward_evidence",
    "research",
    "research_evidence",
    "guidance",
    "management_guidance",
    "constraint",
    "constraints",
    "management_constraints",
    "program",
    "programs",
    "investment_program",
    "investment_programs",
    "consensus",
    "consensus_contributor",
}
_STRUCTURAL_CONTEXT = {
    "segments": "operating",
    "definitions": "operating",
    "observations": "operating",
    "operating_evidence": "operating",
    "operating_segments": "operating",
    "driver_definitions": "operating",
    "operating_definitions": "operating",
    "driver_observations": "operating",
    "operating_observations": "operating",
    "historical_facts": "historical_summary",
    "research_evidence": "research",
    "research": "research",
    "management_guidance": "guidance",
    "guidance": "guidance",
    "evidence_consensus": "consensus",
    "consensus": "consensus",
}
_PAYLOAD_CONTAINER_KEYS = {
    "evidence_snapshot",
    "operating_evidence",
    "operating",
    "hybrid",
    "segments",
    "operating_segments",
    "definitions",
    "driver_definitions",
    "operating_definitions",
    "observations",
    "driver_observations",
    "operating_observations",
    "manual_forward_driver_observations",
    "forward_evidence",
    "management_guidance",
    "guidance",
    "management_constraints",
    "investment_programs",
    "investment_program_facts",
    "historical_facts",
    "normalized_historical_facts",
    "research_evidence",
    "research",
    "evidence_consensus",
    "consensus",
    "manual_overrides",
    "overrides",
    "evidence",
    "context",
    "provenance",
    "estimate",
    "contributors",
    "source_provenance",
    "audit_records",
    "rejected",
    "applications",
    "accepted",
    "metadata",
    "item",
    "dimensions",
    "units",
    "values",
    "keyword_hits",
    "document_audits",
}
_PAYLOAD_RECORD_KEYS = {
    "kind",
    "source_type",
    "source_category",
    "segment_id",
    "name",
    "parent_id",
    "scope",
    "currency",
    "dimensions",
    "driver_id",
    "archetype",
    "output_metric",
    "input_metrics",
    "formula_id",
    "required_inputs",
    "optional_inputs",
    "fiscal_year",
    "fiscal_period",
    "period",
    "period_key",
    "metric",
    "value",
    "unit",
    "units",
    "original_unit",
    "original_scale",
    "scale",
    "basis",
    "scope_evidence",
    "is_total",
    "is_component",
    "exhaustive",
    "origin",
    "confidence",
    "low",
    "base",
    "high",
    "point",
    "minimum",
    "maximum",
    "scope_id",
    "source",
    "source_id",
    "source_concept",
    "source_date",
    "filing_date",
    "filed",
    "observed_on",
    "published_on",
    "as_of",
    "retrieved_at",
    "period_end",
    "period_start",
    "date",
    "accession",
    "accession_number",
    "provider",
    "taxonomy",
    "document_name",
    "source_text_hash",
    "supporting_text",
    "evidence_verified",
    "extraction_model",
    "filing_form",
    "source_document",
    "source_document_type",
    "filename",
    "document_type",
    "record_type",
    "program_id",
    "status",
    "purpose",
    "method",
    "market",
    "market_name",
    "market_id",
    "geography",
    "segment",
    "company",
    "company_name",
    "competitor",
    "competitor_name",
    "product",
    "product_name",
    "facility",
    "market_scope",
    "share_basis",
    "growth_basis",
    "constraint_type",
    "price_type",
    "observation_type",
    "observation",
    "constraint",
    "notes",
    "publisher",
    "reference",
    "locator",
    "url",
    "governing",
    "number_sources",
    "dispersion",
    "sources",
    "governing_source_type",
    "governing_number_sources",
    "extraction_confidence",
    "fiscal_quarter",
    "period_type",
    "metric_name",
    "value_kind",
    "qualifier",
    "segment_name",
    "is_primary",
    "cleaned_size",
    "bounded_context_size",
    "keyword_hits",
    "accepted_records",
    "rejected_records",
    "accepted_segments",
    "accepted_definitions",
    "accepted_observations",
    "accepted_investment_programs",
    "unsupported_evidence",
    "missing_evidence",
    "unusable_evidence",
    "warnings",
    "reason",
    "methodology",
    "applications",
    "evidence_only",
    "rejected_reasons",
    "cache_hits",
    "cache_misses",
    "filings_inspected",
    "documents_inspected",
    "extracted_guidance_records",
    "extracted_at",
    "model",
    "reasoning_effort",
    "prompt_version",
    "schema_version",
    "content_hash",
    "document_filename",
    "item",
    "metadata",
    "evidence_id",
    "provenance",
}

_TEXT_MAP_CONTEXTS = frozenset(
    {"dimensions", "units", "metadata", "keyword_hits", "values", "text_map"}
)
_STRICT_MAPPING_CONTEXTS = frozenset(
    {
        "segments",
        "operating_segments",
        "definitions",
        "driver_definitions",
        "operating_definitions",
        "observations",
        "driver_observations",
        "operating_observations",
        "investment_programs",
        "investment_program_facts",
        "management_guidance",
        "guidance",
        "research_evidence",
        "research",
        "evidence_consensus",
        "consensus",
        "manual_overrides",
        "overrides",
        "evidence",
        "context",
        "provenance",
        "estimate",
        "contributors",
        "audit_records",
        "rejected",
        "applications",
        "accepted",
        "evidence_only",
        "document_audits",
        "source_provenance",
        "item",
    }
)
_WRAPPER_MAPPING_CONTEXTS = frozenset(
    {"evidence_snapshot", "operating_evidence", "operating", "hybrid"}
)

_KNOWN_TYPED_PAYLOAD_TYPES = (
    EvidenceContext,
    EvidenceProvenance,
    EstimateRange,
    ResearchEvidence,
    EvidenceContributor,
    EvidenceConsensus,
    OperatingSegment,
    EvidenceReference,
    OperatingDriverDefinition,
    OperatingDriverObservation,
    OperatingInvestmentProgram,
    OperatingDriverForecast,
    OperatingEvidenceAuditRecord,
    OperatingEvidenceExtractionResult,
    OperatingEvidenceRejection,
    OperatingExtractionCacheEntry,
    OperatingDocumentAudit,
    ExtractedOperatingSegment,
    ExtractedOperatingDriverDefinition,
    ExtractedOperatingObservation,
    ExtractedOperatingInvestmentProgram,
    ExtractedOperatingEvidenceResponse,
    EvidenceReference,
    ManagementGuidance,
    MonetaryForecastConstraint,
    GuidanceApplication,
    GuidanceDocumentAudit,
    GuidanceOverlayResult,
    GuidanceRejection,
    GuidanceExtractionCacheEntry,
    ExtractedGuidanceItem,
    ExtractedGuidanceResponse,
    ForecastOverride,
    HistoricalFactSummary,
)

_KNOWN_MAPPING_TYPES = (
    *_KNOWN_TYPED_PAYLOAD_TYPES,
    EvidenceSnapshot,
)
_RESEARCH_ITEM_ADAPTER = TypeAdapter(EvidenceItem)


def _model_mapping_keys(model: type[BaseModel]) -> frozenset[str]:
    keys: set[str] = set(model.model_fields)
    for field in model.model_fields.values():
        alias = field.validation_alias
        choices = getattr(alias, "choices", (alias,) if alias is not None else ())
        keys.update(str(item) for item in choices)
    return frozenset(item.casefold() for item in keys)


def _model_accepts_mapping(model: type[BaseModel], value: Mapping[str, Any]) -> bool:
    keys = {str(key).casefold() for key in value}
    config = model.model_config
    if config.get("extra") != "forbid" and not keys <= _model_mapping_keys(model):
        return False
    try:
        model.model_validate(value)
    except (TypeError, ValueError, ValidationError):
        return False
    return True


def _is_known_mapping_contract(value: Any) -> bool:
    """Recognize only mappings that validate as a supported evidence contract."""

    if not isinstance(value, Mapping):
        return False
    for model in _KNOWN_MAPPING_TYPES:
        if _model_accepts_mapping(model, value):
            return True
    try:
        _RESEARCH_ITEM_ADAPTER.validate_python(value)
    except (TypeError, ValueError, ValidationError):
        return False
    return True


def _is_dated_generic_record(value: Mapping[str, Any]) -> bool:
    """Keep support for content-free, explicitly linked evidence records."""

    keys = {str(key).casefold() for key in value}
    return bool(
        keys & {"evidence_id", "source_id"}
        and keys & (_DATE_KEYS | {"period_end"})
        and keys <= _PAYLOAD_CONTAINER_KEYS | _PAYLOAD_RECORD_KEYS
    )


def _is_container_mapping(value: Mapping[str, Any]) -> bool:
    keys = {str(key).casefold() for key in value}
    return bool(keys & _PAYLOAD_CONTAINER_KEYS)


def _payload_mapping_category(
    value: Mapping[str, Any], context: str | None
) -> str | None:
    """Classify one mapping record without mistaking nested metadata for one."""

    keys = {str(key).casefold() for key in value}
    if keys & {
        "segments",
        "definitions",
        "observations",
        "research_evidence",
        "management_guidance",
        "evidence_consensus",
        "audit_records",
        "rejected",
    }:
        return None
    if "kind" in keys and keys & _DATE_KEYS:
        return "research"
    if "segment_id" in keys and (
        "name" in keys or "driver_id" in keys or "program_id" in keys
    ):
        return "operating"
    if "program_id" in keys:
        return "program"
    if "metric" in keys and "period_type" in keys and "filing_date" in keys:
        return "guidance"
    if "evidence_id" in keys and keys & (_DATE_KEYS | {"period_end"}):
        return {
            "forward": "forward",
            "forward_evidence": "forward",
            "research_evidence": "research",
            "research": "research",
            "guidance": "guidance",
            "management_guidance": "guidance",
            "consensus": "consensus",
            "consensus_contributor": "consensus_contributor",
        }.get(context, context if context in _REQUIRED_CONTEXT else "operating")
    if context in _REQUIRED_CONTEXT and keys & {
        "value",
        "metric",
        "driver_id",
        "low",
        "program_id",
        "segment_id",
        "name",
        "source_date",
        "filing_date",
        "filed",
        "published_on",
        "observed_on",
        "date",
        "as_of",
    }:
        return context
    if context == "historical_summary" and keys & {"value", "metric"}:
        return context
    return None


def _unrecognized_payload_paths(payload: Any, label: str) -> tuple[str, ...]:
    """Reject arbitrary mapping leaves that could bypass evidence auditing."""

    unknown: list[str] = []
    seen: set[int] = set()

    def visit(item: Any, path: str, context: str | None = None) -> None:
        if item is None or id(item) in seen:
            return
        if isinstance(item, (str, bytes, int, float, bool, Decimal, datetime.date)):
            if context not in _PAYLOAD_RECORD_KEYS and context not in _TEXT_MAP_CONTEXTS:
                unknown.append(path)
            return
        seen.add(id(item))
        if isinstance(item, Mapping):
            keys = {str(key).casefold() for key in item}
            if not keys:
                return
            if context in _TEXT_MAP_CONTEXTS:
                for key, nested in item.items():
                    visit(nested, f"{path}.{key}", "text_map")
                return
            known_contract = _is_known_mapping_contract(item)
            if (
                context in _STRICT_MAPPING_CONTEXTS
                and not known_contract
                and not _is_dated_generic_record(item)
            ) or (
                context in _WRAPPER_MAPPING_CONTEXTS
                and not known_contract
                and not _is_dated_generic_record(item)
                and not _is_container_mapping(item)
            ) or (
                context is None
                and not known_contract
                and not _is_dated_generic_record(item)
                and not _is_container_mapping(item)
            ):
                unknown.append(path)
                return
            for key, nested in item.items():
                key_text = str(key).casefold()
                if key_text in _PAYLOAD_CONTAINER_KEYS or key_text in _PAYLOAD_RECORD_KEYS:
                    visit(nested, f"{path}.{key_text}", key_text)
                else:
                    unknown.append(f"{path}.{key_text}")
            return
        if isinstance(item, Sequence | set | frozenset):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]", context)
            return
        # Strict Pydantic evidence contracts have already rejected unknown
        # fields.  Traverse their public attributes only when the object is a
        # known evidence node; arbitrary collaborator payload objects are not
        # silently accepted as evidence containers.
        if not isinstance(item, _KNOWN_MAPPING_TYPES):
            unknown.append(path)
            return
        values = getattr(item, "__dict__", None)
        if isinstance(values, dict):
            for key, nested in values.items():
                key_text = str(key).casefold()
                if key_text not in _DATE_KEYS:
                    visit(nested, f"{path}.{key_text}", key_text)

    visit(payload, label)
    return tuple(dict.fromkeys(unknown))


def _required_payload_entries(payload: Any, label: str) -> tuple[_Seen, ...]:
    """Find forward-like records hidden inside operating result objects."""

    entries: list[_Seen] = []
    seen: set[int] = set()
    index = 0

    def visit(item: Any, context: str | None = None) -> None:
        nonlocal index
        if item is None or id(item) in seen or isinstance(item, (str, bytes, int, float, bool, Decimal, datetime.date)):
            return
        seen.add(id(item))
        if isinstance(item, Mapping):
            keys = {str(key).casefold() for key in item}
            category = _payload_mapping_category(item, context)
            if category is not None and keys & _PAYLOAD_RECORD_KEYS:
                dates = _iter_dates(item)
                entries.append(
                    _Seen(
                        _identity(item, category, index),
                        category,
                        max(dates) if dates else None,
                        dates,
                        required_date=True,
                        payload=item,
                    )
                )
                index += 1
            for key, nested in item.items():
                key_text = str(key).casefold()
                nested_context = (
                    key_text
                    if key_text in _REQUIRED_CONTEXT
                    or key_text in {
                        "context",
                        "provenance",
                        "dimensions",
                        "units",
                        "metadata",
                        "keyword_hits",
                        "values",
                    }
                    else _STRUCTURAL_CONTEXT.get(key_text, context)
                )
                visit(nested, nested_context)
            return
        if isinstance(item, Sequence | set | frozenset):
            for nested in item:
                visit(nested, context)
            return
        class_name = type(item).__name__.casefold()
        origin = str(getattr(getattr(item, "origin", None), "value", getattr(item, "origin", ""))).casefold()
        category = context
        if isinstance(item, ResearchEvidence):
            category = "research"
        elif isinstance(item, EvidenceContributor):
            category = "consensus_contributor"
        elif isinstance(item, EvidenceConsensus):
            category = "consensus"
        elif isinstance(
            item,
            (
                EvidenceReference,
                OperatingSegment,
                OperatingDriverDefinition,
                OperatingDriverObservation,
                OperatingInvestmentProgram,
                OperatingDriverForecast,
                OperatingEvidenceAuditRecord,
                OperatingEvidenceExtractionResult,
                OperatingEvidenceRejection,
                OperatingExtractionCacheEntry,
                OperatingDocumentAudit,
                ExtractedOperatingSegment,
                ExtractedOperatingDriverDefinition,
                ExtractedOperatingObservation,
                ExtractedOperatingInvestmentProgram,
                ExtractedOperatingEvidenceResponse,
            ),
        ):
            category = "operating"
        elif isinstance(item, HistoricalFactSummary):
            category = "historical_summary"
        elif isinstance(
            item,
            (
                ManagementGuidance,
                GuidanceApplication,
                GuidanceOverlayResult,
                GuidanceRejection,
                GuidanceExtractionCacheEntry,
                ExtractedGuidanceItem,
                ExtractedGuidanceResponse,
            ),
        ):
            category = "guidance"
        elif isinstance(item, MonetaryForecastConstraint):
            category = "constraint"
        elif origin in _REQUIRED_CONTEXT:
            category = origin
        elif "research" in class_name:
            category = "research"
        elif "guidance" in class_name:
            category = "guidance"
        elif "investmentprogram" in class_name:
            category = "program"
        elif "constraint" in class_name:
            category = "constraint"
        elif "consensus" in class_name:
            category = "consensus"
        elif "contributor" in class_name:
            category = "consensus_contributor"
        elif "operatingsegment" in class_name or "operatingdriverdefinition" in class_name:
            category = "operating"
        elif "operatingdriverobservation" in class_name:
            category = "operating"
        elif "evidencereference" in class_name or "operatingdocumentaudit" in class_name:
            category = "operating"
        elif "historicalfactsummary" in class_name:
            category = "historical_summary"
        if category in _REQUIRED_CONTEXT or category == "operating" or category == "historical_summary":
            dates = _iter_dates(item)
            entries.append(
                _Seen(
                    _identity(item, category, index),
                    category,
                    max(dates) if dates else None,
                    dates,
                    required_date=True,
                    payload=item,
                )
            )
            index += 1
        values = getattr(item, "__dict__", None)
        if isinstance(values, dict):
            for name, nested in values.items():
                if name.casefold() not in _DATE_KEYS:
                    visit(nested, context)

    visit(payload)
    return tuple(entries)


def canonical_information_identity(value: Any, category: str) -> str:
    """Return a stable identity used by availability-manifest links."""

    category = category.casefold().replace("-", "_")
    category = {
        "forward": "observation",
        "forward_evidence": "observation",
        "historical_summary": "historical",
        "research_evidence": "research",
        "consensus_contributor": "consensus",
    }.get(category, category)
    if category in {"normalized_fact", "fact"}:
        return ":".join(
            (
                "normalized",
                str(getattr(_field(value, "concept"), "value", _field(value, "concept", ""))),
                str(_field(value, "fiscal_year", "")),
                str(getattr(_field(value, "fiscal_period"), "value", _field(value, "fiscal_period", ""))),
                str(_field(value, "period_end", "")),
                str(_field(value, "source_concept", "")),
                str(_field(value, "provider", "")),
            )
        )
    if category in {"operating", "segment"} and _field(value, "segment_id") is not None and _field(value, "driver_id") is None:
        return f"segment:{_field(value, 'segment_id')}"
    if category in {"operating", "definition"} and _field(value, "driver_id") is not None and _field(value, "fiscal_year") is None:
        return f"definition:{_field(value, 'segment_id')}:{_field(value, 'driver_id')}"
    if category in {"operating", "observation", "forward"} and _field(value, "driver_id") is not None:
        period = getattr(_field(value, "fiscal_period"), "value", _field(value, "fiscal_period", "FY"))
        return f"observation:{_field(value, 'segment_id')}:{_field(value, 'driver_id')}:{_field(value, 'fiscal_year')}:{period}"
    if category in {"historical", "historical_summary"}:
        scope = getattr(_field(value, "scope"), "value", _field(value, "scope", "company"))
        return f"historical:{scope}:{_field(value, 'scope_id', 'company')}:{_field(value, 'metric')}:{_field(value, 'fiscal_year')}:{_field(value, 'fiscal_period', 'FY')}"
    if category in {"program", "investment_program"}:
        return f"program:{_field(value, 'program_id', '')}"
    if category in {"manual", "override"}:
        return f"manual:{_field(value, 'scope', 'company')}:{_field(value, 'scope_id', 'company')}:{_field(value, 'metric', _field(value, 'driver_id', ''))}"
    explicit = _field(value, "evidence_id") or _field(value, "accession_number") or _field(value, "accession")
    if explicit:
        return f"{category}:{explicit}"
    return f"{category}:{content_hash(_stable_value(value))[:24]}"


def canonical_information_content_hash(value: Any) -> str:
    """Hash the complete canonical consumed payload, not only its lookup key."""

    return content_hash(_stable_value(value))


def _identity(value: Any, category: str, index: int) -> str:
    return canonical_information_identity(value, category)


def _linked_entry(
    seen: _Seen,
    manifest: Mapping[str, InformationAvailabilityRecord],
    value: Any = None,
) -> _Seen:
    record = manifest.get(seen.evidence_id)
    if record is None:
        return seen
    value = seen.payload if value is None else value
    errors: list[str] = []
    if value is not None and record.content_hash != canonical_information_content_hash(value):
        errors.append("content hash does not match consumed payload")
    expected_category = seen.category.casefold().replace("-", "_")
    actual_category = record.category.casefold().replace("-", "_")
    category_families = {
        "normalized_fact": {"normalized_fact", "fact"},
        "historical_summary": {"historical_summary", "historical"},
        "consensus_contributor": {"consensus_contributor", "consensus"},
        "forward": {"forward", "forward_evidence", "observation", "operating"},
        "operating": {"operating", "segment", "definition", "observation"},
    }
    if actual_category not in category_families.get(expected_category, {expected_category}):
        errors.append("category metadata does not match consumed payload")
    intrinsic_source_id = _field(value, "source_id") or _field(value, "accession_number")
    if intrinsic_source_id is not None and str(intrinsic_source_id) != record.source_id:
        errors.append("source metadata does not match consumed payload")
    dates = tuple(sorted(set((*seen.dates, record.available_on))))
    return _Seen(
        seen.evidence_id,
        seen.category,
        seen.date or record.available_on,
        dates,
        seen.required_date,
        seen.available,
        seen.reason,
        seen.payload,
        "; ".join(errors) if errors else None,
    )


def _record(seen: _Seen, as_of: datetime.date) -> LeakageEvidenceRecord:
    if seen.date is None:
        status = LeakageStatus.UNDATED
    elif not seen.available:
        status = LeakageStatus.EXCLUDED
    else:
        status = LeakageStatus.INCLUDED
    return LeakageEvidenceRecord(
        evidence_id=seen.evidence_id,
        category=seen.category,
        status=status,
        date=seen.date,
        dates=seen.dates,
        reason=seen.reason,
    )


def _information_entries(case: ForecastBacktestCase) -> tuple[_Seen, ...]:
    entries: list[_Seen] = []
    financials = case.point_in_time_financials
    if financials.retrieved_at is not None:
        date = financials.retrieved_at.date()
        entries.append(
            _Seen(
                "financials-retrieved-at",
                "normalized_snapshot",
                date,
                (date,),
                required_date=True,
            )
        )
    manifest = {item.identity: item for item in case.availability_manifest}
    for index, item in enumerate(financials.observations):
        dates = tuple(
            sorted(
                {
                    item.period_end,
                    *(([item.filed] if item.filed is not None else [])),
                }
            )
        )
        seen = _linked_entry(
            _Seen(
                _identity(item, "normalized_fact", index),
                "normalized_fact",
                item.filed,
                dates,
                required_date=True,
                payload=item,
            ),
            manifest,
            item,
        )
        available = (
            item.period_end <= case.as_of
            and seen.date is not None
            and seen.date <= case.as_of
        )
        entries.append(
            _Seen(
                seen.evidence_id,
                seen.category,
                seen.date,
                seen.dates,
                seen.required_date,
                available=available,
                reason=None
                if available
                else f"not available under point_in_time on {case.as_of.isoformat()}",
                payload=seen.payload,
                manifest_error=seen.manifest_error,
            )
        )

    input_value = case.reasoning_input
    if input_value is not None:
        field_categories = (
            ("segments", "operating", True),
            ("definitions", "operating", True),
            ("observations", "operating", True),
            ("manual_forward_driver_observations", "forward", True),
            ("management_guidance", "guidance", True),
            ("management_constraints", "constraint", True),
            ("investment_programs", "program", True),
            ("historical_facts", "historical_summary", True),
            ("research_evidence", "research", True),
            ("evidence_consensus", "consensus", True),
            ("manual_overrides", "manual", True),
        )
        for field_name, category, required in field_categories:
            for index, item in enumerate(getattr(input_value, field_name, ())):
                dates = _iter_dates(item)
                # A consensus object obtains its publication dates from every
                # contributor.  For an EvidenceReference, this finds the
                # nested filing_date without depending on one model shape.
                date = max(dates) if dates else None
                entries.append(
                    _linked_entry(
                        _Seen(
                            _identity(item, category, index),
                            category,
                            date,
                            dates,
                            required_date=required,
                            payload=item,
                        ),
                        manifest,
                        item,
                    )
                )
                if field_name == "evidence_consensus":
                    for _contributor_index, contributor in enumerate(
                        getattr(item, "contributors", ())
                    ):
                        contributor_dates = _iter_dates(contributor)
                        entries.append(
                            _linked_entry(
                                _Seen(
                                    canonical_information_identity(
                                        contributor, "consensus"
                                    ),
                                    "consensus_contributor",
                                    max(contributor_dates)
                                    if contributor_dates
                                    else None,
                                    contributor_dates,
                                    required_date=True,
                                    payload=contributor,
                                ),
                                manifest,
                                contributor,
                            )
                        )

    for label, payload in (
        ("evidence_snapshot", case.evidence_snapshot),
        ("operating_evidence", case.operating_evidence),
        ("hybrid_evidence", case.hybrid_evidence),
    ):
        if payload is None:
            continue
        dates = _iter_dates(payload)
        entries.append(
            _Seen(
                _identity(payload, label, 0),
                label,
                max(dates) if dates else None,
                dates,
                required_date=False,
            )
        )
        entries.extend(
            _linked_entry(item, manifest)
            for item in _required_payload_entries(payload, label)
        )
    return tuple(entries)


def audit_case(
    case: ForecastBacktestCase,
    *,
    availability_mode: ObservationAvailabilityMode = ObservationAvailabilityMode.POINT_IN_TIME,
) -> LeakageAudit:
    """Enumerate and audit every information-set date without calling services."""

    case = case if isinstance(case, ForecastBacktestCase) else ForecastBacktestCase.model_validate(case)
    mode = ObservationAvailabilityMode(availability_mode)
    entries = list(_information_entries(case))
    issues: list[str] = []
    information_dates: list[tuple[str, datetime.date]] = []
    included: list[LeakageEvidenceRecord] = []
    excluded: list[LeakageEvidenceRecord] = []
    undated: list[LeakageEvidenceRecord] = []
    for label, payload in (
        ("evidence_snapshot", case.evidence_snapshot),
        ("operating_evidence", case.operating_evidence),
        ("hybrid_evidence", case.hybrid_evidence),
    ):
        if payload is not None:
            issues.extend(
                f"{label} contains unrecognized evidence leaf/container {path}"
                for path in _unrecognized_payload_paths(payload, label)
            )
    used_identities = {item.evidence_id for item in entries}
    for record in case.availability_manifest:
        if record.identity not in used_identities:
            issues.append(
                f"availability manifest record {record.identity} is not linked to a consumed item"
            )
        if record.available_on > case.as_of:
            issues.append(
                f"availability manifest record {record.identity} dated "
                f"{record.available_on.isoformat()} after as_of={case.as_of.isoformat()}"
            )

    for seen in entries:
        for date in seen.dates or ((seen.date,) if seen.date is not None else ()):
            information_dates.append((seen.evidence_id, date))
        if seen.required_date and seen.date is None:
            issues.append(f"{seen.category}:{seen.evidence_id} has no availability date")
        if seen.manifest_error:
            issues.append(f"{seen.category}:{seen.evidence_id}: {seen.manifest_error}")
        for date in seen.dates or ((seen.date,) if seen.date is not None else ()):
            if date > case.as_of:
                issues.append(
                    f"{seen.category}:{seen.evidence_id} dated {date.isoformat()} after "
                    f"as_of={case.as_of.isoformat()}"
                )
        if seen.reason and not seen.required_date:
            excluded.append(_record(seen, case.as_of))
        else:
            item = _record(seen, case.as_of)
            if item.status == LeakageStatus.INCLUDED:
                included.append(item)
            elif item.status == LeakageStatus.EXCLUDED:
                excluded.append(item)
            else:
                undated.append(item)

    if case.reasoning_input is not None:
        if case.reasoning_input.as_of != case.as_of:
            issues.append("reasoning_input.as_of does not match case.as_of")
        if case.reasoning_input.forecast_years != case.fiscal_years:
            issues.append("reasoning_input forecast horizon does not match case fiscal_years")
        if case.reasoning_input.company_id != case.point_in_time_financials.company_id:
            issues.append("reasoning_input.company_id does not match case financials")
        if case.reasoning_input.ticker and case.reasoning_input.ticker.casefold() != case.ticker.casefold():
            issues.append("reasoning_input.ticker does not match case ticker")
        if case.reasoning_input.company_name and case.reasoning_input.company_name.casefold() != case.company.casefold():
            issues.append("reasoning_input.company_name does not match case company")
    if _contains_actual(case.reasoning_input) or _contains_actual(case.evidence_snapshot):
        issues.append("actual outcome data is present in the information set")

    return LeakageAudit(
        case_id=case.case_id,
        as_of=case.as_of,
        availability_mode=mode.value,
        included=tuple(sorted(included, key=lambda item: (item.category, item.evidence_id))),
        excluded=tuple(sorted(excluded, key=lambda item: (item.category, item.evidence_id))),
        undated=tuple(sorted(undated, key=lambda item: (item.category, item.evidence_id))),
        information_dates=tuple(sorted(set(information_dates), key=lambda item: (item[0], item[1]))),
        issues=tuple(dict.fromkeys(issues)),
        valid=not issues,
    )


def enforce_leakage(case: ForecastBacktestCase, **kwargs: Any) -> LeakageAudit:
    audit = audit_case(case, **kwargs)
    if not audit.valid:
        raise LeakageError(audit)
    return audit


def audit_actual_outcomes(actuals: ActualOutcomeData, *, as_of: datetime.date) -> ActualOutcomeAudit:
    """Audit realized dates independently; this output is never a prompt input."""

    actuals = actuals if isinstance(actuals, ActualOutcomeData) else ActualOutcomeData.model_validate(actuals)
    dates: list[tuple[str, datetime.date]] = []
    for index, item in enumerate(actuals.observations):
        dates.append((f"actual-financial-{index}:{item.metric}:{item.fiscal_year}", item.period_end))
        if item.source_date is not None:
            dates.append((f"actual-financial-source-{index}", item.source_date))
    for index, item in enumerate(actuals.assumption_outcomes):
        if item.source_date is not None:
            dates.append((f"actual-assumption-source-{index}:{item.target}", item.source_date))
    dates.extend((f"actual-outcome-date-{index}", date) for index, date in enumerate(actuals.outcome_dates))
    subsequent = tuple(sorted(item for item in dates if item[1] > as_of))
    pre_cutoff = tuple(sorted(item for item in dates if item[1] <= as_of))
    return ActualOutcomeAudit(
        as_of=as_of,
        dates=tuple(sorted(dates)),
        subsequent_dates=subsequent,
        pre_cutoff_dates=pre_cutoff,
    )


def cutoff_financials(
    financials: NormalizedCompanyFinancials,
    *,
    as_of: datetime.date,
    availability_mode: ObservationAvailabilityMode = ObservationAvailabilityMode.POINT_IN_TIME,
    availability_manifest: Sequence[InformationAvailabilityRecord] = (),
) -> NormalizedCompanyFinancials:
    """Return the exact immutable snapshot a baseline is allowed to consume."""

    mode = ObservationAvailabilityMode(availability_mode)
    if mode != ObservationAvailabilityMode.POINT_IN_TIME:
        raise ValueError("Evaluation cutoff financials require POINT_IN_TIME availability")
    manifest = {
        item.identity: item
        for item in (
            record
            if isinstance(record, InformationAvailabilityRecord)
            else InformationAvailabilityRecord.model_validate(record)
            for record in availability_manifest
        )
    }

    def validated_observation(item: Any) -> tuple[Any, _Seen]:
        dates = tuple(
            sorted(
                {
                    item.period_end,
                    *(([item.filed] if item.filed is not None else [])),
                }
            )
        )
        seen = _linked_entry(
            _Seen(
                canonical_information_identity(item, "normalized_fact"),
                "normalized_fact",
                item.filed,
                dates,
                required_date=True,
                payload=item,
            ),
            manifest,
            item,
        )
        if seen.manifest_error:
            raise ValueError(
                f"Availability manifest validation failed for {seen.evidence_id}: "
                f"{seen.manifest_error}"
            )
        return item, seen

    observations = tuple(
        item
        for item, seen in map(validated_observation, financials.observations)
        if item.period_end <= as_of
        and seen.date is not None
        and seen.date <= as_of
        and all(date <= as_of for date in seen.dates)
    )
    if isinstance(financials, ImmutableNormalizedCompanyFinancials):
        payload = {
            "provider": financials.provider,
            "company_id": financials.company_id,
            "company_name": financials.company_name,
            "ticker": financials.ticker,
            "identifiers": financials.identifiers,
            "retrieved_at": financials.retrieved_at,
        }
    else:
        payload = financials.model_dump(mode="python")
    payload["observations"] = observations
    return ImmutableNormalizedCompanyFinancials.model_validate(payload)


audit_information_set = audit_case
validate_information_set = enforce_leakage


__all__ = [
    "LeakageError",
    "LeakageAuditor",
    "ActualOutcomeAudit",
    "audit_case",
    "audit_information_set",
    "canonical_information_identity",
    "canonical_information_content_hash",
    "enforce_leakage",
    "validate_information_set",
    "audit_actual_outcomes",
    "cutoff_financials",
]
