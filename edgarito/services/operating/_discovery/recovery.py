"""Gap recovery, vocabulary retries, and deterministic extraction merging."""

from __future__ import annotations

import re
from typing import Any

from edgarito.schemas.operating_history import OperatingEvidenceGap
from edgarito.services.guidance.documents import extract_operating_context
from edgarito.services.operating._discovery.audit import (
    retry_summary_audit,
    retry_vocabulary_audit,
)
from edgarito.services.operating._discovery.contracts import DiscoveryState
from edgarito.services.operating.history import OperatingHistoryAssembler


def append_unique(
    target: list[Any],
    seen: dict[Any, Any],
    key: Any,
    item: Any,
    warnings: list[str],
    label: str,
) -> bool:
    previous = seen.get(key)
    if previous is None:
        seen[key] = item
        target.append(item)
        return True
    if previous != item:
        warnings.append(f"Conflicting duplicate {label} retained from first source")
    return False


def merge_segment_identity(previous, current):
    if previous is None or previous.name != previous.segment_id:
        return current
    if current.name == current.segment_id:
        return previous
    return previous.model_copy(update={"name": current.name})


def merge_extraction_entry(entry, *, state: DiscoveryState) -> None:
    for item in entry.segments:
        item = merge_segment_identity(state.seen_segments.get(item.segment_id), item)
        append_unique(
            state.segments,
            state.seen_segments,
            item.segment_id,
            item,
            state.warnings,
            "segment",
        )
    for item in entry.definitions:
        append_unique(
            state.definitions,
            state.seen_definitions,
            (item.segment_id, item.driver_id),
            item,
            state.warnings,
            "driver definition",
        )
    for item in entry.observations:
        key = (
            item.segment_id,
            OperatingHistoryAssembler._canonical_metric(item.driver_id),
            item.fiscal_year,
            item.fiscal_period,
            item.period_key,
            item.scope,
            item.scope_evidence,
            item.is_total,
            item.is_component,
        )
        previous = state.seen_observations.get(key)
        if previous is None:
            previous = next(
                (
                    existing
                    for existing in state.observations
                    if (
                        existing.segment_id,
                        OperatingHistoryAssembler._canonical_metric(
                            existing.driver_id
                        ),
                        existing.fiscal_year,
                        existing.fiscal_period,
                        existing.period_key,
                        existing.scope,
                        existing.scope_evidence,
                        existing.is_total,
                        existing.is_component,
                    )
                    == key
                ),
                None,
            )
        if previous is not None:
            references = tuple(
                dict.fromkeys(
                    reference
                    for reference in (
                        *previous.source_provenance,
                        previous.evidence,
                        *item.source_provenance,
                        item.evidence,
                    )
                    if reference is not None
                )
            )
            base = (
                item if previous.origin == "derived" and item.origin != "derived" else previous
            )
            merged = base.model_copy(
                update={
                    "evidence": base.evidence or previous.evidence or item.evidence,
                    "source_provenance": references,
                    "basis": base.basis or previous.basis or item.basis,
                    "scope": base.scope or previous.scope or item.scope,
                    "scope_evidence": base.scope_evidence
                    or previous.scope_evidence
                    or item.scope_evidence,
                    "is_total": base.is_total or item.is_total,
                    "is_component": base.is_component or item.is_component,
                    "exhaustive": base.exhaustive or item.exhaustive,
                }
            )
            state.seen_observations[key] = merged
            for index, existing in enumerate(state.observations):
                existing_key = (
                    existing.segment_id,
                    OperatingHistoryAssembler._canonical_metric(existing.driver_id),
                    existing.fiscal_year,
                    existing.fiscal_period,
                    existing.period_key,
                    existing.scope,
                    existing.scope_evidence,
                    existing.is_total,
                    existing.is_component,
                )
                if existing_key == key:
                    state.observations[index] = merged
                    break
            continue
        if append_unique(
            state.observations,
            state.seen_observations,
            key,
            item,
            state.warnings,
            "operating observation",
        ) and item.origin == "management_guidance":
            state.management_constraints.append(item)
    for item in entry.investment_programs:
        append_unique(
            state.programs,
            state.seen_programs,
            (item.program_id, item.fiscal_year, item.fiscal_period),
            item,
            state.warnings,
            "investment program",
        )
    state.rejected.extend(entry.rejected)
    state.unsupported.extend(entry.unsupported_evidence)
    state.missing.extend(entry.missing_evidence)
    state.unusable.extend(entry.unusable_reasons)
    state.audit_records.extend(entry.audit_records)


def gap_terms(gaps: tuple[OperatingEvidenceGap, ...]) -> tuple[str, ...]:
    aliases = {
        "revenue": ("revenue", "revenues", "sales", "net sales"),
        "volume": (
            "volume",
            "volumes",
            "units",
            "deliveries",
            "shipments",
            "production",
        ),
        "price": ("price", "prices", "pricing", "asp", "average selling price"),
        "subscribers": ("subscriber", "subscribers", "users", "members"),
        "arpu": ("arpu", "average revenue per user"),
    }
    return tuple(
        dict.fromkeys(
            term
            for gap in gaps
            for term in aliases.get(gap.metric.casefold(), (gap.metric,))
        )
    )


def targeted_gap_documents(
    retry_documents: list[tuple[Any, Any, str]],
    gaps: tuple[OperatingEvidenceGap, ...],
    vocabulary_terms: list[Any] | tuple[Any, ...],
) -> tuple[tuple[Any, Any, str], ...]:
    terms = gap_terms(gaps)
    discovered_terms = tuple(
        str(getattr(item, "raw_term", item)) for item in vocabulary_terms
    )
    all_terms = tuple(dict.fromkeys((*terms, *discovered_terms)))

    def score(item: tuple[Any, Any, str]) -> tuple[int, str, str]:
        filing, document, text = item
        clean = text.casefold()
        score_value = sum(
            3
            for term in all_terms
            if re.search(rf"\b{re.escape(term.casefold())}\b", clean)
        )
        for gap in gaps:
            if gap.fiscal_year is not None:
                if filing.report_date and filing.report_date.year == gap.fiscal_year:
                    score_value += 8
                if filing.filing_date.year in {gap.fiscal_year, gap.fiscal_year + 1}:
                    score_value += 4
                if re.search(
                    rf"\b(?:FY\s*)?{gap.fiscal_year}\b", text, re.IGNORECASE
                ):
                    score_value += 8
            if gap.fiscal_period == "FQ" and gap.period_key:
                if re.search(rf"\b{gap.period_key}\b", text, re.IGNORECASE):
                    score_value += 6
                if filing.report_date and (
                    filing.report_date.month - 1
                ) // 3 + 1 == int(gap.period_key[-1]):
                    score_value += 5
        return (
            -score_value,
            filing.report_date.isoformat() if filing.report_date else "",
            document.filename,
        )

    return tuple(sorted(retry_documents, key=score))


def targeted_gap_context(
    clean_text: str,
    gaps: tuple[OperatingEvidenceGap, ...],
    vocabulary_terms: tuple[str, ...],
    *,
    max_chars: int = 24_000,
    window_chars: int = 1_000,
) -> str:
    terms = tuple(dict.fromkeys((*vocabulary_terms, *gap_terms(gaps))))
    patterns = [rf"\b{re.escape(term)}\b" for term in terms if term]
    patterns.extend(
        rf"\b(?:FY\s*)?{gap.fiscal_year}\b"
        for gap in gaps
        if gap.fiscal_year is not None
    )
    patterns.extend(rf"\b{gap.period_key}\b" for gap in gaps if gap.period_key)
    matches = [
        match
        for pattern in patterns
        for match in re.finditer(pattern, clean_text, re.IGNORECASE)
    ]
    if not matches:
        return ""
    windows: list[tuple[int, int]] = []
    for match in sorted(matches, key=lambda item: item.start()):
        start = max(0, match.start() - window_chars)
        end = min(len(clean_text), match.end() + window_chars)
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
    selected: list[str] = []
    used = 0
    for start, end in windows:
        snippet = clean_text[start:end]
        if used + len(snippet) + (2 if selected else 0) > max_chars:
            break
        selected.append(snippet)
        used += len(snippet) + (2 if selected else 0)
    return "\n\n".join(selected)


def _assemble(service: Any, state: DiscoveryState, company_id: str) -> Any:
    history = service._history_assembler.assemble(
        state.observations,
        segments=state.segments,
        definitions=state.definitions,
        company_id=company_id,
    )
    state.segments = list(history.segments)
    state.definitions = list(history.definitions)
    state.observations = list(history.observations)
    return history


async def recover_sec_gaps(
    service: Any,
    state: DiscoveryState,
    *,
    history: Any,
    company_id: str,
    as_of,
    extraction_years: tuple[int, ...],
    industry,
    business_archetype,
) -> tuple[Any, Any, tuple[OperatingEvidenceGap, ...]]:
    """Retry source-ranked SEC documents for detected history gaps."""

    initial_history = history
    detected_gaps = initial_history.audit.gaps_detected
    if detected_gaps and state.candidate_documents:
        normal_vocabulary_terms = (
            tuple(
                item[0]
                for item in service._vocabulary.normal_terms(
                    industry, business_archetype
                )
            )
            if service._vocabulary is not None
            else ()
        )
        targeted_documents = targeted_gap_documents(
            state.candidate_documents,
            detected_gaps,
            (*state.vocabulary_terms, *normal_vocabulary_terms),
        )
        targeted_entries = 0
        targeted_keys: set[tuple[str, str]] = set()
        for filing, document, clean_text in targeted_documents:
            if targeted_entries >= service.max_gap_documents:
                break
            key = (filing.accession_number, document.filename)
            if key in targeted_keys:
                continue
            targeted_keys.add(key)
            if state.documents_inspected >= service.max_documents and key not in {
                (item[0].accession_number, item[1].filename)
                for item in state.retry_documents
            }:
                state.documents_inspected += 1
            try:
                gap_terms_value = gap_terms(detected_gaps)
                context_text = targeted_gap_context(
                    clean_text,
                    detected_gaps,
                    gap_terms_value,
                )
                if not context_text:
                    continue
                entry, cache_hit = await service._extractor.extract(
                    filing,
                    document,
                    clean_text,
                    valuation_date=as_of,
                    source_text=clean_text,
                    context_text=context_text,
                    fiscal_years=extraction_years,
                )
                state.hits += int(cache_hit)
                state.misses += int(not cache_hit)
                merge_extraction_entry(entry, state=state)
                targeted_entries += 1
                targeted_history = _assemble(service, state, company_id)
                if not targeted_history.audit.gaps_unresolved:
                    break
            except Exception as exc:
                state.warnings.append(
                    f"Operating evidence gap extraction skipped for "
                    f"{document.filename}: {exc}"
                )
        history = _assemble(service, state, company_id)
        if targeted_entries:
            history = _assemble(service, state, company_id)
            state.warnings.extend(history.audit.warnings)
    return history, initial_history, detected_gaps


async def retry_vocabulary(
    service: Any,
    state: DiscoveryState,
    *,
    history: Any,
    company_id: str,
    as_of,
    extraction_years: tuple[int, ...],
    industry,
    business_archetype,
) -> tuple[Any, str]:
    """Retry extraction with grounded KPI vocabulary when history is unusable."""

    non_revenue = sum(
        1
        for item in state.observations
        if item.driver_id
        not in {
            "revenue",
            "segment_revenue",
            "sales",
            "net_sales",
            "total_revenue",
            "revenue_growth",
            "sales_growth",
        }
    )
    retry_reason = ""
    if not state.observations or non_revenue == 0:
        retry_reason = "accepted non-revenue observations are zero"
    elif not history.audit.accepted_pairs or history.audit.missing_pairs:
        retry_reason = "usable reconstruction pairs or coverage unavailable"
    if service._vocabulary is None or not retry_reason or not state.retry_documents:
        return history, retry_reason

    retry_terms: list[Any] = []
    for filing, document, clean_text in state.retry_documents:
        try:
            discovered = state.discovered_by_document.get(document.filename)
            if not discovered:
                discovered, audit = await service._vocabulary.discover(
                    context=extract_operating_context(clean_text),
                    source_document=document.filename,
                    source_text=clean_text,
                    industry=industry,
                    business_archetype=business_archetype,
                    as_of=as_of,
                    force=True,
                    fallback_reason=retry_reason,
                )
            else:
                discovered = tuple(discovered)
            audit = retry_vocabulary_audit(
                service._vocabulary,
                discovered,
                industry=industry,
                business_archetype=business_archetype,
                retry_reason=retry_reason,
            )
            retry_terms.extend(discovered)
            state.vocabulary_audits.append(audit)
            if not discovered:
                continue
            context_text = (
                extract_operating_context(clean_text)
                + "\n\nAdditional grounded KPI terminology: "
                + ", ".join(item.raw_term for item in discovered)
            )
            entry, cache_hit = await service._extractor.extract(
                filing,
                document,
                clean_text,
                valuation_date=as_of,
                source_text=clean_text,
                context_text=context_text,
                fiscal_years=extraction_years,
            )
            state.hits += int(cache_hit)
            state.misses += int(not cache_hit)
            merge_extraction_entry(entry, state=state)
        except Exception as exc:
            state.warnings.append(
                f"KPI vocabulary retry skipped for {document.filename}: {exc}"
            )
    if retry_terms:
        state.vocabulary_terms.extend(retry_terms)
        state.vocabulary_audits.append(
            retry_summary_audit(
                service._vocabulary,
                retry_terms,
                industry=industry,
                business_archetype=business_archetype,
                retry_reason=retry_reason,
                new_observations=len(state.observations) - history.audit.accepted_observations,
            )
        )
        history = _assemble(service, state, company_id)
    return history, retry_reason


__all__ = [
    "append_unique",
    "gap_terms",
    "merge_extraction_entry",
    "merge_segment_identity",
    "recover_sec_gaps",
    "retry_vocabulary",
    "targeted_gap_context",
    "targeted_gap_documents",
]
