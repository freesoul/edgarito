"""Discovery quality audits and vocabulary retry diagnostics."""

from __future__ import annotations

from typing import Any

from edgarito.schemas.vocabulary import KpiVocabularyAudit
from edgarito.services.operating.history import OperatingHistoryAssembler
from edgarito.services.operating.vocabulary import normalize_industry_namespace


def merge_vocabulary_audits(audits: list[Any]) -> Any | None:
    if not audits:
        return None
    first = audits[0]
    return first.model_copy(
        update={
            "global_count": max(item.global_count for item in audits),
            "industry_count": max(item.industry_count for item in audits),
            "discovered_count": sum(item.discovered_count for item in audits),
            "rejected_count": sum(item.rejected_count for item in audits),
            "cache_status": (
                "hit" if any(item.cache_status == "hit" for item in audits) else "miss"
            ),
            "terms": tuple(
                dict.fromkeys(term for item in audits for term in item.terms)
            ),
            "diagnostics": tuple(
                dict.fromkeys(
                    diagnostic for item in audits for diagnostic in item.diagnostics
                )
            ),
            "raw_industry": next(
                (item.raw_industry for item in audits if item.raw_industry), ""
            ),
            "normalized_industry": next(
                (
                    item.normalized_industry
                    for item in audits
                    if item.normalized_industry
                ),
                "",
            ),
            "selected_archetype": next(
                (item.selected_archetype for item in audits if item.selected_archetype),
                "",
            ),
            "fallback_triggered": any(item.fallback_triggered for item in audits),
            "fallback_reason": next(
                (item.fallback_reason for item in audits if item.fallback_reason), ""
            ),
            "retry": any(item.retry for item in audits),
            "new_observations": sum(item.new_observations for item in audits),
            "validated_terms": tuple(
                dict.fromkeys(term for item in audits for term in item.validated_terms)
            ),
        }
    )


def retry_vocabulary_audit(
    vocabulary: Any,
    discovered: tuple[Any, ...] | list[Any],
    *,
    industry: str | None,
    business_archetype: Any | None,
    retry_reason: str,
) -> KpiVocabularyAudit:
    return KpiVocabularyAudit(
        global_count=len(vocabulary.GLOBAL_KPI_TERMS),
        terms=tuple(
            term for term, _metric in vocabulary.normal_terms(industry, business_archetype)
        ),
        raw_industry=str(industry or ""),
        normalized_industry=normalize_industry_namespace(industry),
        selected_archetype=str(
            getattr(business_archetype, "value", business_archetype) or ""
        ),
        fallback_triggered=True,
        fallback_reason=retry_reason,
        retry=True,
        discovered_count=len(discovered),
        validated_terms=tuple(item.raw_term for item in discovered),
        cache_status="retry",
    )


def retry_summary_audit(
    vocabulary: Any,
    retry_terms: list[Any],
    *,
    industry: str | None,
    business_archetype: Any | None,
    retry_reason: str,
    new_observations: int,
) -> KpiVocabularyAudit:
    return KpiVocabularyAudit(
        global_count=len(vocabulary.GLOBAL_KPI_TERMS),
        industry_count=vocabulary.industry_term_count(industry, business_archetype),
        terms=tuple(
            item[0] for item in vocabulary.normal_terms(industry, business_archetype)
        ),
        discovered_count=len(retry_terms),
        validated_terms=tuple(item.raw_term for item in retry_terms),
        raw_industry=str(industry or ""),
        normalized_industry=normalize_industry_namespace(industry),
        selected_archetype=str(
            getattr(business_archetype, "value", business_archetype) or ""
        ),
        fallback_triggered=True,
        fallback_reason=retry_reason,
        retry=True,
        new_observations=new_observations,
        cache_status="retry",
    )


def finalize_history_audit(
    history: Any,
    initial_history: Any,
    detected_gaps: tuple[Any, ...],
    ir_resolved: tuple[Any, ...],
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
    def gap_with_status(gap, status: str):
        references = tuple(
            dict.fromkeys(
                f"{reference.accession or ''}:{reference.document_name or ''}"
                for item in history.observations
                if item.segment_id == gap.segment_id
                and OperatingHistoryAssembler._canonical_metric(item.driver_id)
                == gap.metric
                and (gap.fiscal_year is None or item.fiscal_year == gap.fiscal_year)
                and item.fiscal_period == gap.fiscal_period
                and (gap.period_key is None or item.period_key == gap.period_key)
                for reference in (*item.source_provenance, item.evidence)
                if reference is not None
            )
        )
        return gap.model_copy(
            update={
                "status": status,
                "source_documents": tuple(
                    dict.fromkeys((*gap.source_documents, *references))
                ),
            }
        )

    resolved_gaps = tuple(
        gap_with_status(gap, "resolved")
        for gap in detected_gaps
        if not any(
            item.key == gap.key
            or (
                gap.fiscal_year is None
                and item.segment_id == gap.segment_id
                and item.metric.casefold() == gap.metric.casefold()
            )
            for item in history.audit.gaps_unresolved
        )
    )
    unresolved_gaps = tuple(
        gap_with_status(gap, "unresolved") for gap in history.audit.gaps_unresolved
    )
    history = history.model_copy(
        update={
            "audit": history.audit.model_copy(
                update={
                    "gaps_detected": detected_gaps,
                    "gaps_resolved": resolved_gaps,
                    "gaps_unresolved": unresolved_gaps,
                    "gaps_resolved_sec": tuple(
                        item
                        for item in resolved_gaps
                        if item.key not in {gap.key for gap in ir_resolved}
                    ),
                    "gaps_resolved_ir": ir_resolved,
                    "gap_diagnostics": tuple(
                        dict.fromkeys(
                            (
                                *history.audit.gap_diagnostics,
                                *(
                                    item
                                    for gap in unresolved_gaps
                                    for item in gap.diagnostics
                                ),
                            )
                        )
                    ),
                    "new_fy_periods": tuple(
                        sorted(
                            set(history.audit.historical_revenue_pairs)
                            - set(initial_history.audit.historical_revenue_pairs)
                        )
                    ),
                    "new_ltm_periods": tuple(
                        sorted(
                            f"{item.segment_id}/FY{item.fiscal_year}/LTM"
                            for item in history.observations
                            if item.fiscal_period == "LTM"
                            and not any(
                                old.segment_id == item.segment_id
                                and old.driver_id == item.driver_id
                                and old.fiscal_year == item.fiscal_year
                                and old.fiscal_period == "LTM"
                                for old in initial_history.observations
                            )
                        )
                    ),
                }
            )
        }
    )
    return history, resolved_gaps, unresolved_gaps


__all__ = [
    "finalize_history_audit",
    "merge_vocabulary_audits",
    "retry_summary_audit",
    "retry_vocabulary_audit",
]
