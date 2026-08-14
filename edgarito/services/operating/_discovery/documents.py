"""SEC filing inventory, document selection, and first-pass extraction."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any

from edgarito.schemas.operating import OperatingDocumentAudit
from edgarito.schemas.vocabulary import KpiVocabularyAudit
from edgarito.services.guidance.documents import (
    clean_document_text,
    extract_operating_context,
    guidance_keyword_hits,
    is_exhibit_document,
    is_periodic_filing,
)
from edgarito.services.openai import OpenAIAuthenticationError
from edgarito.services.operating._discovery.contracts import DiscoveryState
from edgarito.services.operating._discovery.recovery import (
    merge_extraction_entry,
)
from edgarito.services.operating.extraction import operating_keyword_hits
from edgarito.services.operating.vocabulary import normalize_industry_namespace


@dataclass(frozen=True)
class FilingPreparation:
    cik: int
    filings: tuple[Any, ...]
    selected_filings: tuple[Any, ...]
    extraction_years: tuple[int, ...]
    raw_filings_received: int
    raw_filings_in_range: int
    candidate_filings: int


async def prepare_filings(
    service: Any,
    *,
    ticker: str | None,
    cik: int | None,
    financials: Any | None,
    company_id: str | int | None,
    as_of: datetime.date,
    refresh_sec: bool,
    fiscal_years: tuple[int, ...] | None,
) -> tuple[FilingPreparation | None, str | None]:
    if ticker is None and financials is not None:
        ticker = getattr(financials, "ticker", None)
    if cik is None and company_id is not None:
        try:
            cik = int(company_id)
        except (TypeError, ValueError):
            pass
    if cik is None:
        if not ticker:
            return None, "Operating evidence discovery skipped: no SEC identifier"
        try:
            cik = await service._edgar.get_cik(
                ticker,
                use_cache=not refresh_sec,
                make_cache=True,
            )
        except Exception as exc:
            return (
                None,
                "Operating evidence discovery skipped: SEC identifier lookup "
                f"failed: {exc}",
            )

    try:
        get_filings = getattr(service._edgar, "get_raw_operating_filings", None)
        if get_filings is None:
            get_filings = getattr(service._edgar, "get_operating_filings", None)
        if get_filings is None:
            get_filings = service._edgar.get_guidance_filings
        filings = await get_filings(
            cik,
            as_of=as_of,
            lookback_days=service.lookback_days,
            use_cache=not refresh_sec,
            make_cache=True,
        )
        raw_filings_received = len(filings)
        raw_filings_in_range = len(filings)
    except Exception as exc:
        return (
            None,
            "Operating evidence discovery skipped: SEC filing retrieval failed: "
            f"{exc}",
        )

    try:
        selected_filings = service._selector.select_operating_filings(
            filings,
            limit=service.max_filings,
        )
        candidate_filings = len(selected_filings)
    except Exception as exc:
        return (
            None,
            "Operating evidence discovery skipped: filing selection failed: "
            f"{exc}",
        )

    selected_filings = [
        *[filing for filing in selected_filings if is_periodic_filing(filing)],
        *[filing for filing in selected_filings if not is_periodic_filing(filing)],
    ]
    extraction_years = (
        tuple(
            sorted(
                {
                    *(fiscal_years or ()),
                    *(
                        filing.report_date.year
                        for filing in selected_filings
                        if filing.report_date is not None
                    ),
                    *(filing.filing_date.year for filing in selected_filings),
                }
            )
        )
        or fiscal_years
    )
    return (
        FilingPreparation(
            cik=cik,
            filings=tuple(filings),
            selected_filings=tuple(selected_filings),
            extraction_years=tuple(extraction_years or ()),
            raw_filings_received=raw_filings_received,
            raw_filings_in_range=raw_filings_in_range,
            candidate_filings=candidate_filings,
        ),
        None,
    )


def initialize_vocabulary_audit(
    service: Any,
    state: DiscoveryState,
    *,
    industry: str | None,
    business_archetype: Any | None,
) -> None:
    if service._vocabulary is None:
        return
    normal = service._vocabulary.normal_terms(industry, business_archetype)
    state.vocabulary_audits.append(
        KpiVocabularyAudit(
            global_count=(
                len(service._vocabulary.GLOBAL_KPI_TERMS)
                if hasattr(service._vocabulary, "GLOBAL_KPI_TERMS")
                else 0
            ),
            industry_count=service._vocabulary.industry_term_count(
                industry, business_archetype
            ),
            terms=tuple(term for term, _metric in normal),
            cache_status="not_needed",
            raw_industry=str(industry or ""),
            normalized_industry=normalize_industry_namespace(industry),
            selected_archetype=str(
                getattr(business_archetype, "value", business_archetype) or ""
            ),
        )
    )


async def collect_sec_documents(
    service: Any,
    preparation: FilingPreparation,
    state: DiscoveryState,
    *,
    as_of: datetime.date,
    refresh_sec: bool,
    industry: str | None,
    business_archetype: Any | None,
) -> None:
    """Inspect selected documents and merge accepted first-pass evidence."""

    for filing_index, filing in enumerate(preparation.selected_filings):
        state.filings_inspected += 1
        try:
            populated = await service._edgar.get_filing_documents(
                filing,
                use_cache=not refresh_sec,
                make_cache=True,
            )
        except Exception as exc:
            state.warnings.append(
                f"Operating document retrieval skipped for "
                f"{filing.accession_number}: {exc}"
            )
            continue

        remaining_capacity = service.max_documents - state.documents_inspected
        remaining_filings = len(preparation.selected_filings) - filing_index - 1
        reserved_for_later = min(
            remaining_filings,
            max(0, remaining_capacity - 1),
        )
        document_limit = max(
            0,
            min(
                service.max_documents_per_filing,
                remaining_capacity - reserved_for_later,
            ),
        )
        try:
            documents = service._selector.select_operating_documents(
                populated,
                limit=document_limit if document_limit > 0 else 0,
            )
        except Exception as exc:
            state.warnings.append(
                f"Operating document selection skipped for "
                f"{filing.accession_number}: {exc}"
            )
            continue
        all_candidates = service._selector.operating_document_candidates(populated)
        state.exhibits_found += sum(
            is_exhibit_document(document) for document in populated.documents
        )
        for candidate in all_candidates:
            candidate_text = clean_document_text(candidate.content)
            if candidate_text:
                state.candidate_documents.append((filing, candidate, candidate_text))
        if not documents and any(
            document.is_pdf for document in populated.documents
        ):
            state.warnings.append(
                f"{filing.form} {filing.accession_number} has no selected "
                "HTML/text operating evidence document; PDF extraction is unsupported"
            )

        for document in documents:
            if state.documents_inspected >= service.max_documents:
                break
            state.documents_inspected += 1
            clean_text = clean_document_text(document.content)
            context_text = extract_operating_context(clean_text)
            state.retry_documents.append((filing, document, clean_text))
            if service._vocabulary is not None:
                try:
                    discovered, vocabulary_audit = await service._vocabulary.discover(
                        context=context_text,
                        source_document=document.filename,
                        source_text=clean_text,
                        industry=industry,
                        business_archetype=business_archetype,
                        as_of=as_of,
                    )
                    state.vocabulary_terms.extend(discovered)
                    state.discovered_by_document[document.filename] = tuple(discovered)
                    state.vocabulary_audits.append(vocabulary_audit)
                    if discovered:
                        context_text += (
                            "\n\nAdditional grounded KPI terminology: "
                            + ", ".join(item.raw_term for item in discovered)
                        )
                except Exception as exc:
                    state.warnings.append(f"KPI vocabulary discovery skipped: {exc}")
            document_audit = OperatingDocumentAudit(
                filing_form=filing.form,
                filing_date=filing.filing_date,
                accession_number=filing.accession_number,
                filename=document.filename,
                document_type=document.document_type,
                is_primary=service._selector.is_primary(filing, document),
                cleaned_size=len(clean_text),
                bounded_context_size=len(context_text),
                keyword_hits={
                    **guidance_keyword_hits(clean_text),
                    **operating_keyword_hits(clean_text),
                },
            )
            if document.is_pdf:
                state.warnings.append(
                    f"Operating evidence skipped for {filing.accession_number} "
                    f"{document.filename}: PDF extraction is unsupported"
                )
                state.document_audits.append(document_audit)
                continue
            if not clean_text:
                state.warnings.append(
                    f"Operating evidence skipped for {filing.accession_number} "
                    f"{document.filename}: document text is empty"
                )
                state.document_audits.append(document_audit)
                continue
            try:
                entry, cache_hit = await service._extractor.extract(
                    filing,
                    document,
                    clean_text,
                    valuation_date=as_of,
                    source_text=clean_text,
                    context_text=context_text,
                    fiscal_years=preparation.extraction_years,
                )
            except OpenAIAuthenticationError as exc:
                state.warnings.append(
                    f"Operating evidence extraction skipped for "
                    f"{filing.accession_number} {document.filename}: "
                    f"OpenAI authentication failed ({exc})"
                )
                state.document_audits.append(document_audit)
                continue
            except Exception as exc:
                state.warnings.append(
                    f"Operating evidence extraction skipped for "
                    f"{filing.accession_number} {document.filename}: {exc}"
                )
                state.document_audits.append(document_audit)
                continue

            state.hits += int(cache_hit)
            state.misses += int(not cache_hit)
            merge_extraction_entry(
                entry,
                state=state,
            )
            state.document_audits.append(
                document_audit.model_copy(
                    update={
                        "accepted_segments": len(entry.segments),
                        "accepted_definitions": len(entry.definitions),
                        "accepted_observations": len(entry.observations),
                        "accepted_investment_programs": len(entry.investment_programs),
                        "rejected_records": len(entry.rejected),
                        "unsupported_evidence": len(entry.unsupported_evidence),
                        "missing_evidence": len(entry.missing_evidence),
                        "unusable_evidence": len(entry.unusable_reasons),
                    }
                )
            )
            for reason in entry.unusable_reasons:
                state.warnings.append(f"Operating evidence unusable: {reason}")


__all__ = [
    "FilingPreparation",
    "collect_sec_documents",
    "initialize_vocabulary_audit",
    "prepare_filings",
]
