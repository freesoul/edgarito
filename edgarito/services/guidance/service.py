from __future__ import annotations

import datetime
from dataclasses import dataclass

from edgarito.schemas.guidance.management import (
    GuidanceDocumentAudit,
    GuidanceRejection,
    ManagementGuidance,
)
from edgarito.services.guidance.documents import (
    GuidanceDocumentSelector,
    clean_document_text,
    extract_guidance_context,
    guidance_keyword_hits,
    is_periodic_filing,
)
from edgarito.services.guidance.extraction import ManagementGuidanceExtractor
from edgarito.services.guidance.resolver import ManagementGuidanceResolver
from edgarito.services.openai import OpenAIAuthenticationError
from edgarito.services.providers.edgar import EdgarClient


@dataclass(frozen=True)
class GuidanceDiscoveryResult:
    records: tuple[ManagementGuidance, ...] = ()
    rejected: tuple[GuidanceRejection, ...] = ()
    warnings: tuple[str, ...] = ()
    cache_hits: int = 0
    cache_misses: int = 0
    filings_inspected: int = 0
    documents_inspected: int = 0
    extracted_guidance_records: int = 0
    rejected_records: int = 0
    document_audits: tuple[GuidanceDocumentAudit, ...] = ()


class ManagementGuidanceService:
    """Optional, failure-isolated SEC-to-current-guidance orchestration."""

    def __init__(
        self,
        edgar: EdgarClient,
        extractor: ManagementGuidanceExtractor,
        *,
        selector: GuidanceDocumentSelector | None = None,
        lookback_days: int = 180,
        max_filings: int = 4,
        max_documents_per_filing: int = 3,
        max_documents: int = 6,
    ) -> None:
        self._edgar = edgar
        self._extractor = extractor
        self._selector = selector or GuidanceDocumentSelector()
        self.lookback_days = lookback_days
        self.max_filings = max_filings
        self.max_documents_per_filing = max_documents_per_filing
        self.max_documents = max_documents

    async def retrieve(
        self,
        *,
        ticker: str | None,
        cik: int | None,
        as_of: datetime.date,
        refresh_sec: bool = False,
    ) -> GuidanceDiscoveryResult:
        if cik is None:
            if not ticker:
                return GuidanceDiscoveryResult(
                    warnings=("Management guidance skipped: no SEC identifier",)
                )
            cik = await self._edgar.get_cik(
                ticker, use_cache=not refresh_sec, make_cache=True
            )
        filings = await self._edgar.get_guidance_filings(
            cik,
            as_of=as_of,
            lookback_days=self.lookback_days,
            use_cache=not refresh_sec,
            make_cache=True,
        )
        filings = self._selector.select_filings(filings, limit=self.max_filings)
        # The selector retains current-report ranking, but process selected
        # periodic filings first so their guaranteed primary document cannot
        # be consumed by the service-wide document budget before it is seen.
        filings = [
            *[filing for filing in filings if is_periodic_filing(filing)],
            *[filing for filing in filings if not is_periodic_filing(filing)],
        ]
        filings_inspected = len(filings)
        records: list[ManagementGuidance] = []
        rejected: list[GuidanceRejection] = []
        warnings: list[str] = []
        hits = 0
        misses = 0
        documents_inspected = 0
        extracted_guidance_records = 0
        document_audits: list[GuidanceDocumentAudit] = []
        for filing in filings:
            if documents_inspected >= self.max_documents:
                break
            populated = await self._edgar.get_filing_documents(
                filing, use_cache=not refresh_sec, make_cache=True
            )
            documents = self._selector.select_documents(
                populated, limit=self.max_documents_per_filing
            )
            if not documents and any(
                document.is_pdf for document in populated.documents
            ):
                warnings.append(
                    f"{filing.form} {filing.accession_number} has no selected "
                    "HTML/text guidance exhibit; PDF extraction is unsupported"
                )
            for document in documents:
                if documents_inspected >= self.max_documents:
                    break
                documents_inspected += 1
                clean_text = clean_document_text(document.content)
                context_text = extract_guidance_context(clean_text)
                document_audit = GuidanceDocumentAudit(
                    filing_form=filing.form,
                    filing_date=filing.filing_date,
                    accession_number=filing.accession_number,
                    filename=document.filename,
                    document_type=document.document_type,
                    is_primary=self._selector.is_primary(filing, document),
                    cleaned_size=len(clean_text),
                    bounded_context_size=len(context_text),
                    keyword_hits=guidance_keyword_hits(clean_text),
                )
                if document.is_pdf:
                    warnings.append(
                        f"{filing.form} {filing.accession_number} document "
                        f"{document.filename} is PDF; PDF extraction is unsupported"
                    )
                    document_audits.append(document_audit)
                    continue
                if not clean_text:
                    document_audits.append(document_audit)
                    continue
                try:
                    entry, cache_hit = await self._extractor.extract(
                        filing,
                        document,
                        clean_text,
                        valuation_date=as_of,
                        source_text=clean_text,
                        context_text=context_text,
                    )
                except OpenAIAuthenticationError:
                    raise
                except Exception as exc:
                    warnings.append(
                        f"Guidance extraction skipped for {filing.accession_number} "
                        f"{document.filename}: {exc}"
                    )
                    document_audits.append(document_audit)
                    continue
                hits += int(cache_hit)
                misses += int(not cache_hit)
                records.extend(
                    record.model_copy(update={"is_primary": document_audit.is_primary})
                    for record in entry.accepted
                )
                rejected.extend(entry.rejected)
                extracted_guidance_records += len(entry.accepted)
                document_audits.append(
                    document_audit.model_copy(
                        update={
                            "accepted_records": len(entry.accepted),
                            "rejected_records": len(entry.rejected),
                        }
                    )
                )
        resolved = ManagementGuidanceResolver().resolve(records, as_of=as_of)
        warnings.extend(resolved.warnings)
        return GuidanceDiscoveryResult(
            records=resolved.records,
            rejected=tuple(rejected),
            warnings=tuple(dict.fromkeys(warnings)),
            cache_hits=hits,
            cache_misses=misses,
            filings_inspected=filings_inspected,
            documents_inspected=documents_inspected,
            extracted_guidance_records=extracted_guidance_records,
            rejected_records=len(rejected),
            document_audits=tuple(document_audits),
        )
