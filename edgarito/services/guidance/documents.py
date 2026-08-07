from __future__ import annotations

import html
import re
from html.parser import HTMLParser

from edgarito.schemas.providers.edgar.filing import SecFiling, SecFilingDocument

GUIDANCE_TERMS = (
    "earnings",
    "financial results",
    "quarter results",
    "annual results",
    "outlook",
    "guidance",
    "forecast",
    "expects",
    "full year",
    "fiscal year",
    "press release",
    "investor presentation",
)


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.casefold() in {"script", "style"}:
            self._ignored += 1
        elif tag.casefold() in {"p", "div", "br", "tr", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style"} and self._ignored:
            self._ignored -= 1
        elif tag.casefold() in {"p", "div", "tr", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            self.parts.append(data)


def clean_document_text(content: str) -> str:
    parser = _TextParser()
    try:
        parser.feed(content)
        text = " ".join(parser.parts)
    except (ValueError, AssertionError):
        text = re.sub(r"<[^>]+>", " ", content)
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", "\n", text)).strip()


def normalize_evidence(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text).replace("\xa0", " ")).strip()


class GuidanceDocumentSelector:
    """Conservative deterministic gate before any model request."""

    def select_filings(
        self, filings: list[SecFiling], *, limit: int = 4
    ) -> list[SecFiling]:
        ranked = sorted(
            ((self._filing_score(filing), filing) for filing in filings),
            key=lambda pair: (pair[0], pair[1].filing_date),
            reverse=True,
        )
        return [filing for score, filing in ranked if score > 0][:limit]

    def select_documents(
        self, filing: SecFiling, *, limit: int = 3
    ) -> list[SecFilingDocument]:
        ranked = sorted(
            ((self._document_score(filing, document), document) for document in filing.documents),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [document for score, document in ranked if score > 0][:limit]

    @staticmethod
    def _filing_score(filing: SecFiling) -> int:
        score = 0
        item_values = {item.casefold() for item in filing.items}
        if any(item.startswith("2.02") for item in item_values):
            score += 10
        if any(item.startswith("7.01") for item in item_values):
            score += 5
        metadata = " ".join(
            (filing.primary_document, filing.primary_document_description)
        ).casefold()
        score += sum(2 for term in GUIDANCE_TERMS if term in metadata)
        # 6-K has no item taxonomy; keep it eligible for document inspection.
        if filing.form.upper().startswith("6-K"):
            score += 1
        return score

    @staticmethod
    def _document_score(filing: SecFiling, document: SecFilingDocument) -> int:
        if document.is_pdf:
            return 0
        document_type = document.document_type.upper()
        metadata = f"{document.description} {document.filename}".casefold()
        score = sum(3 for term in GUIDANCE_TERMS if term in metadata)
        if re.fullmatch(r"EX-99(?:\.\d+)?", document_type):
            score += 8 if document_type == "EX-99.1" else 5
        if document.filename == filing.primary_document:
            score += 3
        # Content is only used for deterministic ranking, never as an inferred
        # financial assumption.
        sample = clean_document_text(document.content[:250_000]).casefold()
        score += min(8, sum(1 for term in GUIDANCE_TERMS if term in sample))
        if filing.form.upper().startswith("6-K") and score < 3:
            return 0
        return score
