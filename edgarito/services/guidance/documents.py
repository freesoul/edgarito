from __future__ import annotations

import html
import re
from bisect import bisect_left
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
    "expect",
    "expected",
    "expecting",
    "capital expenditures",
    "capex",
    "revenue",
    "margin",
    "capacity",
    "forecasting",
    "target",
    # Operating-driver evidence uses the same bounded SEC context seam as
    # management guidance.  These terms improve recall without interpreting
    # any value or creating a forecast.
    "volume",
    "price",
    "subscriber",
    "users",
    "arpu",
    "average revenue per user",
    "utilization",
    "transactions",
    "take rate",
    "store count",
    "sales per store",
    "production",
    "shipments",
    "deliveries",
    "investment",
    "facility",
    "data center",
)

GUIDANCE_CONTEXT_MAX_CHARS = 24_000
GUIDANCE_CONTEXT_WINDOW_CHARS = 900

_CURRENT_REPORT_FORMS = frozenset({"8-K", "8-K/A", "6-K", "6-K/A"})
_PERIODIC_REPORT_FORMS = frozenset({"10-Q", "10-Q/A", "10-K", "10-K/A"})
_CURRENT_REPORT_QUOTA = 3
_PERIODIC_REPORT_QUOTA = 2


def is_periodic_filing(filing: SecFiling) -> bool:
    """Return whether a filing is a periodic report requiring primary priority."""

    return filing.form.upper() in _PERIODIC_REPORT_FORMS

# Keep the context vocabulary broader than the filing metadata vocabulary.  A
# primary 10-Q/10-K often has no useful item metadata, so its guidance has to be
# found in the cleaned filing text.
_GUIDANCE_CONTEXT_PATTERNS = (
    (r"\bexpect(?:s|ed|ing)?\b", 10),
    (r"\bguidance\b", 9),
    (r"\boutlook\b", 8),
    (r"\bforecast(?:s|ed|ing)?\b", 8),
    (r"\banticipat(?:e|es|ed|ing)\b", 7),
    (r"\btarget(?:s|ed|ing)?\b", 7),
    (r"\bproject(?:s|ed|ing)?\b", 6),
    (r"\bplan(?:s|ned|ning)?\b", 5),
    (r"\bestimat(?:e|es|ed|ing)\b", 5),
    (r"\bcapital expenditure(?:s)?\b", 6),
    (r"\bcapex\b", 6),
    (r"\brevenue\b", 3),
    (r"\bmargin(?:s)?\b", 3),
    (r"\bcapacity\b", 3),
    (r"\bearnings\b", 2),
    (r"\bfree cash flow\b", 2),
    (r"\boperating cash flow\b", 2),
    (r"\bbookings?\b", 2),
    (r"\bbacklog\b", 2),
    (r"\bvolumes?\b", 4),
    (r"\bprices?\b", 3),
    (r"\bsubscribers?\b", 5),
    (r"\busers?\b", 4),
    (r"\barpu\b", 6),
    (r"\baverage revenue per user\b", 6),
    (r"\butili[sz]ation\b", 5),
    (r"\btransactions?\b", 4),
    (r"\btake rate\b", 5),
    (r"\bstore count\b", 4),
    (r"\bsales per store\b", 5),
    (r"\bproduction\b", 4),
    (r"\bshipments?\b", 4),
    (r"\bdeliveries\b", 4),
    (r"\binvestments?\b", 3),
    (r"\bfacilit(?:y|ies)\b", 3),
    (r"\bdata cent(?:er|re)s?\b", 3),
    (r"\beps\b", 2),
)

_GUIDANCE_AUDIT_PATTERNS = {
    "expect": r"\bexpect(?:s|ed|ing)?\b",
    "capex": r"\bcapex\b",
    "capital expenditures": r"\bcapital expenditures\b",
    "revenue": r"\brevenues?\b",
    "margin": r"\bmargins?\b",
}


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


def guidance_keyword_hits(text: str) -> dict[str, int]:
    """Count the bounded set of guidance terms shown in audit output."""

    return {
        keyword: len(re.findall(pattern, text, flags=re.IGNORECASE))
        for keyword, pattern in _GUIDANCE_AUDIT_PATTERNS.items()
    }


def extract_guidance_context(
    clean_text: str,
    *,
    max_chars: int = GUIDANCE_CONTEXT_MAX_CHARS,
    window_chars: int = GUIDANCE_CONTEXT_WINDOW_CHARS,
) -> str:
    """Return deterministic, bounded text windows around guidance vocabulary.

    Every selected window is a direct slice of ``clean_text``.  Separators only
    mark omitted regions; they never rewrite the source text that the model can
    quote for evidence.  Higher-signal forward-looking terms win when a long
    document contains more windows than the hard context budget permits.
    """
    if max_chars <= 0 or not clean_text:
        return ""
    if len(clean_text) <= max_chars:
        return clean_text

    matches: list[tuple[int, int, int]] = []
    for pattern, weight in _GUIDANCE_CONTEXT_PATTERNS:
        matches.extend(
            (match.start(), match.end(), weight)
            for match in re.finditer(pattern, clean_text, flags=re.IGNORECASE)
        )
    if not matches:
        return clean_text[:max_chars]

    matches.sort(key=lambda item: (item[0], item[1], -item[2]))
    match_starts = [item[0] for item in matches]
    candidates: list[tuple[int, int, int]] = []
    for start, end, _ in matches:
        window_start = max(0, start - window_chars)
        window_end = min(len(clean_text), end + window_chars)
        first_match = bisect_left(match_starts, window_start)
        last_match = bisect_left(match_starts, window_end)
        window_matches = matches[first_match:last_match]
        priority = sum(weight for _, _, weight in window_matches)
        forward_matches = sum(weight >= 5 for _, _, weight in window_matches)
        # Numerical context is more useful than a bare mention of a metric and
        # helps retain concrete management guidance when the cap is reached.
        number_count = len(
            re.findall(
                r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?",
                clean_text[window_start:window_end],
            )
        )
        candidates.append(
            (
                window_start,
                window_end,
                forward_matches * 100 + priority + min(number_count, 3),
            )
        )

    selected: list[tuple[int, int]] = []
    used = 0
    separator = "\n\n[... omitted ...]\n\n"
    for start, end, _ in sorted(
        candidates, key=lambda item: (-item[2], item[0], item[1])
    ):
        if any(
            start < other_end and end > other_start
            for other_start, other_end in selected
        ):
            continue
        separator_size = len(separator) if selected else 0
        if used + separator_size + end - start > max_chars:
            continue
        selected.append((start, end))
        used += separator_size + end - start

    if not selected:
        # A single window is intentionally much smaller than the hard cap, but
        # retain a useful source slice even for unusually small custom caps.
        start, end, _ = candidates[0]
        center = (start + end) // 2
        window_start = max(0, min(center - max_chars // 2, len(clean_text) - max_chars))
        return clean_text[window_start : window_start + max_chars]

    selected.sort()
    return separator.join(clean_text[start:end] for start, end in selected)


class GuidanceDocumentSelector:
    """Conservative deterministic gate before any model request."""

    def select_filings(
        self, filings: list[SecFiling], *, limit: int = 4
    ) -> list[SecFiling]:
        if limit <= 0:
            return []

        ranked = self._rank_unique_filings(filings)
        current = [
            filing for filing in ranked if filing.form.upper() in _CURRENT_REPORT_FORMS
        ]
        periodic = [
            filing for filing in ranked if filing.form.upper() in _PERIODIC_REPORT_FORMS
        ]

        # Keep the service's existing total filing limit while reserving one
        # slot for a periodic report whenever an eligible one is available.
        # This leaves the default allocation at three current reports and one
        # periodic report (with up to two periodic reports when capacity
        # remains), so a fourth 8-K cannot crowd out the latest 10-Q/10-K.
        current_limit = min(
            _CURRENT_REPORT_QUOTA,
            len(current),
            limit - int(bool(periodic)),
        )
        periodic_limit = min(
            _PERIODIC_REPORT_QUOTA,
            len(periodic),
            limit - current_limit,
        )

        selected = current[:current_limit]
        selected.extend(self._select_periodic(periodic, periodic_limit))

        # The category form sets are disjoint, but retain an explicit final
        # identity-based deduplication guard for callers supplying repeated
        # metadata rows.
        unique: list[SecFiling] = []
        seen: set[tuple[int, str]] = set()
        for filing in selected:
            identity = (filing.cik, filing.accession_number)
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(filing)
        return unique[:limit]

    def _rank_unique_filings(self, filings: list[SecFiling]) -> list[SecFiling]:
        ranked = sorted(
            (
                (self._filing_score(filing), filing)
                for filing in filings
                if filing.form.upper() in _CURRENT_REPORT_FORMS | _PERIODIC_REPORT_FORMS
            ),
            key=self._filing_sort_key,
            reverse=True,
        )
        unique: list[SecFiling] = []
        seen: set[tuple[int, str]] = set()
        for score, filing in ranked:
            if score <= 0:
                continue
            identity = (filing.cik, filing.accession_number)
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(filing)
        return unique

    @staticmethod
    def _filing_sort_key(pair: tuple[int, SecFiling]) -> tuple:
        score, filing = pair
        return (
            score,
            filing.filing_date,
            (
                filing.acceptance_datetime.isoformat()
                if filing.acceptance_datetime is not None
                else ""
            ),
            filing.accession_number,
            filing.primary_document,
        )

    def _select_periodic(
        self, periodic: list[SecFiling], limit: int
    ) -> list[SecFiling]:
        if limit <= 0:
            return []

        selected = periodic[:limit]
        latest = max(periodic, key=self._filing_recency_key)
        selected_ids = {(filing.cik, filing.accession_number) for filing in selected}
        latest_id = (latest.cik, latest.accession_number)
        if latest_id not in selected_ids:
            selected[-1] = latest
            selected.sort(
                key=lambda filing: self._filing_sort_key(
                    (self._filing_score(filing), filing)
                ),
                reverse=True,
            )
        return selected

    @staticmethod
    def _filing_recency_key(filing: SecFiling) -> tuple:
        return (
            filing.filing_date,
            (
                filing.acceptance_datetime.isoformat()
                if filing.acceptance_datetime is not None
                else ""
            ),
            filing.accession_number,
            filing.primary_document,
        )

    def select_documents(
        self, filing: SecFiling, *, limit: int = 3
    ) -> list[SecFilingDocument]:
        if limit <= 0:
            return []

        ranked = self._rank_documents(filing)
        if filing.form.upper() not in _PERIODIC_REPORT_FORMS:
            # Keep the existing 8-K/6-K ranking and eligibility behavior intact.
            return [document for score, document in ranked if score > 0][:limit]

        primary = next(
            (
                document
                for _score, document in ranked
                if self.is_primary(filing, document)
            ),
            None,
        )
        if primary is None:
            return [document for score, document in ranked if score > 0][:limit]

        # A periodic filing's primary report is the authoritative place for
        # forward-looking language.  It is selected even when its metadata and
        # text score are lower than an exhibit's score; only the remaining
        # slots use the normal exhibit ranking.
        selected = [self._mark_primary(primary)]
        selected.extend(
            self._mark_primary(document, value=False)
            for score, document in ranked
            if score > 0
            and not self.is_primary(filing, document)
        )
        return selected[:limit]

    def _rank_documents(
        self, filing: SecFiling
    ) -> list[tuple[int, SecFilingDocument]]:
        return sorted(
            (
                (self._document_score(filing, document), document)
                for document in filing.documents
            ),
            key=lambda pair: (pair[0], pair[1].sequence or "", pair[1].filename),
            reverse=True,
        )

    @staticmethod
    def is_primary(filing: SecFiling, document: SecFilingDocument) -> bool:
        """Return whether ``document`` is the filing metadata's primary file."""

        return document.filename.casefold() == filing.primary_document.casefold()

    @classmethod
    def _mark_primary(
        cls, document: SecFilingDocument, *, value: bool = True
    ) -> SecFilingDocument:
        # The provider document schema deliberately has no filing reference, so
        # expose the selection decision on the returned document without
        # changing the raw SEC document content.
        if document.is_primary == value:
            return document
        return document.model_copy(update={"is_primary": value})

    @staticmethod
    def _filing_score(filing: SecFiling) -> int:
        form = filing.form.upper()
        # Current reports are intentionally a strong category: their item
        # metadata is useful when present, but 8-K/6-K filings remain ahead of
        # periodic filings even when metadata is sparse.  Periodic filings get
        # only a small baseline so item-less 10-Q/10-K primary reports are
        # still inspected.
        score = 20 if form in {"8-K", "8-K/A", "6-K", "6-K/A"} else 0
        if form in {"10-Q", "10-Q/A", "10-K", "10-K/A"}:
            score += 1
        item_values = {item.casefold() for item in filing.items}
        if any(item.startswith("2.02") for item in item_values):
            score += 10
        if any(item.startswith("7.01") for item in item_values):
            score += 5
        metadata = " ".join(
            (filing.primary_document, filing.primary_document_description)
        ).casefold()
        score += sum(2 for term in GUIDANCE_TERMS if term in metadata)
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
