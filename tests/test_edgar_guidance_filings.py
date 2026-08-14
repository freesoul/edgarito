import asyncio
import datetime
from types import SimpleNamespace

from edgarito.schemas.providers.edgar.filing import SecFiling, SecFilingDocument
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.guidance.documents import GuidanceDocumentSelector
from edgarito.services.providers.edgar import EdgarClient

SUBMISSION = """<SEC-DOCUMENT>
<DOCUMENT>
<TYPE>6-K
<SEQUENCE>1
<FILENAME>wrapper.htm
<DESCRIPTION>Report of foreign private issuer
<TEXT><html><body>Quarterly results are attached.</body></html></TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<SEQUENCE>2
<FILENAME>ex991.htm
<DESCRIPTION>Financial results press release
<TEXT><html><body>We expect fiscal-year revenue of $10 billion.</body></html></TEXT>
</DOCUMENT>
</SEC-DOCUMENT>"""


class _Response:
    status = 200

    def __init__(self, content):
        self.content = content

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def text(self, **kwargs):
        return self.content


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return self.responses.pop(0)


def _filing() -> SecFiling:
    return SecFiling(
        cik=1234,
        accession_number="0000001234-26-000007",
        form="6-K",
        filing_date=datetime.date(2026, 7, 15),
        primary_document="wrapper.htm",
    )


def test_archive_url_and_sgml_document_metadata():
    filing = _filing()
    documents = EdgarClient.parse_submission_documents(SUBMISSION)

    assert filing.archive_url == (
        "https://www.sec.gov/Archives/edgar/data/1234/"
        "000000123426000007/0000001234-26-000007.txt"
    )
    assert [item.document_type for item in documents] == ["6-K", "EX-99.1"]
    assert documents[1].filename == "ex991.htm"
    assert documents[1].description == "Financial results press release"
    assert "expect fiscal-year revenue" in documents[1].content


def test_full_submission_and_parsed_documents_are_reused_from_cache(tmp_path):
    session = _Session([_Response(SUBMISSION)])
    client = EdgarClient(FileSystemCache(tmp_path), "Tester test@example.com", session)

    first = asyncio.run(client.get_filing_documents(_filing()))
    second = asyncio.run(client.get_filing_documents(_filing()))

    assert first == second
    assert len(first.documents) == 2
    assert session.calls == [_filing().archive_url]
    cached = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert {"full-submission.txt", "documents.json"} <= cached


class _MetadataClient(EdgarClient):
    def __init__(self, rows, tmp_path):
        super().__init__(
            FileSystemCache(tmp_path), "Tester test@example.com", _Session([])
        )
        self.rows = rows

    async def get_submissions(self, cik, use_cache=True, make_cache=True):
        recent = SimpleNamespace(transpose=lambda: self.rows)
        return SimpleNamespace(filings=SimpleNamespace(recent=recent))


def _row(form, filed, accession):
    return SimpleNamespace(
        accessionNumber=accession,
        filingDate=filed,
        acceptanceDateTime=datetime.datetime.combine(filed, datetime.time(12, 0)),
        reportDate=filed,
        form=form,
        items="2.02,7.01",
        primaryDocument="primary.htm",
        primaryDocDescription="Results",
    )


def test_form_filter_and_as_of_cutoff_include_amendments(tmp_path):
    client = _MetadataClient(
        [
            _row("8-K", datetime.date(2026, 6, 1), "a"),
            _row("8-K/A", datetime.date(2026, 6, 2), "b"),
            _row("6-K/A", datetime.date(2026, 6, 3), "c"),
            _row("10-Q", datetime.date(2026, 6, 4), "d"),
            _row("6-K", datetime.date(2026, 8, 2), "future"),
        ],
        tmp_path,
    )

    filings = asyncio.run(
        client.get_guidance_filings(
            1, as_of=datetime.date(2026, 8, 1), lookback_days=120
        )
    )

    assert [item.form for item in filings] == ["10-Q", "6-K/A", "8-K/A", "8-K"]
    assert all(item.filing_date <= datetime.date(2026, 8, 1) for item in filings)


def test_guidance_forms_include_periodic_reports():
    assert EdgarClient.GUIDANCE_FORMS == frozenset(
        {"8-K", "8-K/A", "6-K", "6-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A"}
    )


def test_filing_ranking_keeps_current_reports_above_itemless_periodic_reports():
    selector = GuidanceDocumentSelector()
    current = _filing().model_copy(
        update={
            "form": "8-K",
            "items": (),
            "accession_number": "0000001234-26-000008",
        }
    )
    foreign_current = _filing().model_copy(
        update={
            "form": "6-K/A",
            "items": (),
            "accession_number": "0000001234-26-000009",
        }
    )
    periodic = _filing().model_copy(
        update={
            "form": "10-Q/A",
            "items": (),
            "filing_date": datetime.date(2026, 7, 15),
            "accession_number": "0000001234-26-000010",
        }
    )
    annual = _filing().model_copy(
        update={
            "form": "10-K",
            "items": (),
            "filing_date": datetime.date(2026, 7, 14),
            "accession_number": "0000001234-26-000011",
        }
    )

    assert selector._filing_score(current) > selector._filing_score(periodic)
    assert selector._filing_score(foreign_current) > selector._filing_score(annual)
    assert selector._filing_score(periodic) > 0
    assert [item.form for item in selector.select_filings([annual, periodic])] == [
        "10-Q/A",
        "10-K",
    ]


def test_latest_periodic_report_cannot_be_crowded_out():
    selector = GuidanceDocumentSelector()
    current_reports = [
        _filing().model_copy(
            update={
                "form": "8-K",
                "filing_date": datetime.date(2026, 7, day),
                "accession_number": f"0000001234-26-00000{day}",
            }
        )
        for day in range(11, 15)
    ]
    latest_periodic = _filing().model_copy(
        update={
            "form": "10-Q",
            "filing_date": datetime.date(2026, 7, 15),
            "accession_number": "0000001234-26-000015",
            "items": (),
            "primary_document": "10q.htm",
            "primary_document_description": "Quarterly report",
        }
    )

    selected = selector.select_filings(current_reports + [latest_periodic])

    assert latest_periodic in selected
    assert len(selected) == 4
    assert [item.form for item in selected] == ["8-K", "8-K", "8-K", "10-Q"]


def test_latest_periodic_report_wins_periodic_quota():
    selector = GuidanceDocumentSelector()
    older_periodic = _filing().model_copy(
        update={
            "form": "10-K",
            "filing_date": datetime.date(2026, 6, 30),
            "accession_number": "0000001234-26-000016",
            "primary_document_description": "Annual guidance outlook revenue",
        }
    )
    latest_periodic = _filing().model_copy(
        update={
            "form": "10-Q",
            "filing_date": datetime.date(2026, 7, 15),
            "accession_number": "0000001234-26-000017",
            "items": (),
        }
    )

    assert selector.select_filings([older_periodic, latest_periodic], limit=1) == [
        latest_periodic
    ]


def test_filing_selection_deduplicates_accessions():
    selector = GuidanceDocumentSelector()
    filing = _filing().model_copy(update={"form": "8-K"})

    assert selector.select_filings([filing, filing]) == [filing]


def test_periodic_primary_document_is_selected_before_higher_ranked_exhibits():
    primary = SecFilingDocument(
        filename="primary.htm",
        document_type="10-Q",
        description="Quarterly report",
        content="Selected primary report.",
    )
    exhibits = tuple(
        SecFilingDocument(
            filename=f"ex99.{index}.htm",
            document_type=f"EX-99.{index}",
            description="Earnings outlook guidance revenue margin capex",
            content="We expect revenue and margin guidance.",
            sequence=str(index + 1),
        )
        for index in range(1, 4)
    )
    filing = _filing().model_copy(
        update={
            "form": "10-Q",
            "primary_document": "primary.htm",
            "documents": (primary, *exhibits),
        }
    )

    selected = GuidanceDocumentSelector().select_documents(filing, limit=2)

    assert [document.filename for document in selected] == [
        "primary.htm",
        "ex99.1.htm",
    ]
    assert selected[0].is_primary
    assert not selected[1].is_primary


def test_operating_selector_prefers_explicit_8k_exhibits_and_exposes_candidates():
    wrapper = SecFilingDocument(
        filename="wrapper.htm",
        document_type="8-K",
        description="Current report",
        content="The attached release is incorporated by reference.",
    )
    exhibit = SecFilingDocument(
        filename="release.htm",
        document_type="EX-99.1",
        description="Quarterly deliveries and revenue table",
        content="Q2 2026 deliveries 20 and revenue 100.",
    )
    filing = _filing().model_copy(
        update={
            "form": "8-K",
            "primary_document": "wrapper.htm",
            "documents": (wrapper, exhibit),
        }
    )

    selected = GuidanceDocumentSelector().select_operating_documents(filing, limit=1)

    assert selected[0].document_type == "EX-99.1"
    assert GuidanceDocumentSelector().select_exhibit_documents(filing) == [exhibit]
