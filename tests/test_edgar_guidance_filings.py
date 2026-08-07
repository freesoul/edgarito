import asyncio
import datetime
from types import SimpleNamespace

from edgarito.schemas.providers.edgar.filing import SecFiling
from edgarito.services.cache.filesystem_cache import FileSystemCache
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
        acceptanceDateTime=datetime.datetime.combine(
            filed, datetime.time(12, 0)
        ),
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

    assert [item.form for item in filings] == ["6-K/A", "8-K/A", "8-K"]
    assert all(item.filing_date <= datetime.date(2026, 8, 1) for item in filings)
