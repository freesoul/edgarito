from __future__ import annotations

import datetime
import hashlib

from pydantic import BaseModel, ConfigDict, Field


class SecFilingDocument(BaseModel):
    """One document block from an immutable SEC full submission."""

    model_config = ConfigDict(frozen=True)

    filename: str
    document_type: str
    description: str = ""
    sequence: str | None = None
    content: str
    # Set by the guidance selector when the document is matched to its filing
    # metadata.  Raw SEC document blocks do not carry this relationship.
    is_primary: bool = False

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def is_pdf(self) -> bool:
        return self.filename.casefold().endswith(".pdf")


class SecFiling(BaseModel):
    """Provider-neutral SEC filing metadata and retrieved documents."""

    model_config = ConfigDict(frozen=True)

    cik: int
    accession_number: str
    form: str
    filing_date: datetime.date
    acceptance_datetime: datetime.datetime | None = None
    report_date: datetime.date | None = None
    items: tuple[str, ...] = ()
    primary_document: str
    primary_document_description: str = ""
    documents: tuple[SecFilingDocument, ...] = Field(default_factory=tuple)

    @property
    def accession_without_dashes(self) -> str:
        return self.accession_number.replace("-", "")

    @property
    def archive_url(self) -> str:
        return (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{self.cik}/{self.accession_without_dashes}/"
            f"{self.accession_number}.txt"
        )
