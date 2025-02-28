import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from edgarito.schemas.edgar_responses.submission_custom import TransposedFiling


class Address(BaseModel):
    street1: str
    street2: Optional[str]
    city: str
    stateOrCountry: str
    zipCode: str
    stateOrCountryDescription: str


class Addresses(BaseModel):
    mailing: Address
    business: Address


class FilingRecent(BaseModel):
    accessionNumber: List[str]
    filingDate: List[datetime.date]
    acceptanceDateTime: List[datetime.datetime]
    act: List[str]
    form: List[str]
    fileNumber: List[str]
    filmNumber: List[str]
    items: List[str]
    core_type: List[Optional[str]]
    size: List[int]
    isXBRL: List[bool]
    isInlineXBRL: List[bool]
    primaryDocument: List[str]
    primaryDocDescription: List[str]
    reportDate: List[Optional[datetime.date]]

    @field_validator("isXBRL", "isInlineXBRL", mode="before")
    def parse_bool_list(cls, v):
        if isinstance(v, list):
            return [bool(item) for item in v]
        return v

    @field_validator("reportDate", mode="before")
    def empty_str_to_none(cls, v):
        if isinstance(v, list):
            return [None if isinstance(item, str) and item == "" else item for item in v]
        return v

    def extend_in_place(self, other: "FilingRecent") -> None:
        """
        Dynamically extends all list fields of this instance using the corresponding lists from another FilingRecent.
        """
        for field_name in self.model_fields.keys():
            self_val = getattr(self, field_name)
            other_val = getattr(other, field_name)
            if isinstance(self_val, list) and isinstance(other_val, list):
                self_val.extend(other_val)

    def transpose(self) -> List[TransposedFiling]:
        """
        Transposes the FilingRecent instance (with list fields) into a list
        of FilingRecentRow, where each row holds one element from each field.
        Assumes all list fields have the same length.
        """
        n = len(self.accessionNumber)  # Using one field to determine the number of rows.
        rows = []
        for i in range(n):
            row_data = {}
            for field_name in self.model_fields.keys():
                # Extract the i-th element for each field.
                row_data[field_name] = getattr(self, field_name)[i]
            rows.append(TransposedFiling(**row_data))
        return rows


class FilingFile(BaseModel):
    name: str
    filingCount: int
    filingFrom: datetime.date
    filingTo: datetime.date


class Filings(BaseModel):
    recent: FilingRecent
    files: List[FilingFile]


class FormerName(BaseModel):
    name: str
    from_: datetime.datetime = Field(..., alias="from")  # from is a reserved keyword in Python.
    to: datetime.datetime

    class Config:
        populate_by_name = True


class CompanySubmissionsResponse(BaseModel):
    cik: int
    entityType: str
    sic: int
    sicDescription: str
    ownerOrg: str
    insiderTransactionForOwnerExists: int
    insiderTransactionForIssuerExists: int
    name: str
    tickers: List[str]
    exchanges: List[str]
    ein: str
    description: str
    website: str
    investorWebsite: str
    category: str
    fiscalYearEnd: datetime.datetime
    stateOfIncorporation: str
    stateOfIncorporationDescription: str
    addresses: Addresses
    phone: str
    flags: str
    formerNames: List[FormerName]
    filings: Filings
