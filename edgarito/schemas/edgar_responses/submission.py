import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


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
    filingDate: List[datetime.datetime]
    acceptanceDateTime: List[datetime.datetime]
    act: List[str]
    form: List[str]
    fileNumber: List[str]
    filmNumber: List[str]
    items: List[str]
    core_type: List[str]
    size: List[int]
    isXBRL: List[bool]
    isInlineXBRL: List[bool]
    primaryDocument: List[str]
    primaryDocDescription: List[str]


class FilingFile(BaseModel):
    name: str
    filingCount: int
    filingFrom: datetime.datetime
    filingTo: datetime.datetime


class Filings(BaseModel):
    recent: FilingRecent
    files: List[FilingFile]


class FormerName(BaseModel):
    name: str
    from_: datetime.datetime = Field(..., alias="from")  # from is a reserved keyword in Python.
    to: datetime.datetime

    class Config:
        allow_population_by_field_name = True


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
