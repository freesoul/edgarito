import datetime
from typing import List, Dict, Optional, Union

from pydantic import BaseModel, Field, field_validator

from edgarito.enums.edgar.core_filing_type import CoreFilingType
from edgarito.enums.edgar.period import FiscalPeriod


class Measurement(BaseModel):
    end: datetime.date
    val: Union[int, float]
    accn: str
    fy: Optional[int] = None  # SEC API sometimes returns null
    fp: Optional[FiscalPeriod] = None  # SEC API sometimes returns null
    form: str  # Raw form string from SEC API
    filed: datetime.date
    frame: Optional[str] = None
    start: Optional[datetime.date] = None
    
    @field_validator('fp', mode='before')
    @classmethod
    def convert_empty_string_to_none(cls, v):
        """Convert empty strings to None for optional fiscal period field."""
        if v == '':
            return None
        return v

    @property
    def calendar_year(self) -> Optional[int]:
        """
        Get the calendar year for this measurement.
        For period measurements (income statement, cash flow), use start date.
        For point-in-time measurements (balance sheet), use end date.
        """
        if self.start:
            return self.start.year
        # Balance sheet items don't have start date, use end date
        return self.end.year

    @property
    def parsed_type(self) -> Optional[CoreFilingType]:
        """
        Try to parse the form string into a CoreFilingType enum.
        Returns None if the form type is not in the core filing types.
        """
        return CoreFilingType.try_from_string(self.form)


class FactUnits(BaseModel):
    USD: Optional[List[Measurement]] = None
    shares: Optional[List[Measurement]] = None
    pure: Optional[List[Measurement]] = None

    class Config:
        extra = "allow"


class Fact(BaseModel):
    label: Optional[str] = None
    description: Optional[str] = None
    units: FactUnits


class Facts(BaseModel):
    dei: Dict[str, Fact]
    us_gaap: Optional[Dict[str, Fact]] = Field(None, alias="us-gaap")


class CompanyFacts(BaseModel):
    cik: int
    entityName: str
    facts: Facts
