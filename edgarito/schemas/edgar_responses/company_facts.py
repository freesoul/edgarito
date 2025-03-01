import datetime
from typing import List, Dict, Optional, Union

from pydantic import BaseModel, Field

from edgarito.enums.edgar.filing_type import FilingType


class Measurement(BaseModel):
    end: datetime.date
    val: Union[int, float]
    accn: str
    fy: int
    fp: str
    form: str
    filed: datetime.date
    frame: Optional[str] = None
    start: Optional[datetime.date] = None

    @property
    def parsed_type(self) -> Optional[FilingType]:
        try:
            return FilingType(self.form)
        except ValueError:
            return None


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
    us_gaap: Dict[str, Fact] = Field(..., alias="us-gaap")


class CompanyFacts(BaseModel):
    cik: int
    entityName: str
    facts: Facts
