import datetime
from typing import List, Dict, Optional, Union

from pydantic import BaseModel, Field


class Measurement(BaseModel):
    end: datetime.date
    val: Union[int, float]
    accn: str
    fy: int
    fp: str
    form: str
    filed: datetime.date
    frame: Optional[str] = None


class GaapMeasurement(BaseModel):
    start: datetime.date


class Fact(BaseModel):
    label: Optional[str] = None
    description: Optional[str] = None
    units: Dict[str, List[Measurement]]


class GaapDataUnits(BaseModel):
    USD: Optional[List[GaapMeasurement]] = None
    shares: Optional[List[GaapMeasurement]] = None


class Facts(BaseModel):
    dei: Dict[str, Fact]
    us_gaap: Dict[str, Fact] = Field(..., alias="us-gaap")


class CompanyFacts(BaseModel):
    cik: int
    entityName: str
    facts: Facts
