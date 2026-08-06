from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Sector(str, Enum):
    COMMUNICATION_SERVICES = "Communication Services"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    ENERGY = "Energy"
    FINANCIALS = "Financials"
    HEALTHCARE = "Healthcare"
    INDUSTRIALS = "Industrials"
    TECHNOLOGY = "Technology"
    MATERIALS = "Materials"
    REAL_ESTATE = "Real Estate"
    UTILITIES = "Utilities"


class NormalizedCompanyClassification(BaseModel):
    provider: str
    company_id: str
    company_name: str
    ticker: str
    sector: Optional[Sector] = None
    industry: Optional[str] = None
    source_sector: Optional[str] = None
    source_industry: Optional[str] = None
    sector_taxonomy: str = "edgarito-sector-v1"
    industry_taxonomy: str
    country: Optional[str] = None
    exchange: Optional[str] = None
