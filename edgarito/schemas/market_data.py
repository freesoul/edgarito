"""
Schemas for market data from external sources (e.g., Yahoo Finance).
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MarketData(BaseModel):
    """Market data for a company's stock."""
    
    ticker: str = Field(..., description="Stock ticker symbol")
    market_cap: Optional[float] = Field(
        None, 
        description="Market capitalization in USD",
        ge=0
    )
    current_price: Optional[float] = Field(
        None,
        description="Current stock price in USD",
        ge=0
    )
    enterprise_value: Optional[float] = Field(
        None,
        description="Enterprise Value (Market Cap + Debt - Cash) in USD"
    )
    ev_to_ebitda: Optional[float] = Field(
        None,
        description="Enterprise Value / EBITDA ratio"
    )
    peg_ratio: Optional[float] = Field(
        None,
        description="Price/Earnings to Growth ratio"
    )
    short_percent_float: Optional[float] = Field(
        None,
        description="Percentage of float shares sold short",
        ge=0,
        le=100
    )
    insider_ownership_percent: Optional[float] = Field(
        None,
        description="Percentage of shares held by insiders",
        ge=0,
        le=100
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this data was retrieved"
    )
    source: str = Field(
        default="yahoo_finance",
        description="Data source identifier"
    )
    
    @field_validator('ticker')
    @classmethod
    def ticker_not_empty(cls, v: str) -> str:
        """Ensure ticker is not empty."""
        if not v or not v.strip():
            raise ValueError("Ticker symbol cannot be empty")
        return v.strip().upper()
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "ticker": "TSLA",
                "market_cap": 850000000000.0,
                "current_price": 250.50,
                "timestamp": "2025-01-15T10:30:00",
                "source": "yahoo_finance"
            }
        }
