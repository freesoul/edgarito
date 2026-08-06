import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class YahooPriceRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    observed_on: datetime.date
    close: Optional[Decimal] = None
    adjusted_close: Optional[Decimal] = None
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    volume: Optional[int] = Field(default=None, ge=0)
    dividend: Optional[Decimal] = None
    split_factor: Optional[Decimal] = None


class YahooMarketHistory(BaseModel):
    """Serializable daily Yahoo price and corporate-action history."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    currency: str
    exchange: Optional[str] = None
    retrieved_at: datetime.datetime
    source_version: str
    rows: tuple[YahooPriceRow, ...]
