import datetime
from decimal import Decimal
from typing import Optional

from pydantic import Field, model_validator

from edgarito.schemas.providers.alphavantage.fundamentals import AlphaVantageModel


class DailyTimeSeriesMetadata(AlphaVantageModel):
    information: str = Field(alias="1. Information")
    symbol: str = Field(alias="2. Symbol")
    last_refreshed: datetime.date = Field(alias="3. Last Refreshed")
    output_size: str = Field(alias="4. Output Size")
    timezone: str = Field(alias="5. Time Zone")


class DailyPrice(AlphaVantageModel):
    open: Decimal = Field(alias="1. open")
    high: Decimal = Field(alias="2. high")
    low: Decimal = Field(alias="3. low")
    close: Decimal = Field(alias="4. close")
    volume: int = Field(alias="5. volume", ge=0)


class DailyTimeSeriesResponse(AlphaVantageModel):
    metadata: DailyTimeSeriesMetadata = Field(alias="Meta Data")
    time_series: dict[datetime.date, DailyPrice] = Field(alias="Time Series (Daily)")

    @model_validator(mode="after")
    def require_prices(self) -> "DailyTimeSeriesResponse":
        if not self.time_series:
            raise ValueError("Alpha Vantage daily time series cannot be empty")
        return self


class GlobalQuote(AlphaVantageModel):
    symbol: str = Field(alias="01. symbol")
    open: Decimal = Field(alias="02. open")
    high: Decimal = Field(alias="03. high")
    low: Decimal = Field(alias="04. low")
    price: Decimal = Field(alias="05. price")
    volume: int = Field(alias="06. volume", ge=0)
    latest_trading_day: datetime.date = Field(alias="07. latest trading day")
    previous_close: Decimal = Field(alias="08. previous close")
    change: Decimal = Field(alias="09. change")
    change_percent: str = Field(alias="10. change percent")


class GlobalQuoteResponse(AlphaVantageModel):
    quote: GlobalQuote = Field(alias="Global Quote")


class DividendEvent(AlphaVantageModel):
    ex_dividend_date: datetime.date
    declaration_date: Optional[datetime.date] = None
    record_date: Optional[datetime.date] = None
    payment_date: Optional[datetime.date] = None
    amount: Decimal


class DividendResponse(AlphaVantageModel):
    symbol: str
    data: list[DividendEvent]


class SplitEvent(AlphaVantageModel):
    effective_date: datetime.date
    split_factor: Decimal


class SplitResponse(AlphaVantageModel):
    symbol: str
    data: list[SplitEvent]


__all__ = [
    "DailyPrice",
    "DailyTimeSeriesMetadata",
    "DailyTimeSeriesResponse",
    "DividendEvent",
    "DividendResponse",
    "GlobalQuote",
    "GlobalQuoteResponse",
    "SplitEvent",
    "SplitResponse",
]
