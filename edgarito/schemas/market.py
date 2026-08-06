import datetime
import re
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edgarito.schemas.identifiers import SecurityIdentifiers

_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")


class MarketDataFrequency(str, Enum):
    SNAPSHOT = "snapshot"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class ReferenceSeriesKind(str, Enum):
    GOVERNMENT_YIELD = "government_yield"
    POLICY_RATE = "policy_rate"
    INFLATION_RATE = "inflation_rate"
    REAL_GDP_GROWTH = "real_gdp_growth"
    NOMINAL_GDP_GROWTH = "nominal_gdp_growth"
    EXCHANGE_RATE = "exchange_rate"
    OTHER = "other"


class ReferenceValueUnit(str, Enum):
    PERCENTAGE_POINTS = "percentage_points"
    PERCENT_CHANGE = "percent_change"
    INDEX_POINTS = "index_points"
    DECIMAL = "decimal"
    CURRENCY_PER_CURRENCY = "currency_per_currency"


class PriceBar(BaseModel):
    """One provider-normalized end-of-period security price observation."""

    model_config = ConfigDict(frozen=True)

    observed_on: datetime.date
    close: Decimal
    adjusted_close: Optional[Decimal] = None
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    volume: Optional[int] = Field(default=None, ge=0)

    @field_validator("close", "adjusted_close", "open", "high", "low")
    @classmethod
    def validate_price(cls, value: Optional[Decimal]) -> Optional[Decimal]:
        if value is not None and (not value.is_finite() or value <= 0):
            raise ValueError("Prices must be finite and greater than zero")
        return value

    @model_validator(mode="after")
    def validate_range(self) -> "PriceBar":
        comparable = [
            value for value in (self.open, self.close, self.low) if value is not None
        ]
        if self.high is not None and comparable and self.high < max(comparable):
            raise ValueError("High price cannot be below open, close, or low")
        comparable = [
            value for value in (self.open, self.close, self.high) if value is not None
        ]
        if self.low is not None and comparable and self.low > min(comparable):
            raise ValueError("Low price cannot be above open, close, or high")
        return self


class CashDividend(BaseModel):
    model_config = ConfigDict(frozen=True)

    ex_date: datetime.date
    amount: Decimal
    currency: str
    declaration_date: Optional[datetime.date] = None
    record_date: Optional[datetime.date] = None
    payment_date: Optional[datetime.date] = None

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value: Decimal) -> Decimal:
        if not value.is_finite() or value <= 0:
            raise ValueError("Dividend amount must be finite and greater than zero")
        return value

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return _normalize_currency(value)


class StockSplit(BaseModel):
    model_config = ConfigDict(frozen=True)

    effective_date: datetime.date
    from_shares: Decimal = Field(default=Decimal(1))
    to_shares: Decimal

    @field_validator("from_shares", "to_shares")
    @classmethod
    def validate_shares(cls, value: Decimal) -> Decimal:
        if not value.is_finite() or value <= 0:
            raise ValueError("Stock split shares must be finite and greater than zero")
        return value

    @property
    def factor(self) -> Decimal:
        return self.to_shares / self.from_shares


class SecurityMarketData(BaseModel):
    """Timestamped prices and corporate actions for one listed security."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    identifiers: SecurityIdentifiers
    currency: str
    exchange: Optional[str] = None
    frequency: MarketDataFrequency = MarketDataFrequency.DAILY
    retrieved_at: datetime.datetime
    source_version: Optional[str] = None
    prices: tuple[PriceBar, ...] = ()
    dividends: tuple[CashDividend, ...] = ()
    splits: tuple[StockSplit, ...] = ()

    @field_validator("provider", "provider_symbol", "exchange", "source_version")
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Text fields cannot be blank")
        return normalized

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return _normalize_currency(value)

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime.datetime) -> datetime.datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_payload(self) -> "SecurityMarketData":
        if not (self.prices or self.dividends or self.splits):
            raise ValueError("Market data must contain a price or corporate action")
        price_dates = [price.observed_on for price in self.prices]
        if len(price_dates) != len(set(price_dates)):
            raise ValueError("Price observations cannot repeat a date")
        return self

    @property
    def latest_price(self) -> Optional[PriceBar]:
        return max(self.prices, key=lambda price: price.observed_on, default=None)


class ReferenceObservation(BaseModel):
    """One observation in an economic or market reference series."""

    model_config = ConfigDict(frozen=True)

    period_end: datetime.date
    value: Decimal
    available_on: Optional[datetime.date] = None

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("Reference observations must be finite")
        return value


class ReferenceMarketSeries(BaseModel):
    """Provider-neutral series for Treasury, FRED, ECB, or similar data."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1)
    series_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: ReferenceSeriesKind
    unit: ReferenceValueUnit
    frequency: MarketDataFrequency
    retrieved_at: datetime.datetime
    observations: tuple[ReferenceObservation, ...]
    currency: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    tenor_months: Optional[int] = Field(default=None, ge=1)
    source_version: Optional[str] = None

    @field_validator("provider", "series_id", "name", "source_version")
    @classmethod
    def normalize_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Text fields cannot be blank")
        return normalized

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_currency(value) if value is not None else None

    @field_validator("country", "region")
    @classmethod
    def normalize_geography(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("Geography cannot be blank")
        return normalized

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime.datetime) -> datetime.datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_observations(self) -> "ReferenceMarketSeries":
        if not self.observations:
            raise ValueError("Reference series must contain at least one observation")
        periods = [observation.period_end for observation in self.observations]
        if len(periods) != len(set(periods)):
            raise ValueError("Reference observations cannot repeat a period")
        return self

    @property
    def latest_observation(self) -> ReferenceObservation:
        return max(self.observations, key=lambda observation: observation.period_end)


def _normalize_currency(value: str) -> str:
    normalized = value.strip().upper()
    if not _CURRENCY_PATTERN.fullmatch(normalized):
        raise ValueError("Currency must be a three-letter ISO code")
    return normalized


__all__ = [
    "CashDividend",
    "MarketDataFrequency",
    "PriceBar",
    "ReferenceMarketSeries",
    "ReferenceObservation",
    "ReferenceSeriesKind",
    "ReferenceValueUnit",
    "SecurityMarketData",
    "StockSplit",
]
