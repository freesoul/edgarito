from enum import Enum


class ProviderName(str, Enum):
    SEC = "sec"
    ALPHAVANTAGE = "alphavantage"
    FMP = "fmp"
