from typing import Mapping, Optional

from pydantic import BaseModel, model_validator

from edgarito.enums.granularity import Granularity
from edgarito.enums.provider import ProviderName
from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.normalization.financials import FinancialConcept


class RetrieveFinancialsRequest(BaseModel):
    ticker: Optional[str] = None
    cik: Optional[int] = None
    isin: Optional[str] = None
    exchange: Optional[str] = None
    exchange_symbols: Optional[Mapping[str, str]] = None
    provider_symbols: Optional[Mapping[ProviderName | str, str]] = None
    identifiers: Optional[SecurityIdentifiers] = None
    granularity: Optional[Granularity] = Granularity.ANNUAL
    concepts: Optional[set[FinancialConcept]] = None
    use_cache: bool = True
    make_cache: bool = True

    @model_validator(mode="after")
    def require_one_identifier(self):
        supplied_fields = any(
            (
                self.ticker,
                self.cik,
                self.isin,
                self.exchange,
                self.exchange_symbols,
                self.provider_symbols,
            )
        )
        if self.identifiers is not None and supplied_fields:
            raise ValueError(
                "Use identifiers or individual identifier arguments, not both"
            )
        self.security_identifiers()
        return self

    def security_identifiers(self) -> SecurityIdentifiers:
        return self.identifiers or SecurityIdentifiers(
            ticker=self.ticker,
            cik=self.cik,
            isin=self.isin,
            exchange=self.exchange,
            exchange_symbols=dict(self.exchange_symbols or {}),
            provider_symbols=dict(self.provider_symbols or {}),
        )
