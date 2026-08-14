import re
from typing import Optional

from edgarito.config.classification import CLASSIFICATION
from edgarito.schemas.identifiers import SecurityIdentifiers
from edgarito.schemas.normalization.classification import (
    NormalizedCompanyClassification,
    Sector,
)
from edgarito.schemas.providers.alphavantage.fundamentals import CompanyOverview
from edgarito.schemas.providers.fmp.fundamentals import CompanyProfile
from edgarito.schemas.providers.yahoo.fundamentals import YahooCompanyFinancials

SECTOR_ALIASES = CLASSIFICATION.sector_aliases


class CompanyClassificationNormalizer:
    def normalize_alphavantage(
        self, overview: CompanyOverview
    ) -> NormalizedCompanyClassification:
        return self._normalize(
            provider="alphavantage",
            symbol=overview.symbol,
            company_name=overview.name,
            cik=overview.cik,
            sector=overview.sector,
            industry=overview.industry,
            country=overview.country,
            exchange=overview.exchange,
        )

    def normalize_fmp(self, profile: CompanyProfile) -> NormalizedCompanyClassification:
        return self._normalize(
            provider="fmp",
            symbol=profile.symbol,
            company_name=profile.company_name,
            cik=profile.cik,
            sector=profile.sector,
            industry=profile.industry,
            country=profile.country,
            exchange=profile.exchange,
        )

    def normalize_yahoo(
        self, financials: YahooCompanyFinancials
    ) -> NormalizedCompanyClassification:
        return self._normalize(
            provider="yahoo",
            symbol=financials.symbol,
            company_name=financials.company_name,
            cik=None,
            sector=financials.sector,
            industry=financials.industry,
            country=financials.country,
            exchange=financials.exchange,
            identifiers=financials.identifiers,
        )

    def _normalize(
        self,
        *,
        provider: str,
        symbol: str,
        company_name: str,
        cik: Optional[str],
        sector: Optional[str],
        industry: Optional[str],
        country: Optional[str],
        exchange: Optional[str],
        identifiers: Optional[SecurityIdentifiers] = None,
    ) -> NormalizedCompanyClassification:
        source_sector = self._clean(sector)
        source_industry = self._clean(industry)
        return NormalizedCompanyClassification(
            provider=provider,
            company_id=self._company_id(cik, symbol),
            company_name=company_name,
            ticker=symbol.upper(),
            identifiers=identifiers,
            sector=self._sector(source_sector),
            industry=self._display_label(source_industry),
            source_sector=source_sector,
            source_industry=source_industry,
            industry_taxonomy=f"{provider}-profile",
            country=self._clean(country),
            exchange=self._clean(exchange),
        )

    @staticmethod
    def _sector(value: Optional[str]) -> Optional[Sector]:
        return SECTOR_ALIASES.get(value.casefold()) if value else None

    @staticmethod
    def _clean(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        cleaned = re.sub(r"\s+", " ", value).strip()
        return cleaned or None

    @staticmethod
    def _display_label(value: Optional[str]) -> Optional[str]:
        if value and value == value.upper():
            return value.title()
        return value

    @staticmethod
    def _company_id(cik: Optional[str], symbol: str) -> str:
        if cik and cik.isdigit():
            return cik.zfill(10)
        return cik or symbol.upper()
