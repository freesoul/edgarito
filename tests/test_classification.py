import asyncio
import json

import pytest

import edgarito.cli.__main__ as cli_module
from edgarito.cli import main
from edgarito.config.providers import ClassificationProviderConfiguration
from edgarito.enums.provider import ProviderName
from edgarito.schemas.normalization.classification import (
    NormalizedCompanyClassification,
    Sector,
)
from edgarito.schemas.providers.alphavantage.fundamentals import CompanyOverview
from edgarito.schemas.providers.fmp.fundamentals import CompanyProfile
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization.classification import (
    CompanyClassificationNormalizer,
)
from edgarito.services.reconciliation.classification import (
    ClassificationCrosscheckWarning,
    CompanyClassificationService,
)

SECTOR_CASES = (
    ("AAPL", "Technology", "TECHNOLOGY", Sector.TECHNOLOGY),
    (
        "GOOGL",
        "Communication Services",
        "COMMUNICATION SERVICES",
        Sector.COMMUNICATION_SERVICES,
    ),
    (
        "AMZN",
        "Consumer Cyclical",
        "CONSUMER CYCLICAL",
        Sector.CONSUMER_DISCRETIONARY,
    ),
    (
        "KO",
        "Consumer Defensive",
        "CONSUMER DEFENSIVE",
        Sector.CONSUMER_STAPLES,
    ),
    ("JNJ", "Healthcare", "HEALTHCARE", Sector.HEALTHCARE),
    ("JPM", "Financial Services", "FINANCIAL SERVICES", Sector.FINANCIALS),
    ("CAT", "Industrials", "INDUSTRIALS", Sector.INDUSTRIALS),
    ("XOM", "Energy", "ENERGY", Sector.ENERGY),
    ("LIN", "Basic Materials", "BASIC MATERIALS", Sector.MATERIALS),
    ("NEE", "Utilities", "UTILITIES", Sector.UTILITIES),
    ("PLD", "Real Estate", "REAL ESTATE", Sector.REAL_ESTATE),
)


def _classification(provider: str, industry="Consumer Electronics"):
    return NormalizedCompanyClassification(
        provider=provider,
        company_id="0000320193",
        company_name="Apple Inc.",
        ticker="AAPL",
        sector=Sector.TECHNOLOGY,
        industry=industry,
        source_sector="Technology",
        source_industry=industry,
        industry_taxonomy=f"{provider}-profile",
    )


class _FakeProvider:
    def __init__(self, name, classification):
        self.name = name
        self.classification = classification

    async def retrieve(self, ticker, use_cache, make_cache):
        return self.classification


def test_normalizes_provider_profiles_to_a_common_sector():
    normalizer = CompanyClassificationNormalizer()
    alpha = normalizer.normalize_alphavantage(
        CompanyOverview(
            Symbol="AAPL",
            Name="Apple Inc.",
            CIK="320193",
            Sector="TECHNOLOGY",
            Industry="CONSUMER ELECTRONICS",
            Country="USA",
            Exchange="NASDAQ",
        )
    )
    fmp = normalizer.normalize_fmp(
        CompanyProfile(
            symbol="AAPL",
            companyName="Apple Inc.",
            cik="320193",
            sector="Technology",
            industry="Consumer Electronics",
            country="US",
            exchange="NASDAQ",
        )
    )

    assert alpha.sector == fmp.sector == Sector.TECHNOLOGY
    assert alpha.industry == fmp.industry == "Consumer Electronics"
    assert alpha.source_sector == "TECHNOLOGY"
    assert alpha.company_id == "0000320193"


@pytest.mark.parametrize(
    ("ticker", "fmp_sector", "alphavantage_sector", "expected"), SECTOR_CASES
)
def test_normalizes_all_supported_sectors_for_each_provider(
    ticker, fmp_sector, alphavantage_sector, expected
):
    normalizer = CompanyClassificationNormalizer()
    fmp = normalizer.normalize_fmp(
        CompanyProfile(symbol=ticker, companyName=ticker, sector=fmp_sector)
    )
    alphavantage = normalizer.normalize_alphavantage(
        CompanyOverview(Symbol=ticker, Name=ticker, Sector=alphavantage_sector)
    )

    assert fmp.sector == expected
    assert alphavantage.sector == expected


def test_classification_configuration_can_be_overridden():
    defaults = ClassificationProviderConfiguration.from_environment({})
    assert defaults.default_provider == ProviderName.FMP
    assert defaults.available_providers == (
        ProviderName.FMP,
        ProviderName.ALPHAVANTAGE,
    )

    configured = ClassificationProviderConfiguration.from_environment(
        {
            "EDGARITO_CLASSIFICATION_DEFAULT_PROVIDER": "alphavantage",
            "classification_available_providers": "alphavantage,fmp",
        }
    )
    assert configured.default_provider == ProviderName.ALPHAVANTAGE


def test_service_crosschecks_classification_without_merging(tmp_path):
    configuration = ClassificationProviderConfiguration()
    primary = _FakeProvider(ProviderName.FMP, _classification("fmp"))
    secondary = _FakeProvider(
        ProviderName.ALPHAVANTAGE,
        _classification("alphavantage", "Computer Hardware"),
    )
    service = CompanyClassificationService(
        FileSystemCache(tmp_path),
        configuration,
        providers={ProviderName.FMP: primary, ProviderName.ALPHAVANTAGE: secondary},
    )

    with pytest.warns(ClassificationCrosscheckWarning, match="industry differs"):
        result = asyncio.run(service.retrieve("AAPL"))

    assert result.industry == "Consumer Electronics"
    assert len(service.last_crosschecks) == 1


def test_cli_retrieves_classification_from_cached_fmp_profile(
    tmp_path, capsys, monkeypatch
):
    profile = [
        {
            "symbol": "AAPL",
            "companyName": "Apple Inc.",
            "cik": "320193",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "country": "US",
            "exchange": "NASDAQ",
        }
    ]
    cache = FileSystemCache(tmp_path)
    cache.save("providers/fmp/AAPL/profile.json", json.dumps(profile))
    monkeypatch.setattr(cli_module, "FMP_API_KEY", "test-api-key")

    exit_code = main(
        [
            "classification",
            "--ticker",
            "AAPL",
            "--cache-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Sector: Technology" in output
    assert "Industry: Consumer Electronics" in output
    assert "Source sector: Technology" in output
