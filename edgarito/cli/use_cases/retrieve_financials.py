from edgarito.enums.provider import ProviderName
from edgarito.schemas.cli.use_cases.retrieve_financials import RetrieveFinancialsRequest
from edgarito.schemas.normalization.financials import NormalizedCompanyFinancials
from edgarito.services.normalization.sec_us_gaap import SecUsGaapNormalizer
from edgarito.services.providers.edgar import EdgarClient


class RetrieveFinancials:
    """Retrieve and normalize company historicals from the SEC provider."""

    def __init__(self, edgar: EdgarClient, normalizer: SecUsGaapNormalizer):
        self._edgar = edgar
        self._normalizer = normalizer

    async def execute(
        self, request: RetrieveFinancialsRequest
    ) -> NormalizedCompanyFinancials:
        identifiers = request.security_identifiers()
        sec_symbol = identifiers.symbol_for(ProviderName.SEC)
        ticker = identifiers.ticker or sec_symbol
        cik = identifiers.cik
        if sec_symbol is not None:
            cik = await self._edgar.get_cik(
                sec_symbol,
                use_cache=request.use_cache,
                make_cache=request.make_cache,
            )

        if cik is None:
            raise ValueError(
                "SEC retrieval requires a CIK, ticker, or SEC provider symbol"
            )
        facts = await self._edgar.get_company_facts(
            cik,
            use_cache=request.use_cache,
            make_cache=request.make_cache,
        )
        return self._normalizer.normalize(
            facts,
            ticker=ticker,
            granularity=request.granularity,
            concepts=request.concepts,
        )
