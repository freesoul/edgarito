from edgarito.schemas.normalization.financials import NormalizedCompanyFinancials
from edgarito.schemas.cli.use_cases.retrieve_financials import RetrieveFinancialsRequest
from edgarito.services.normalization.sec_us_gaap import SecUsGaapNormalizer
from edgarito.services.providers.edgar import EdgarClient


class RetrieveFinancials:
    """Retrieve and normalize company historicals from the SEC provider."""

    def __init__(self, edgar: EdgarClient, normalizer: SecUsGaapNormalizer):
        self._edgar = edgar
        self._normalizer = normalizer

    async def execute(self, request: RetrieveFinancialsRequest) -> NormalizedCompanyFinancials:
        ticker = request.ticker.upper() if request.ticker else None
        cik = request.cik
        if ticker is not None:
            cik = await self._edgar.get_cik(
                ticker,
                use_cache=request.use_cache,
                make_cache=request.make_cache,
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
