# Edgarito

Edgarito retrieves company fundamentals from configured providers, caches raw responses, normalizes them into provider-neutral observations, optionally crosschecks providers, and displays historical financial statements in a CLI.

## Setup

```bash
uv sync
```

SEC requests require a descriptive user agent containing contact information. Add it to a `.env` file in the repository root:

```bash
EDGARITO_USER_AGENT="Your Name your-email@example.com"
```

The existing `.cli.env` format is also supported using its lowercase `user_agent` and `cache_path` keys. Shell environment variables and the `--user-agent` option are supported as well; the explicit command-line option takes precedence.

## Display financials

```bash
uv run edgarito financials --ticker AAPL
uv run edgarito financials --ticker AAPL --period quarterly --limit 8
uv run edgarito financials --cik 320193 --period all
uv run edgarito financials --ticker AAPL --concept revenue --concept net_income
uv run edgarito financials --ticker AAPL --provider alphavantage
uv run edgarito financials --ticker AAPL --provider fmp
uv run edgarito financials --ticker SAP.DEX --market eu
uv run edgarito financials --ticker RACE --market eu --period quarterly --limit 8
uv run edgarito financials --ticker AAPL --crosscheck
```

You can also run the Python module directly:

```bash
uv run python -m edgarito financials --ticker AAPL --user-agent "Your Name your-email@example.com"
```

Provider responses are cached below `cache/providers/`. Use `--refresh` to bypass existing snapshots. Asterisks in quarterly output mark values derived from reported YTD or full-year facts.

## Provider configuration

Configure a default and an allow-list for each supported market in `.env`:

```dotenv
us_default_provider=sec
us_available_providers=sec,alphavantage,fmp
eu_default_provider=alphavantage
eu_available_providers=alphavantage,fmp
```

The uppercase `EDGARITO_US_DEFAULT_PROVIDER`, `EDGARITO_US_AVAILABLE_PROVIDERS`, `EDGARITO_EU_DEFAULT_PROVIDER`, and `EDGARITO_EU_AVAILABLE_PROVIDERS` forms are also supported. A default must be in its market's allow-list. SEC supports US stocks; Alpha Vantage and FMP support US and EU stocks.

The CLI uses the configured provider for the selected market unless `--provider` is supplied. Market detection is not automatic: `--market` defaults to `us`, so use `--market eu` explicitly for EU companies and other IFRS-reporting issuers that should use the EU provider configuration.

The ticker alone does not always identify the appropriate reporting market. For example, Ferrari N.V. trades in the US under `RACE` but reports IFRS facts. Without `--market eu`, Edgarito selects the US default provider (SEC), whose current normalizer only supports US-GAAP facts:

```bash
uv run edgarito financials --ticker RACE --market eu --period quarterly --limit 8
```

`--provider` overrides the default only within the selected market's available-provider list. `--crosscheck` compares the selected result with every other available provider and emits warnings without changing or filling the selected dataset.

## Current flow

```text
FinancialDataService
    -> configured provider client
    -> provider normalizer
    -> NormalizedCompanyFinancials
    -> optional FinancialsCrosschecker
    -> FinancialsConsolePresenter
```

The normalized observations retain source taxonomy, source concept, accession number, filing form, filed date, unit, and derivation metadata so future providers can feed the same schema and be compared under `crosscheck`.

## Alpha Vantage provider

Configure the API key in `.env` or `.cli.env`:

```dotenv
alphavantage_api_key="your-api-key"
```

Retrieve and normalize fundamentals programmatically:

```python
import asyncio

from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization.alphavantage import AlphaVantageNormalizer
from edgarito.services.providers.alphavantage import AlphaVantageClient
from edgarito.settings import ALPHAVANTAGE_API_KEY


async def main():
    async with AlphaVantageClient(
        cache=FileSystemCache("cache"),
        api_key=ALPHAVANTAGE_API_KEY,
    ) as client:
        source = await client.get_company_financials("AAPL")
    return AlphaVantageNormalizer().normalize(source)


financials = asyncio.run(main())
```

The provider caches each Alpha Vantage endpoint separately below `cache/providers/alphavantage/`.

## Financial Modeling Prep provider

Configure the FMP API key in `.env` or `.cli.env`. The existing lowercase key is supported, as is the uppercase environment-variable form:

```dotenv
fmp_key="your-api-key"
# FMP_API_KEY="your-api-key"
```

Select FMP explicitly from the CLI:

```bash
uv run edgarito financials --ticker AAPL --provider fmp
uv run edgarito financials --ticker RACE --market eu --provider fmp --period quarterly
```

Retrieve and normalize FMP fundamentals directly:

```python
import asyncio

from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization.fmp import FmpNormalizer
from edgarito.services.providers.fmp import FmpClient
from edgarito.settings import FMP_API_KEY


async def main():
    async with FmpClient(
        cache=FileSystemCache("cache"),
        api_key=FMP_API_KEY,
    ) as client:
        source = await client.get_company_financials("AAPL")
    return FmpNormalizer().normalize(source)


financials = asyncio.run(main())
```

FMP profile, income statement, balance sheet, and cash-flow responses are cached separately below `cache/providers/fmp/`. Annual and quarterly statement responses use separate snapshots. Statement retrieval defaults to the latest five periods, which is compatible with FMP's entry-level subscription limit; direct `FmpClient` callers can set `statement_limit` for plans that allow more. The API key is sent as the `apikey` query parameter and is not included in cache paths or cached responses.

## Programmatic retrieval and crosschecking

`FinancialDataService.retrieve()` crosschecks other available providers by default and returns the selected provider's data unchanged. Crosscheck discrepancies and secondary-provider failures are warnings. Pass `crosscheck=False` to disable this behavior. The detailed reports remain available through `service.last_crosschecks`.

```python
import asyncio

from edgarito.enums.market import Market
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.reconciliation.financials import FinancialDataService
from edgarito.settings import (
    ALPHAVANTAGE_API_KEY,
    EDGARITO_USER_AGENT,
    FMP_API_KEY,
    PROVIDER_CONFIGURATION,
)


async def main():
    async with FinancialDataService(
        cache=FileSystemCache("cache"),
        provider_configuration=PROVIDER_CONFIGURATION,
        user_agent=EDGARITO_USER_AGENT,
        alphavantage_api_key=ALPHAVANTAGE_API_KEY,
        fmp_api_key=FMP_API_KEY,
    ) as service:
        return await service.retrieve(ticker="AAPL", market=Market.US)


financials = asyncio.run(main())
```

`FinancialsCrosschecker` can also be used directly when you already have two `NormalizedCompanyFinancials` objects. Its default value tolerance is 1% with a one-unit absolute floor.
