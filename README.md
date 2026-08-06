# Edgarito

Edgarito retrieves company fundamentals and classifications from configured providers, caches raw responses, normalizes them into provider-neutral models, optionally crosschecks providers, calculates historical financial metrics, and builds assumption-driven free cash flow forecasts. Yahoo Finance support is provided through `yfinance` for keyless financial statements and daily market history.

## Setup

```bash
uv sync
cp .env.example .env
```

Configure Edgarito in the repository-root `.env` file:

```dotenv
cache_path=./cache
user_agent="Your Name (your-email@example.com)"
alphavantage_api_key="your-api-key"
fmp_key="your-api-key"
# Optional: raises the rate limit for otherwise free ISIN mapping.
# openfigi_api_key="your-api-key"
```

Replace the placeholders needed by the providers you use. OpenFIGI does not
require a key; omit `openfigi_api_key` unless you want its higher rate limit.

`user_agent` is required only when the SEC provider is selected. The SEC asks clients to identify themselves with contact information. `alphavantage_api_key` and `fmp_key` are required only for their respective providers. Yahoo requires no API key. `cache_path` defaults to `cache` when omitted.

The CLI reads `.env` automatically. `--user-agent` and `--cache-dir` override their corresponding dotenv values for an individual command.

## Display financials

```bash
uv run edgarito financials --ticker AAPL
uv run edgarito financials --isin US0378331005 --provider fmp
uv run edgarito financials --ticker AAPL --period quarterly --limit 8
uv run edgarito financials --cik 320193 --period all
uv run edgarito financials --ticker AAPL --concept revenue --concept net_income
uv run edgarito financials --ticker AAPL --provider alphavantage
uv run edgarito financials --ticker AAPL --provider fmp
uv run edgarito financials --ticker AAPL --provider yahoo
uv run edgarito financials --ticker SAP.DE --market eu
uv run edgarito financials --ticker RACE --market eu --period quarterly --limit 8
uv run edgarito financials --ticker AAPL --crosscheck
```

## Security identifiers and symbol mappings

Every financials, metrics, and classification command accepts a ticker, CIK, or
ISIN as its primary identifier. A CIK can be used directly with SEC. ISINs are
mapped through OpenFIGI's free public API and the result is cached. Resolving a
CIK for a symbol-based provider or an exchange-qualified ticker uses FMP's search
directory and therefore requires `fmp_key`:

```bash
uv run edgarito financials --isin US0378331005 --provider fmp
uv run edgarito financials --cik 320193 --provider fmp
uv run edgarito financials --ticker RACE --exchange MIL --provider fmp
```

Use explicit mappings when providers spell the same listing differently. They
also avoid a lookup and are used during crosschecks:

```bash
uv run edgarito financials \
  --ticker SAP \
  --market eu \
  --exchange XETRA \
  --exchange-symbol XETRA=SAP.DE \
  --provider-symbol yahoo=SAP.DE \
  --provider-symbol alphavantage=SAP.DEX \
  --provider-symbol fmp=SAP.DE \
  --crosscheck
```

Symbol precedence is provider mapping, selected exchange mapping, then canonical
ticker. `--exchange-symbol` and `--provider-symbol` can each be repeated. If an
ISIN has several listings, add `--exchange` or an explicit provider mapping to
remove the ambiguity. See the official
[OpenFIGI mapping documentation](https://www.openfigi.com/api/documentation) and
FMP's [exchange variants documentation](https://site.financialmodelingprep.com/developer/docs/stable/search-exchange-variants).

Programmatic callers can pass the same fields individually or use one reusable
mapping object:

```python
from edgarito.enums.provider import ProviderName
from edgarito.schemas.identifiers import SecurityIdentifiers


identifiers = SecurityIdentifiers(
    ticker="SAP",
    isin="DE0007164600",
    exchange="XETRA",
    exchange_symbols={"XETRA": "SAP.DE"},
    provider_symbols={
        ProviderName.ALPHAVANTAGE: "SAP.DEX",
        ProviderName.FMP: "SAP.DE",
        ProviderName.YAHOO: "SAP.DE",
    },
)

financials = await service.retrieve(identifiers=identifiers, market="eu")
```

Resolved and explicit mappings are retained on
`NormalizedCompanyFinancials.identifiers` and
`NormalizedCompanyClassification.identifiers` for traceability.

You can also run the Python module directly:

```bash
uv run python -m edgarito financials --ticker AAPL --user-agent "Your Name (your-email@example.com)"
```

Provider responses are cached below `<cache_path>/providers/`. Use `--refresh` to bypass existing snapshots. In SEC quarterly output, an asterisk marks a value derived from a reported YTD or full-year fact.

US financial retrieval still defaults to SEC. EU financial retrieval defaults to
Yahoo because it is keyless and accepts Yahoo's exchange-qualified symbols such
as `SAP.DE`, `AIR.PA`, and `ASML.AS`; Alpha Vantage and FMP remain available as
overrides and crosschecks. Yahoo statements are standardized secondary-source
tables, not a substitute for an audited filing. Their available history and line
items vary by security, and Edgarito only maps rows whose meaning is sufficiently
clear. `yfinance` is not affiliated with or endorsed by Yahoo; review Yahoo's
[terms](https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html)
before using its data outside personal research. The implemented methods follow
the [yfinance Ticker API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html).

## Retrieve Yahoo market history

Yahoo also supplies daily raw and adjusted prices, dividends, and splits. The
client runs yfinance's synchronous calls off the event loop, caches a serializable
snapshot, and enables its price-repair option by default. Normalize the snapshot
into Edgarito's shared market schema as follows:

```python
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization import YahooMarketNormalizer
from edgarito.services.providers import YahooFinanceClient


client = YahooFinanceClient(FileSystemCache("cache"))
source = await client.get_price_history("SAP.DE", period="5y")
market_data = YahooMarketNormalizer().normalize(source)
latest_close = market_data.latest_price.close
```

For London and Johannesburg listings quoted in pence/cents (`GBp`/`GBX` and
`ZAc`), normalization converts prices and dividends to base GBP/ZAR units.

SEC normalization includes valuation-oriented reported inputs in addition to the
core statements: pretax income and tax expense; depreciation and amortization;
current assets, receivables, inventory, prepaid/other current assets, current
liabilities, payables, accrued liabilities, and current deferred revenue;
short-term and current/noncurrent long-term debt; interest expense; goodwill and
net intangible assets; dividends paid and dividends per share; and current,
basic weighted-average, and diluted weighted-average shares. Current shares are
read from the SEC `dei` taxonomy and retain that provenance. Weighted-average
shares are not additive, so quarterly values are only emitted when directly
reported rather than derived from YTD or annual observations.

## Retrieve sector and industry

```bash
uv run edgarito classification --ticker AAPL
uv run edgarito classification --ticker RACE --provider alphavantage
uv run edgarito classification --ticker AAPL --crosscheck
```

Classification defaults to FMP and can also use Alpha Vantage. Edgarito maps provider sector labels into one 11-sector vocabulary. Industry classifications remain provider-defined; the normalized result retains the cleaned industry name, original provider labels, and taxonomy metadata rather than silently treating different taxonomies as equivalent.

`--provider` overrides the classification default. `--crosscheck` compares the selected classification with every other configured classification provider, emits warnings for sector or industry differences, and never merges the results. `--refresh` bypasses cached provider profiles.

## Compute metrics

Use the `metrics` command with the same identifier, provider, market, period, cache, and crosscheck options as `financials`:

```bash
uv run edgarito metrics --ticker AAPL
uv run edgarito metrics --ticker AAPL --period quarterly --limit 8
uv run edgarito metrics --ticker AAPL --provider fmp
uv run edgarito metrics --ticker RACE --market eu
uv run edgarito metrics --ticker AAPL --metric revenue_growth --metric net_margin
uv run edgarito metrics --ticker AAPL --crosscheck
```

`--metric` can be repeated and accepts:

- `revenue_growth`
- `operating_margin`
- `net_margin`
- `effective_tax_rate`
- `nopat`
- `ebitda`
- `free_cash_flow`
- `free_cash_flow_margin`
- `operating_working_capital`
- `change_in_operating_working_capital`
- `gross_debt`
- `net_debt`
- `tangible_book_equity`
- `fcff`
- `return_on_assets`
- `return_on_equity`
- `liabilities_to_assets`
- `cash_to_liabilities`
- `operating_cash_flow_to_net_income`

Metrics are calculated only from the selected provider's normalized observations. `--crosscheck` validates the underlying observations against other configured providers but does not merge their values.

Revenue growth compares consecutive periods: year over year for annual data and quarter over quarter for quarterly data. Return on assets and return on equity use average beginning and ending balances; quarterly returns are not annualized. Free cash flow is operating cash flow minus capital expenditures.

Valuation building blocks use strict formulas over atomic normalized facts:

```text
effective tax rate = income tax expense / pretax income
NOPAT              = operating income × (1 - effective tax rate)
EBITDA              = operating income + depreciation and amortization
operating NWC       = receivables + inventory + prepaid/other current assets
                      - payables - accrued liabilities - current deferred revenue
gross debt          = short-term debt + current long-term debt
                      + noncurrent long-term debt
net debt            = gross debt - cash and equivalents
tangible book       = stockholders' equity - goodwill - net intangible assets
FCFF                = NOPAT + depreciation and amortization - capex
                      - change in operating NWC
```

These calculations deliberately require every listed component and consistent
units rather than treating a missing filing fact as zero. A metric is also
omitted when its denominator is zero or a required prior period is not
consecutive. FCFE is not calculated yet because debt issuance and repayment have
not been normalized.

Programmatically, calculate all metrics from an existing `NormalizedCompanyFinancials` object, or pass a selected metric set:

```python
from edgarito.services.metrics import FinancialMetric, FinancialMetricsService


company_metrics = FinancialMetricsService().calculate(
    financials,
    metrics={FinancialMetric.REVENUE_GROWTH, FinancialMetric.NET_MARGIN},
)
```

Each metric observation retains its formula and required input concepts for traceability.

## Forecast free cash flow

The `forecast` command projects annual revenue and free cash flow using explicit,
auditable assumptions. Percentage arguments use percentage points, so `6` means
6%:

```bash
uv run edgarito forecast --ticker AAPL --years 5 \
  --revenue-growth 6 --fcf-margin 25
```

A single growth or margin value is held constant for every projected year. To
provide a year-by-year path, repeat each option exactly once per forecast year:

```bash
uv run edgarito forecast --isin US0378331005 --years 3 \
  --revenue-growth 8 --revenue-growth 6 --revenue-growth 4 \
  --fcf-margin 26 --fcf-margin 25 --fcf-margin 24
```

Either assumption can be omitted. Edgarito then uses its trailing average from
up to the latest complete annual periods; `--historical-window` controls the
maximum window and defaults to three years:

```bash
uv run edgarito forecast --ticker AAPL --years 5
uv run edgarito forecast --ticker AAPL --years 5 --historical-window 5 \
  --revenue-growth 5
```

The deterministic forecast method is:

```text
projected revenue[t] = projected revenue[t-1] × (1 + revenue growth[t])
projected FCF[t]     = projected revenue[t] × FCF margin[t]
historical FCF       = operating cash flow - capital expenditures
```

Forecasting requires normalized annual revenue, operating cash flow, and capital
expenditures in one currency. It does not discount cash flows or calculate a
terminal value yet; the resulting annual FCF observations are the intended input
for the later DCF valuation layer.

Programmatically, assumptions can be constant scalars or complete paths:

```python
from decimal import Decimal

from edgarito.services.forecasting import (
    FreeCashFlowForecastParameters,
    FreeCashFlowForecastService,
)


parameters = FreeCashFlowForecastParameters(
    forecast_years=5,
    revenue_growth=Decimal("6"),
    free_cash_flow_margin=(
        Decimal("26"),
        Decimal("25.5"),
        Decimal("25"),
        Decimal("24.5"),
        Decimal("24"),
    ),
)
forecast = FreeCashFlowForecastService().forecast(financials, parameters)
```

Each forecast observation retains its fiscal year, period end, revenue, FCF,
growth assumption, margin assumption, currency, and formula. The forecast also
records whether each assumption path was explicit or inferred from historical
averages.

## Select suitable valuation models

Before calculating a valuation, `valuation-models` builds an economic profile and
ranks the minimal supported model families:

- FCFF DCF
- Equity DCF / Dividend Discount
- Residual Income
- NAV / Sum-of-the-Parts
- Comparable Multiples

The selector separates economic suitability from data readiness. A model can be
the correct primary method while still being blocked by missing inputs, and a
hard economic rejection is not treated as a data problem.

```bash
uv run edgarito valuation-models --ticker AAPL
uv run edgarito valuation-models --ticker JPM
uv run edgarito valuation-models --isin US02079K3059
```

The economic profile uses normalized sector and provider industry together with
historical revenue, earnings, free cash flow, book equity, lifecycle and
cyclicality. Sector is only an initial clue: bank, insurer, REIT, resource,
pipeline, holding-company and conglomerate patterns take precedence when the
industry identifies the underlying economics. A broad Financials or Real Estate
classification without a specific business type is reported as unresolved and
does not receive a primary model automatically.

Explicit facts can override uncertain provider classifications and mark external
valuation inputs as available:

```bash
uv run edgarito valuation-models --ticker TEST \
  --business-type holding_company \
  --trait multi_segment \
  --available-input segment_values \
  --available-input net_debt \
  --available-input diluted_shares
```

Other useful options include `--lifecycle`, `--cyclicality`, repeatable `--trait`,
repeatable `--available-input`, `--peer-count`, and
`--classification-provider`. The report returns:

- one economically preferred primary intrinsic model;
- conditional intrinsic alternatives;
- comparable-multiple cross-checks and suitable denominator bases;
- hard rejection reasons;
- the required forecasting profile;
- readiness and exact missing inputs for every model.

Programmatically:

```python
from edgarito.services.valuation import (
    BusinessArchetype,
    ValuationInput,
    ValuationModelSelector,
    ValuationProfileBuilder,
    ValuationProfileOverrides,
)


profile = ValuationProfileBuilder().build(
    financials,
    classification,
    ValuationProfileOverrides(
        business_archetype=BusinessArchetype.HOLDING_COMPANY,
        available_inputs={ValuationInput.SEGMENT_VALUES},
    ),
)
selection = ValuationModelSelector().select(profile)
primary = selection.primary
```

The current `operating cash flow - capital expenditures` history is deliberately
not marked as FCFF. Enterprise DCF remains blocked until an explicit FCFF forecast
and its EBIT, tax, depreciation, working-capital and investment assumptions are
available.

## Market data and valuation assumptions

Observed market data and selected valuation assumptions use separate schemas.
`SecurityMarketData` stores dated prices, dividends, and splits for one security;
`ReferenceMarketSeries` stores rates and macroeconomic observations such as a
Treasury yield or ECB inflation series. Both retain provider, retrieval time,
frequency, and source-version metadata. Retrieval timestamps must be
timezone-aware.

```python
import datetime
from decimal import Decimal

from edgarito.schemas import (
    MarketDataFrequency,
    PriceBar,
    SecurityIdentifiers,
    SecurityMarketData,
)


market_data = SecurityMarketData(
    provider="alphavantage",
    provider_symbol="AAPL",
    identifiers=SecurityIdentifiers(ticker="AAPL"),
    currency="USD",
    frequency=MarketDataFrequency.DAILY,
    retrieved_at=datetime.datetime.now(datetime.timezone.utc),
    prices=(
        PriceBar(
            observed_on=datetime.date(2026, 8, 6),
            close=Decimal("212.00"),
        ),
    ),
)
```

A `ValuationAssumption` records a selected scalar separately from its source.
Rates and margins use percentage points, beta values use multiples, and an
assumption can be scoped by currency, country, industry, company, and forecast
year. `ValuationAssumptionSet` groups unique assumptions for one valuation date
and scenario. Market-derived assumptions require a provider and observation
date; reference-dataset assumptions additionally require a dataset name and
version. This ensures that retrieving a rate does not silently turn it into a
valuation input.

### Reference providers

The reference clients normalize their output directly into the schemas above:

- `TreasuryClient` retrieves daily U.S. par yields from Treasury's free XML feed.
- `FredClient` retrieves economic series and optional point-in-time vintages. The
  API is free but requires `FRED_API_KEY` / `fred_api_key`.
- `EcbClient` retrieves one series at a time from the free ECB SDMX API.
- `DamodaranClient` retrieves typed country-risk and U.S. industry-beta tables.
  Each release declares a version, publication date, source URL, and SHA-256. A
  changed upstream file is rejected before it reaches the cache.

```python
import datetime
import os

from edgarito.schemas import ReferenceSeriesKind
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.providers import EcbClient, FredClient, TreasuryClient


cache = FileSystemCache("cache")

async with TreasuryClient(cache) as client:
    treasury = await client.get_par_yield(120, year=2026)

async with FredClient(cache, os.environ["FRED_API_KEY"]) as client:
    fred = await client.get_series(
        "DGS10",
        kind=ReferenceSeriesKind.GOVERNMENT_YIELD,
        vintage_date=datetime.date(2026, 8, 6),
        currency="USD",
        country="US",
    )

async with EcbClient(cache) as client:
    ecb = await client.get_series(
        "FM",
        "D.U2.EUR.4F.KR.MRR_FR.LEV",
        kind=ReferenceSeriesKind.POLICY_RATE,
    )
```

The built-in Damodaran releases are `COUNTRY_RISK_PREMIUMS_2026` and
`US_INDUSTRY_BETAS_2026`. Add a new `ReferenceDatasetRelease` with a verified
checksum when the source publishes an update; do not relabel a mutable URL with
an older version.

## Provider configuration

The built-in provider routing is equivalent to:

```dotenv
us_default_provider=sec
us_available_providers=sec,alphavantage,fmp
eu_default_provider=alphavantage
eu_available_providers=alphavantage,fmp
classification_default_provider=fmp
classification_available_providers=fmp,alphavantage
```

Add any of these lowercase keys to `.env` to override the corresponding default. A default provider must appear in its corresponding available-provider list. SEC supports US financial statements; Alpha Vantage and FMP support financial statements and classifications for both US and EU stocks.

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
       or FinancialMetricsService -> MetricsConsolePresenter
       or FreeCashFlowForecastService -> ForecastConsolePresenter
       or ValuationProfileBuilder -> ValuationModelSelector
          -> ValuationSelectionConsolePresenter

CompanyClassificationService
    -> configured FMP or Alpha Vantage profile client
    -> CompanyClassificationNormalizer
    -> NormalizedCompanyClassification
    -> optional classification crosscheck
    -> ClassificationConsolePresenter
```

Normalized observations share concept, statement, value, currency, fiscal period, source taxonomy, source concept, and filing metadata fields. Provider-specific metadata such as SEC accession numbers and filing forms remains populated when the source supplies it.

## Alpha Vantage provider

Configure the API key in `.env`:

```dotenv
alphavantage_api_key="your-api-key"
```

Retrieve and normalize fundamentals programmatically:

```python
import asyncio
import os

from dotenv import load_dotenv

from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization.alphavantage import AlphaVantageNormalizer
from edgarito.services.providers.alphavantage import AlphaVantageClient


load_dotenv()


async def main():
    async with AlphaVantageClient(
        cache=FileSystemCache(os.getenv("cache_path", "cache")),
        api_key=os.environ["alphavantage_api_key"],
    ) as client:
        source = await client.get_company_financials("AAPL")
    return AlphaVantageNormalizer().normalize(source)


financials = asyncio.run(main())
```

Retrieve the intentionally limited market-data scope separately:

```python
import asyncio
import os

from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization import AlphaVantageMarketNormalizer
from edgarito.services.providers import AlphaVantageClient


async def market():
    cache = FileSystemCache(os.getenv("cache_path", "cache"))
    async with AlphaVantageClient(
        cache, os.environ["alphavantage_api_key"]
    ) as client:
        daily = await client.get_daily_prices("AAPL")
        dividends = await client.get_dividends("AAPL")
        splits = await client.get_splits("AAPL")
        latest_close = await client.get_latest_close("AAPL")

    normalized = AlphaVantageMarketNormalizer().normalize(
        symbol="AAPL",
        currency="USD",
        daily_prices=daily,
        dividends=dividends,
        splits=splits,
    )
    return normalized, latest_close


market_data, latest_close = asyncio.run(market())
```

Daily prices are raw as-traded OHLCV values, so `adjusted_close` remains unset.
The default `compact` request returns the latest 100 observations and works with
free keys. `AlphaVantageOutputSize.FULL` is explicit because Alpha Vantage now
requires a premium key for full daily history. The end-of-day global quote used
by `get_latest_close` remains available to free keys; realtime entitlements are
not requested.

The provider caches each Alpha Vantage endpoint and output-size variant
separately below `<cache_path>/providers/alphavantage/`.

## Financial Modeling Prep provider

Configure the FMP API key in `.env`:

```dotenv
fmp_key="your-api-key"
```

Select FMP explicitly from the CLI:

```bash
uv run edgarito financials --ticker AAPL --provider fmp
uv run edgarito financials --ticker RACE --market eu --provider fmp --period quarterly
```

Retrieve and normalize FMP fundamentals directly:

```python
import asyncio
import os

from dotenv import load_dotenv

from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.normalization.fmp import FmpNormalizer
from edgarito.services.providers.fmp import FmpClient


load_dotenv()


async def main():
    async with FmpClient(
        cache=FileSystemCache(os.getenv("cache_path", "cache")),
        api_key=os.environ["fmp_key"],
    ) as client:
        source = await client.get_company_financials("AAPL")
    return FmpNormalizer().normalize(source)


financials = asyncio.run(main())
```

FMP profile, income statement, balance sheet, and cash-flow responses are cached separately below `<cache_path>/providers/fmp/`. Annual and quarterly statement responses use separate snapshots. Statement retrieval defaults to the latest five periods; direct `FmpClient` callers can set `statement_limit` when their subscription allows more. The API key is not included in cache paths or cached responses.

## Programmatic classification retrieval

`CompanyClassificationService.retrieve()` crosschecks other configured classification providers by default and returns the selected provider's result unchanged. Pass `crosscheck=False` to disable warnings. The CLI is intentionally opt-in through `--crosscheck`.

```python
import asyncio
import os

from dotenv import load_dotenv

from edgarito.config.providers import ClassificationProviderConfiguration
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.reconciliation.classification import CompanyClassificationService


load_dotenv()


async def main():
    async with CompanyClassificationService(
        cache=FileSystemCache(os.getenv("cache_path", "cache")),
        provider_configuration=ClassificationProviderConfiguration.from_environment(
            os.environ
        ),
        alphavantage_api_key=os.getenv("alphavantage_api_key"),
        fmp_api_key=os.getenv("fmp_key"),
    ) as service:
        return await service.retrieve("AAPL")


classification = asyncio.run(main())
```

## Programmatic retrieval and crosschecking

`FinancialDataService.retrieve()` crosschecks other available providers by default and returns the selected provider's data unchanged. Crosscheck discrepancies and secondary-provider failures are warnings. Pass `crosscheck=False` to disable this behavior. The detailed reports remain available through `service.last_crosschecks`.

```python
import asyncio
import os

from dotenv import load_dotenv

from edgarito.config.providers import ProviderConfiguration
from edgarito.enums.market import Market
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.reconciliation.financials import FinancialDataService


load_dotenv()


async def main():
    async with FinancialDataService(
        cache=FileSystemCache(os.getenv("cache_path", "cache")),
        provider_configuration=ProviderConfiguration.from_environment(os.environ),
        user_agent=os.getenv("user_agent"),
        alphavantage_api_key=os.getenv("alphavantage_api_key"),
        fmp_api_key=os.getenv("fmp_key"),
    ) as service:
        return await service.retrieve(ticker="AAPL", market=Market.US)


financials = asyncio.run(main())
```

`FinancialsCrosschecker` can also be used directly when you already have two `NormalizedCompanyFinancials` objects. Its default value tolerance is 1% with a one-unit absolute floor.

## Development checks

Install the development tools and run the same checks expected for the project:

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

To apply safe lint fixes and formatting locally:

```bash
uv run ruff check . --fix
uv run ruff format .
```

Ruff checks Python errors, import ordering, and common bug patterns. The copied SEC filing-type reference table is excluded to avoid rewriting generated reference data.
