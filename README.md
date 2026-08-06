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

## Configure forecast and valuation profiles

The default profile is the versioned JSON file at
`configs/valuation/default.json`. It centralizes parameters for FCFF and
simplified forecasts, discount rates, present-value timing, terminal value,
model-selection overrides, comparable selection, and specialized input history.
The default WACC and perpetual growth are `null`: for `valuation`, that means
"resolve from provider and company data." A CLI or profile value always takes
precedence over an automatically resolved value.

All profile-enabled commands use the root default automatically. Supply another
profile with `--profile`; explicitly supplied CLI options take precedence over
the selected profile:

```bash
uv run edgarito forecast --ticker AAPL --profile configs/valuation/growth.json
uv run edgarito valuation-models --ticker JPM --profile configs/valuation/bank.json
uv run edgarito comparables --ticker AAPL --peer MSFT \
  --profile configs/valuation/growth.json
uv run edgarito specialized-inputs --ticker PLD --type reit \
  --profile configs/valuation/reit.json
```

Company-specific profiles may override unreliable provider classifications for
valuation purposes. For example, the bundled Ferrari profile treats `RACE` as a
consumer-discretionary luxury-goods business; Damodaran's `Apparel` row is used
as the closest available luxury beta proxy:

```bash
uv run edgarito valuation --ticker RACE --market eu \
  --profile configs/valuation/race.json
```

That command retains a conventional perpetuity-growth terminal value. The same
profile also stores an explicitly optimistic 22.5x EBITDA terminal scenario,
with revenue still fading toward a 2.9% stable-growth anchor:

```bash
uv run edgarito valuation --ticker RACE --market eu \
  --profile configs/valuation/race.json \
  --terminal-method exit_multiple
```

The premium scenario is intended as an analyst/market-pricing comparison, not a
replacement for the intrinsic base case. The profile also uses Ferrari's
published 2025 industrial debt and cash bridge, its 2026-2030 margin guidance,
its cumulative capex target, and its announced EUR 3.5 billion 2026-2030
repurchase plan. Buybacks are shown as cash spent and shares retired rather than
being added to FCFF or treated as free per-share accretion.

Custom files may contain only the sections they change. Omitted values inherit
the validated schema defaults. Decimal parameters should be written as JSON
strings to preserve their exact value:

```json
{
  "schema_version": 1,
  "name": "growth",
  "forecast": {
    "fcff": {
      "forecast_years": 7,
      "revenue_growth": ["12", "10", "8", "7", "6", "5", "4"],
      "operating_margin": "25"
    }
  },
  "valuation": {
    "cash_flow_timing": "mid_year",
    "terminal_value": {"perpetual_growth_rate": "2"}
  },
  "comparables": {"max_peers": 10, "minimum_score": 60}
}
```

Unknown fields and invalid ranges fail before provider retrieval. Configured cost
of equity, WACC, and terminal-growth inputs are also reflected in
`valuation-models` data readiness.

## Forecast free cash flow to the firm

The `forecast` command now defaults to a driver-based FCFF forecast. Omitted
drivers are inferred as trailing averages from the latest complete annual
periods; `--historical-window` defaults to three years:

```bash
uv run edgarito forecast --ticker AAPL --years 5
```

Every driver can instead be supplied explicitly. Percentage arguments use
percentage points, so `6` means 6%. A value supplied once is held constant; to
define a path, repeat an option exactly once per forecast year:

```bash
uv run edgarito forecast --ticker AAPL --years 5 \
  --revenue-growth 6 \
  --operating-margin 30 \
  --tax-rate 21 \
  --depreciation-to-revenue 3 \
  --capex-to-revenue 4 \
  --operating-working-capital-to-revenue -10
```

The deterministic operating bridge is:

```text
revenue[t]       = revenue[t-1] × (1 + revenue growth[t])
EBIT[t]          = revenue[t] × operating margin[t]
NOPAT[t]         = EBIT[t] × (1 - tax rate[t])
D&A[t]           = revenue[t] × D&A / revenue[t]
capex[t]         = revenue[t] × capex / revenue[t]
operating NWC[t] = revenue[t] × operating NWC / revenue[t]
change NWC[t]    = operating NWC[t] - operating NWC[t-1]
FCFF[t]          = NOPAT[t] + D&A[t] - capex[t] - change NWC[t]
```

The base operating NWC and inferred drivers use the same calculations as the
historical FCFF metrics. Detailed working-capital liabilities are preferred. If
a filing does not expose a standalone accrued-liabilities fact, operating NWC
falls back to receivables + separately reported inventory + prepaid/other
current assets - total current liabilities + reported current debt. SEC
depreciation and intangible
amortization facts are also combined when the filer does not publish one D&A
fact. These fallbacks retain their formulas and source concepts; unsupported
missing inputs are still reported rather than silently replaced with zero.

This standard corporate FCFF bridge is intended for ordinary non-financial
companies. The valuation-model selector can recommend specialized forecast
profiles for financial intermediaries, REITs, resource producers, project
pipelines, highly cyclical companies, and other exceptions; those profiles are
not silently approximated by this command.

The previous revenue-times-FCF-margin approach remains available as a simplified
scenario. Its cash flow is not FCFF and must not be used as an enterprise DCF
input:

```bash
uv run edgarito forecast --ticker AAPL --method simplified --years 5 \
  --revenue-growth 6 --fcf-margin 25
```

Programmatically, use the explicit FCFF types. The older generic
`FreeCashFlowForecast...` names are compatibility aliases for this new default:

```python
from decimal import Decimal

from edgarito.services.forecasting import FcffForecastParameters, FcffForecastService


parameters = FcffForecastParameters(
    forecast_years=5,
    revenue_growth=Decimal("6"),
    operating_margin=Decimal("30"),
    tax_rate=Decimal("21"),
    depreciation_to_revenue=Decimal("3"),
    capex_to_revenue=Decimal("4"),
    operating_working_capital_to_revenue=Decimal("-10"),
)
forecast = FcffForecastService().forecast(financials, parameters)
```

Use `SimplifiedFcfForecastParameters` and `SimplifiedFcfForecastService` for the
retained scenario model. Neither method discounts cash flows or calculates a
terminal value; that remains the responsibility of the later valuation layer.

## Calculate discount rates and present values

The shared valuation foundation calculates levered beta, CAPM cost of equity,
after-tax cost of debt, market-value WACC, present values, and terminal values.
Rates use percentage points consistently: `8` means 8%. Capital weights in the
result are decimal proportions, so `0.8` means 80%.

```python
from edgarito.services.valuation import (
    CashFlow,
    DiscountRateService,
    PresentValueService,
    TerminalValueService,
)


cost_of_equity = DiscountRateService.cost_of_equity(
    risk_free_rate="4",
    levered_beta="1.1",
    equity_risk_premium="5",
    country_risk_premium="0.5",
)
wacc = DiscountRateService.wacc(
    cost_of_equity=cost_of_equity.cost_of_equity,
    pretax_cost_of_debt="5",
    normalized_tax_rate="25",
    market_value_equity="800",
    market_value_debt="200",
)
terminal = TerminalValueService.perpetuity_growth(
    final_cash_flow="100",
    discount_rate=wacc.wacc,
    perpetual_growth_rate="2",
)
present_values = PresentValueService.discount(
    (
        CashFlow(amount="80", period=1, label="Year 1 FCFF"),
        CashFlow(amount=terminal.terminal_value, period=5, label="Terminal value"),
    ),
    discount_rate=wacc.wacc,
    unit="USD",
)
```

Periods may be fractional, such as `0.5` for a mid-year convention. Perpetuity
growth uses the final explicit-period cash flow and grows it once before applying
the Gordon formula. Terminal values are returned at the end of the explicit
forecast and must be discounted using their stated period. Exit-multiple terminal
values are also supported through `TerminalValueService.exit_multiple`.

## Value a company with FCFF DCF

The `valuation` command now executes a complete FCFF DCF. It builds the
driver-based forecast, discounts each projected FCFF, calculates and discounts
terminal value, then bridges enterprise value to diluted per-share equity value:

```text
enterprise value = PV(explicit FCFF) + PV(terminal value)
equity value     = enterprise value - net debt
value per share  = equity value / diluted shares
```

The root profile leaves WACC and perpetual growth unset. A basic valuation still
runs without valuation-assumption flags because the CLI resolves them from
market, reference, and company data:

```bash
uv run edgarito valuation --ticker AAPL --years 5
```

FCFF valuation uses an adaptive multistage projection by default. It derives the
initial growth regime, assigns up to three high-growth years according to the
gap from perpetual growth, and then fades growth by no more than three
percentage points per year. `--years` is a minimum horizon: when it ends before
the first complete stable year, the projection is extended automatically. Once
stability has been reached, requesting more years only moves stable cash flows
between the explicit and terminal portions, so value remains effectively
unchanged.

Use `--projection-method constant` to retain the previous constant-driver
projection. Multistage behavior can also be tuned or disabled in a profile:

```json
{
  "valuation": {
    "multistage": {
      "enabled": true,
      "max_annual_growth_fade": "3",
      "minimum_transition_years": 3,
      "maximum_transition_years": 10,
      "maximum_high_growth_years": 3,
      "extend_to_stable": true
    }
  }
}
```

An explicit multi-year revenue-growth path is preserved as the first stage and
the adaptive fade begins after its final value. Other explicit driver paths are
preserved and extended using their final supplied values.

For EUR companies, the risk-free rate is the ECB 10-year AAA euro-area yield and
terminal growth uses the trailing ECB HICP distribution. For USD companies, the
risk-free rate is the U.S. Treasury 10-year yield; terminal growth uses FRED
inflation when `FRED_API_KEY` is configured and otherwise a conservative
Treasury-yield-based proxy. Damodaran supplies versioned country-risk/tax and
industry-beta references. Yahoo supplies the latest price, classification, and
market capitalization; reported tax, interest, debt, and shares complete the
company-specific calculation.

For dual listings and ADRs, Yahoo's quote currency can differ from the financial
statement currency. Before calculating market capitalization, valuation converts
the latest quote through the ECB's daily currency-per-euro reference rates and
records `yahoo+ecb-fx` in the WACC source. This includes direct conversions such
as USD to EUR and euro-cross conversions between two non-EUR currencies.

The CLI prints every selected assumption and its provider. Historical cost of
debt and book debt as a market-debt proxy are estimates, so override `--wacc` or
the component fields in a company-specific profile when the issuer's economics
make them unsuitable.

When final explicit FCFF growth remains at least one percentage point away from
perpetual growth, the valuation also warns that the terminal transition is
abrupt and that the result may be sensitive to `--years`.

Or retain them in a selected profile:

```json
{
  "name": "base-dcf",
  "valuation": {
    "discount_rates": {"wacc": "8"},
    "terminal_value": {"perpetual_growth_rate": "2"}
  }
}
```

```bash
uv run edgarito valuation --ticker AAPL \
  --profile configs/valuation/base-dcf.json
```

The valuation inherits all FCFF drivers from the same profile used by
`forecast`. CLI forecast-driver options remain available as overrides. Net debt
is calculated from normalized debt and cash components, while diluted
weighted-average shares are preferred over basic shares. For Yahoo statements,
aggregate `CurrentDebt` takes precedence over a separately reported current
portion, preventing double counting when that atomic field is absent.

Missing bridge data can be supplied as net debt directly, or as gross debt and
cash components:

```json
{
  "valuation": {
    "capital_bridge": {
      "gross_debt": "1800000000",
      "cash_and_equivalents": "2500000000",
      "diluted_shares": "46700000"
    }
  }
}
```

Equivalent CLI overrides are `--net-debt`, or `--gross-debt` together with
`--cash`; use `--shares` to override the diluted count. Reported and manual
amounts must use the financial statements' currency and unscaled base units.

Future buybacks can be modeled as an equity-distribution schedule. The engine
subtracts the present value of the cash spent and retires shares at the modeled
execution price, then reports the value attributable to remaining holders:

```json
{
  "valuation": {
    "share_repurchases": {
      "annual_cash_amounts": ["700000000", "700000000"],
      "initial_purchase_price": null,
      "price_growth_rate": null,
      "discount_rate": null,
      "source": "Published capital-return plan"
    }
  }
}
```

With a null initial price, the valuation's own per-share value is used. Null
price growth and discount rates both resolve to cost of equity (or WACC when a
cost-of-equity assumption is unavailable), so fair-value repurchases do not
create artificial accretion. Set an explicit price or price-growth assumption
to measure buying below or above intrinsic value. CLI overrides are
`--buyback-cash` (repeat per year), `--buyback-price`,
`--buyback-price-growth`, and `--buyback-discount-rate`; use `--no-buybacks` to
disable a profile schedule.

Use `--cash-flow-timing mid_year` for mid-year explicit FCFF discounting.
Terminal value remains at the end of the final forecast year. An exit-multiple
terminal value is available as a separate method and should be treated as a
market cross-check:

```bash
uv run edgarito valuation --ticker AAPL --wacc 8 \
  --terminal-method exit_multiple --exit-multiple 12 --exit-metric ebitda
```

The result reports every cash flow, discount period and factor, terminal-value
contribution, enterprise/equity bridge, input sources, and sensitivity warnings.

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

The simplified `operating cash flow - capital expenditures` history and forecast
are deliberately not marked as FCFF. `valuation-models` assesses readiness but
does not execute the forecast; enterprise DCF becomes ready only when the
driver-based FCFF result and the remaining valuation inputs are supplied.

## Select peers and compute LTM multiples

`comparables` ranks an explicit candidate universe and computes target and peer
multiples from Yahoo's keyless company metadata, quarterly statements, and
latest close:

```bash
uv run edgarito comparables --ticker GOOG \
  --peer META \
  --peer NFLX \
  --peer MSFT \
  --peer AMZN \
  --peer AAPL
```

Candidate selection follows company economics rather than sector alone. The
score considers normalized industry, sector, business archetype, lifecycle,
cyclicality, country, exchange, and revenue scale. Cross-sector candidates are
excluded by default; use `--allow-cross-sector` when a business-model peer sits
in another official sector. `--minimum-score`, `--max-peers`, and
`--preferred-minimum` control selection. Fewer than five selected peers is
reported as a limitation, not silently accepted as a strong sample.

The supplied symbols are the candidate universe, not guaranteed peers. Broad
market discovery remains separate because the configured providers do not
currently offer a reliable free full-market screener.

LTM denominators require four consecutive fiscal quarters. The calculation
uses:

```text
market capitalization = latest close × latest reported shares
enterprise value       = market capitalization + reported debt - cash
LTM EBITDA              = LTM operating income + LTM D&A
LTM FCF                 = LTM operating cash flow - LTM capex
```

The report includes P/E, P/B, P/tangible book, EV/revenue, EV/EBIT,
EV/EBITDA, EV/FCF, and dividend yield, plus peer medians, ranges, and sample
sizes. Negative denominators are marked not meaningful; missing inputs remain
unavailable. Market and reporting currencies must match unless an FX alignment
is supplied in a future valuation layer. REIT AFFO, resource NAV, biotech
pipeline, and SOTP denominators remain part of the later specialized extractors.

Programmatically, use `PeerUniverseSelector`, `LtmMultiplesService`, and
`ComparableMultiplesService` from `edgarito.services.valuation`.

## Extract specialized valuation inputs

SEC Company Facts can supply part of the data needed by specialized valuation
profiles. The `specialized-inputs` command extracts supported standard facts,
retains accession and source-concept provenance, derives only clearly labeled
proxies, and reports what is still missing:

```bash
uv run edgarito specialized-inputs --ticker PLD --type reit
uv run edgarito specialized-inputs --ticker XOM --type resource
uv run edgarito specialized-inputs --ticker MRNA --type biotech
uv run edgarito specialized-inputs --ticker JNJ --type sotp
```

Use `--history` to select the number of latest annual or interim period ends. The extractors
currently provide:

- REIT/property: net income, reported D&A, property-sale gains, impairment, and
  a traceable FFO proxy. It is not mislabeled as NAREIT FFO or AFFO; recurring
  capex, straight-line rent, leasing costs, and company-specific adjustments
  remain required.
- Resources: exploration expense, capitalized exploratory well costs, additions,
  depreciation/depletion/amortization, asset-retirement obligations when
  standardized, and corporate capex.
  Reserve quantities, production profiles, commodity scenarios, asset costs,
  and closure timing remain required for NAV.
- Biotech: reported R&D, acquired in-process R&D when standardized, and cash.
  Named programs, indications, phases, probabilities, launch assumptions, peak
  sales, exclusivity, and program costs remain required for pipeline rNPV.
- SOTP: standardized reportable-segment counts and consolidated segment totals
  when available. Named dimensional segment revenue, profit, assets, capex,
  eliminations, and segment assumptions remain required.

These limitations are structural: SEC Company Facts generally omits the custom
taxonomy and dimensional members contained in filing tables and narrative. Each
report therefore returns `partial` or `blocked` readiness until those inputs are
supplied by a future filing-table extractor or an explicit supplemental dataset.

Programmatic entry points are `ReitInputExtractor`, `ResourceInputExtractor`,
`BiotechInputExtractor`, `SotpInputExtractor`, and
`SpecializedValuationExtractor`.

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

The FCFF DCF resolver applies this precedence:

1. CLI override.
2. Selected profile value.
3. Provider-backed or company-derived value.
4. A specific error naming the unresolved input and its manual profile field.

Automatically derived WACC uses CAPM plus country risk, Hamada relevering of the
industry beta, the latest Yahoo market capitalization, gross debt as the debt
market-value proxy, and an after-tax historical cost of debt. Terminal growth is
bounded below WACC and retains the macro series and methodology in the returned
`ValuationAssumptionSet`.

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
us_available_providers=sec,alphavantage,fmp,yahoo
eu_default_provider=yahoo
eu_available_providers=yahoo,alphavantage,fmp
classification_default_provider=fmp
classification_available_providers=fmp,alphavantage
```

Add any of these lowercase keys to `.env` to override the corresponding default. A default provider must appear in its corresponding available-provider list. SEC supports US financial statements; Alpha Vantage and FMP support financial statements and classifications for both US and EU stocks. Yahoo supports keyless US and EU financial statements but is not used for classification.

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
       or FcffForecastService -> ForecastConsolePresenter
       or SimplifiedFcfForecastService -> ForecastConsolePresenter
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
