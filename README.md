# Edgarito

Edgarito is a Python toolkit and CLI for retrieving, normalizing, reconciling,
and valuing public-company financial data. It keeps provider retrieval,
normalization, forecast assumptions, and valuation mathematics separate so
results remain traceable and auditable.

## Capabilities

- Retrieve financial statements from SEC EDGAR, Yahoo Finance, Alpha Vantage,
  and Financial Modeling Prep.
- Normalize provider data into shared financial concepts and optionally
  crosscheck sources.
- Resolve tickers, CIKs, ISINs, exchanges, and provider-specific symbols.
- Calculate historical operating, profitability, reinvestment, and valuation
  metrics.
- Build driver-based FCFF forecasts with fiscal-period and point-in-time data
  handling.
- Run FCFF DCF, relative, equity, dividend, residual-income, NAV, REIT,
  resource, and pipeline valuation workflows where inputs are available.
- Discover and assess comparable companies without blending independent
  valuation methods into a single opaque result.
- Optionally extract explicit numerical management guidance from SEC 8-K/6-K
  filings using OpenAI Structured Outputs, with source-evidence verification,
  deterministic validation, and cached normalized results.

## Quick start

```bash
uv sync
cp .env.example .env
uv run edgarito financials --ticker AAPL
uv run edgarito metrics --ticker AAPL
uv run edgarito forecast --ticker AAPL
uv run edgarito valuation --ticker AAPL
```

SEC requests require an identifying user agent in `.env`:

```dotenv
cache_path=./cache
user_agent="Your Name (your-email@example.com)"
```

Other provider keys are optional unless that provider is selected. Yahoo is
keyless. OpenAI management-guidance extraction is also optional:

```dotenv
openai_secret_api_key="your-api-key"
OPENAI_MODEL="gpt-5.6-luna"
OPENAI_REASONING_EFFORT="low"
```

Without an OpenAI key, valuation follows the existing deterministic forecast
path and makes no OpenAI requests.

## Common workflows

```bash
# Select a provider or market.
uv run edgarito financials --ticker SAP.DE --market eu --provider yahoo

# Crosscheck available providers.
uv run edgarito financials --ticker AAPL --crosscheck

# Audit forecast and valuation provenance.
uv run edgarito valuation --ticker ASML --market eu --audit

# Inspect model suitability before valuation.
uv run edgarito valuation-models --ticker JPM

# Build a peer-multiple report.
uv run edgarito comparables --ticker AAPL
```

## Documentation

The [detailed guide](docs/guide.md) covers configuration, identifiers,
providers, caching, normalized schemas, forecasting, valuation profiles,
discount rates, decision analysis, comparable companies, specialized models,
programmatic usage, and development checks.

## Development

```bash
uv run ruff check edgarito tests
uv run pytest -q
```

Edgarito is licensed under the [GNU General Public License v3.0](LICENSE).
