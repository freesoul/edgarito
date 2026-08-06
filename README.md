# Edgarito

Edgarito retrieves SEC EDGAR Company Facts, stores the raw provider responses in a filesystem cache, normalizes selected US-GAAP concepts into provider-neutral observations, and displays historical financial statements in a CLI.

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
```

You can also run the Python module directly:

```bash
uv run python -m edgarito financials --ticker AAPL --user-agent "Your Name your-email@example.com"
```

Provider responses are cached below `cache/providers/edgar/`. Use `--refresh` to bypass existing snapshots. Asterisks in quarterly output mark values derived from reported YTD or full-year facts.

## Current flow

```text
EdgarClient
    -> services/cache
    -> SecUsGaapNormalizer
    -> NormalizedCompanyFinancials
    -> FinancialsConsolePresenter
```

The normalized observations retain source taxonomy, source concept, accession number, filing form, filed date, unit, and derivation metadata so future providers can feed the same schema and be compared under `crosscheck`.
