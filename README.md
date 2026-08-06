# Edgarito

Edgarito retrieves SEC EDGAR Company Facts, stores the raw provider responses in a filesystem cache, normalizes selected US-GAAP concepts into provider-neutral observations, and displays historical financial statements in a CLI.

## Install

```bash
pip install -e .
```

SEC requests require a descriptive user agent containing contact information:

```bash
export EDGARITO_USER_AGENT="Your Name your-email@example.com"
```

PowerShell:

```powershell
$env:EDGARITO_USER_AGENT = "Your Name your-email@example.com"
```

## Display financials

```bash
edgarito financials --ticker AAPL
edgarito financials --ticker AAPL --period quarterly --limit 8
edgarito financials --cik 320193 --period all
edgarito financials --ticker AAPL --concept revenue --concept net_income
```

The same command works without installing the console script:

```bash
python -m edgarito financials --ticker AAPL --user-agent "Your Name your-email@example.com"
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
