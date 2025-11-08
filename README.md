# Edgarito

**A comprehensive Python toolkit for SEC EDGAR data analysis and financial statement extraction.**

Edgarito provides an async client for the SEC EDGAR REST API with structured Pydantic schemas, powerful financial statement readers with robust data deduplication, computed financial metrics, and automated red flag detection for investment analysis.

## Features

- 🚀 **Async EDGAR REST API Client** - Fast, cached access to SEC filings data
- 📊 **Financial Statement Readers** - Extract and normalize data from Balance Sheets, Income Statements, and Cash Flow Statements
- 🔍 **Smart Data Handling** - Robust deduplication of repeated filings, amendments, and year-over-year comparatives
- 📈 **Financial Metrics** - Compute ratios like FCF, ROE, profit margins, asset turnover, and more
- ⚠️ **Red Flag Detection** - 25+ automated checks across balance sheet, cash flow, profitability, growth, and valuation
- 🌐 **Market Data Integration** - Yahoo Finance integration for valuation metrics and market sentiment
- 💾 **Intelligent Caching** - FileSystem cache to comply with SEC rate limits and optimize performance
- 🎯 **Clean Data Models** - All responses structured with Pydantic schemas for type safety
- 🖥️ **Rich CLI** - Command-line interface for quick financial data exploration

## Installation

```bash
pip install edgarito
```

## Quick Start - CLI

The CLI provides instant access to financial data and analysis:

### View Financial Statements

```bash
# Display comprehensive financial statements (annual and quarterly)
python -m edgarito.cli financials --ticker AAPL

# Output shows:
# - Total Assets
# - Revenue  
# - Operating Income
# - Net Income
# - Operating Cash Flow
```

### Analyze Red Flags

```bash
# Detect potential accounting issues and financial risks (quarterly recommended for recent data)
python -m edgarito.cli redflags --ticker TSLA --granularity quarterly

# Use annual for longer-term trend analysis
python -m edgarito.cli redflags --ticker AAPL --granularity annual

# Analyzes 30+ red flags across 5 categories:
# - Balance Sheet Health (debt, liquidity, asset quality)
# - Cash Flow Quality (FCF, dividends, dilution)
# - Profitability & Income Quality (margins, returns)
# - Growth & Sustainability (revenue trends, R&D, expenses)
# - Valuation Concerns (market multiples, short interest, insider ownership)
```

### Search and Download Filings

```bash
# Find company CIK from ticker
python -m edgarito.cli cik --ticker AAPL

# List all submissions for a company
python -m edgarito.cli submissions --ticker AAPL --type 10-K

# Download filings
python -m edgarito.cli download --ticker AAPL --type 10-K --limit 5
```

### Other Commands

```bash
# Resolve ticker from CIK
python -m edgarito.cli ticker --cik 320193

# List all available company tickers
python -m edgarito.cli tickers
```

## Quick Start - Python API

### Financial Statements Reader

```python
from edgarito.services.financial.statements import FinancialStatements
from edgarito.services.retrieval.company_data_loader import CompanyDataLoader
from edgarito.enums.granularity import Granularity
import aiohttp

async def analyze_company(ticker: str):
    # Load company data
    async with aiohttp.ClientSession() as session:
        loader = CompanyDataLoader(
            session,
            cache_dir="./cache",
            user_agent="Your Name (your-email@example.com)"
        )
        result = await loader.load_from_ticker(ticker)
    
    # Access financial statements
    statements = FinancialStatements(result.facts)
    
    # Get quarterly data
    revenue = statements.income_statement.get_revenue(Granularity.QUARTERLY)
    net_income = statements.income_statement.get_net_income(Granularity.QUARTERLY)
    op_income = statements.income_statement.get_operating_income(Granularity.QUARTERLY)
    
    # Get annual data
    assets = statements.balance_sheet.get_total_assets(Granularity.ANNUAL)
    cash_flow = statements.cash_flow.get_operating_cash_flow(Granularity.ANNUAL)
    
    # Compute metrics
    revenue_aligned = revenue.intersect(assets)
    assets_aligned = assets.intersect(revenue)
    asset_turnover = statements.metrics.asset_turnover(revenue_aligned, assets_aligned)
    
    # Access period data
    for period, value in zip(revenue.periods, revenue.values):
        print(f"{period.fiscal_year} {period.fiscal_period}: ${value / 1_000_000:.1f}M")

# Run
import asyncio
asyncio.run(analyze_company("AAPL"))
```

### Low-Level EDGAR REST API

```python
from edgarito.services.retrieval.edgar_rest_client.low_level_client import EDGARLowLevelClient
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.enums.edgar.core_filing_type import CoreFilingType

async def main():
    # Create cache (required for SEC compliance)
    cache = FileSystemCache(root_directory="./cache")

    # Create EDGAR client
    async with EDGARLowLevelClient(
        cache=cache, 
        user_agent="Your Name (your-email@example.com)"
    ) as edgar:
        
        # Get company tickers
        tickers = await edgar.get_tickers()
        aapl_cik = next(t.cik for t in tickers if t.ticker == "AAPL")
        
        # Get company facts (US-GAAP data)
        company = await edgar.get_company_facts(cik=aapl_cik)
        
        # Access specific facts
        for fact_name, fact_info in company.facts.us_gaap.items():
            if fact_info.units.USD:
                print(f"{fact_name} - {fact_info.label}:")
                for measurement in fact_info.units.USD:
                    if measurement.parsed_type == CoreFilingType.FILING_10K:
                        print(f"  FY{measurement.fy}: ${measurement.val:,.0f}")
        
        # Get submissions/filings
        submissions = await edgar.get_submissions(cik=aapl_cik)
        recent_10k = [f for f in submissions.filings.recent 
                      if f.form == "10-K"][0]
        print(f"Latest 10-K: {recent_10k.filingDate}")

import asyncio
asyncio.run(main())
```

## Key Features in Detail

### Robust Data Handling

Edgarito handles real-world data quality issues automatically:

- **Deduplication**: Removes duplicate measurements from amendments and restatements
- **YTD Conversion**: Converts year-to-date cumulative values to individual quarters
- **Q4 Calculation**: Derives Q4 data from full-year values when not directly reported
- **Fiscal Year Normalization**: Handles different fiscal year-end conventions
- **Multi-Concept Fallbacks**: Tries alternative US-GAAP concepts when primary ones are unavailable

### Financial Statement Readers

Three specialized readers provide clean access to financial data:

1. **BalanceSheetReader** - Assets, liabilities, equity, working capital
2. **IncomeStatementReader** - Revenue, operating income, net income, EPS
3. **CashFlowStatementReader** - Operating, investing, financing cash flows

### Computed Metrics

The `FinancialMetrics` class calculates key ratios:

- Profitability: ROE, ROA, profit margins, ROIC
- Efficiency: Asset turnover, inventory turnover, days ratios
- Liquidity: Current ratio, quick ratio, cash ratio
- Leverage: Debt ratios, interest coverage
- Valuation: P/E, P/B, EV/EBITDA (with market data)
- Cash Flow: Free cash flow, FCF yield, cash conversion

### Red Flag Detection

Automated analysis identifies potential issues across multiple categories:

**Balance Sheet Health:**
- High debt-to-equity ratios
- Liquidity concerns (current/quick ratios)
- Negative tangible book value
- Rising liabilities vs assets

**Cash Flow Quality:**
- Operating cash flow < Net Income
- Negative free cash flow
- Unsustainable dividends
- High stock-based compensation
- Frequent share issuance

**Profitability & Income Quality:**
- Declining gross/operating margins
- Low net margins
- Poor capital efficiency (ROE, ROIC)
- Volatile earnings patterns

**Growth & Sustainability:**
- Revenue growth below inflation
- High or rising SG&A expenses
- Declining R&D investment

**Valuation Concerns** (with Yahoo Finance integration):
- High P/S, P/E, P/B ratios
- Excessive dividend yields
- High PEG ratios (overpriced growth)
- Elevated EV/EBITDA multiples
- High short interest (>10%)
- Low insider ownership (<2%)

## Configuration

Edgarito can be configured via a `.cli.env` file in your project root:

```properties
# Required
cache_path=./cache
user_agent=Your Name (your-email@example.com)

# Optional
taxonomy_url=https://xbrl.fasb.org/us-gaap/2025/elts/us-gaap-2025.xsd
```

### Customizing Red Flags Thresholds

All red flag thresholds are configurable via environment variables using the `red_flags__` prefix with double underscore.

**Tiered Severity Levels:** Many thresholds have multiple tiers (critical/warning/info) for more actionable alerts:

```properties
# Balance Sheet - Tiered thresholds
red_flags__debt_to_equity_ratio_warning=1.0       # Warning if D/E > 1.0
red_flags__debt_to_equity_ratio_critical=2.0      # Critical if D/E > 2.0

red_flags__current_ratio_critical=1.0             # Critical if < 1.0
red_flags__current_ratio_warning=1.5              # Warning if < 1.5

red_flags__quick_ratio_critical=0.5               # Critical if < 0.5
red_flags__quick_ratio_warning=1.0                # Warning if < 1.0

red_flags__interest_coverage_critical=1.5         # Critical if < 1.5 (default risk)
red_flags__interest_coverage_warning=3.0          # Warning if < 3.0

# Growth - Tiered thresholds
red_flags__sga_percent_revenue_info=25.0          # Info if SG&A > 25%
red_flags__sga_percent_revenue_warning=30.0       # Warning if SG&A > 30%

# Single-level thresholds (info level)
red_flags__stock_comp_percent_ocf=10.0            # Default: 10.0
red_flags__net_margin_percent=3.0                 # Default: 3.0
red_flags__roe_percent=10.0                       # Default: 10.0
red_flags__roic_percent=7.0                       # Default: 7.0
red_flags__revenue_cagr_inflation=3.0             # Default: 3.0
red_flags__price_to_sales=10.0                    # Default: 10.0
red_flags__peg_ratio=2.0                          # Default: 2.0
red_flags__ev_to_ebitda=15.0                      # Default: 15.0
```

See `.cli.env` for a complete list of all 30+ configurable thresholds with descriptions.

## Data Quality

Edgarito prioritizes data accuracy with:

- **Shortest Duration Preference**: Selects individual quarterly data over YTD cumulative
- **Most Recent Filing**: Prefers latest amendments over original filings
- **Consistent Fiscal Years**: Normalizes fiscal year assignments across all periods
- **Validation**: Checks Q4 calculations against full-year values

## Requirements

- Python 3.8+
- aiohttp
- pydantic
- yfinance (for market data)
- typer (for CLI)

## Project Structure

```
edgarito/
├── cli/                    # Command-line interface
├── enums/                  # Enumerations (filing types, periods, granularity)
├── schemas/               # Pydantic models for API responses
├── services/
│   ├── retrieval/         # EDGAR REST API clients
│   ├── cache/             # FileSystem caching
│   ├── financial/         # Financial statement readers
│   └── analysis/          # Red flags and metrics
└── examples/              # Usage examples
```

## Contributing

Contributions welcome! Areas for improvement:

- Additional financial metrics
- More red flag detection rules  
- Support for IFRS (currently US-GAAP only)
- Direct XBRL parsing from filings (as fallback to REST API)

## License

MIT License - see LICENSE file for details

## Disclaimer

This tool is for informational purposes only. Not financial advice. Always verify data with official SEC filings at https://www.sec.gov/edgar.