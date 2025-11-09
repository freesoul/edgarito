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
# Detect potential accounting issues and financial risks
# Automatically uses quarterly data for recent trends and annual data for long-term patterns
python -m edgarito.cli redflags --ticker TSLA

# The analysis intelligently combines:
# - Quarterly: Current liquidity, recent margins, latest cash flow
# - Annual: 5-year revenue CAGR, long-term profitability trends

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

Automated analysis identifies potential issues with **intelligent data selection** and **sector-aware thresholds**:

**Intelligent Granularity:**
- **Balance Sheet & Cash Flow**: Uses quarterly data for current liquidity and solvency
- **Profitability**: Uses quarterly data for recent margins and performance
- **Growth**: Uses annual data for long-term trends (5-year CAGR)
- **Valuation**: Uses quarterly data for current market multiples

**Sector-Aware Analysis:**
- Automatically adjusts thresholds based on company sector (12 sector profiles)
- Example: Utilities can have 5x higher debt ratios than tech companies
- Descriptions include sector context: "High debt for **technology sector**"

**Red Flag Categories:**

**Balance Sheet Health:**
- High debt-to-equity ratios (tiered: warning/critical)
- Liquidity concerns: current ratio, quick ratio (tiered)
- Interest coverage concerns (tiered)
- Negative tangible book value

**Cash Flow Quality:**
- Operating cash flow vs Free Cash flow concerns (CapEx-aware)
- High stock-based compensation (>10% of OCF)
- Frequent share issuance (dilution)

**Profitability & Income Quality:**
- Declining or volatile gross margins
- Low net margins (<3%)
- Poor capital efficiency: ROE (<10%), ROIC (<7%)

**Growth & Sustainability:**
- Revenue growth below inflation (<3% CAGR)
- High or rising SG&A expenses (>25% info, >30% warning)
- Declining R&D investment

**Valuation Concerns** (with Yahoo Finance integration):
- High P/S (>10), P/E (<5), P/B (>5) ratios
- High PEG ratio (>2.0) - overpriced growth
- Elevated EV/EBITDA (>15) multiples
- High short interest (>10% of float)
- Low insider ownership (<2%)

## Configuration

### Basic Configuration

Edgarito can be configured via a `.cli.env` file in your project root:

```properties
# Required
cache_path=./cache
user_agent=Your Name (your-email@example.com)
```

### Red Flags Configuration

Red flag thresholds are configured in `settings/red_flags.yaml` with **comprehensive inline documentation**:

```yaml
# Balance Sheet Health - Debt and Leverage
# Debt to Equity Ratio (Total Debt / Shareholders' Equity)
# Higher debt increases financial risk and interest burden
debt_to_equity_ratio_warning: 1.0   # WARNING if D/E > 1.0 (debt equals equity)
debt_to_equity_ratio_critical: 2.0  # CRITICAL if D/E > 2.0 (debt is 2x equity)

# Liquidity - Current Ratio (Current Assets / Current Liabilities)
# Measures ability to pay short-term obligations
current_ratio_critical: 1.0         # CRITICAL if < 1.0 (can't cover current liabilities)
current_ratio_warning: 1.5          # WARNING if < 1.5 (tight liquidity buffer)

# Cash Flow Quality
# Stock-Based Compensation as % of Operating Cash Flow
stock_comp_percent_ocf: 10.0        # INFO if > 10% (high dilution)

# Profitability Metrics
net_margin_percent: 3.0             # INFO if < 3% (weak profitability)
roe_percent: 10.0                   # INFO if < 10% (poor returns on equity)

# Valuation Concerns
price_to_sales: 10.0                # INFO if > 10 (expensive relative to sales)
peg_ratio: 2.0                      # INFO if > 2.0 (overpriced growth)
```

**Configuration Files Structure:**

```
settings/
├── red_flags.yaml       # 23 thresholds with full documentation
└── sector_profiles.yaml # Industry-specific adjustments (12 sectors)
```

**Customization Examples:**

```yaml
# Conservative investor - stricter thresholds
debt_to_equity_ratio_warning: 0.5   # Flag any company with D/E > 0.5
current_ratio_warning: 2.0          # Want strong liquidity buffer
roe_percent: 15.0                   # Expect excellent returns

# Growth investor - more lenient on profitability
net_margin_percent: 0.0             # Accept unprofitable growth companies
roe_percent: 5.0                    # Lower return expectations
price_to_sales: 20.0                # Accept higher valuations
```

### Industry-Aware Thresholds

Red flags automatically adjust based on company sector via `settings/sector_profiles.yaml`:

**Example adjustments:**
- **Utilities**: 5.0x debt multiplier (capital-intensive, stable cash flows)
- **Technology**: 1.125x OCF multiplier (light assets, high margins)
- **Financial Services**: 10.0x debt multiplier (leverage is part of business model)
- **Real Estate**: 0.714x FCF multiplier (capital-intensive, depreciation-heavy)

**How it works:**

```bash
# Technology company with D/E = 0.8
# Base threshold: 1.0 (WARNING)
# Tech multiplier: 0.8x → Adjusted threshold: 0.8
# Result: ⚠️ WARNING "High debt for technology sector"

# Utility company with D/E = 3.0
# Base threshold: 1.0 (WARNING)
# Utility multiplier: 5.0x → Adjusted threshold: 5.0  
# Result: ✅ No flag (normal for utilities)
```

CLI output automatically displays sector context:

```
====================================================================================================
RED FLAGS ANALYSIS: AAPL - Apple Inc.
Sector: Technology | Industry: Consumer Electronics
====================================================================================================
```

**Advanced:** You can override thresholds via environment variables using `red_flags__` prefix:

```properties
red_flags__debt_to_equity_ratio_warning=1.5
red_flags__current_ratio_critical=0.8
```

Environment variables take precedence over YAML configuration.

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