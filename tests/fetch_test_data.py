"""
Script to fetch real EDGAR data for testing purposes.
This creates fixtures from actual SEC data for reproducible tests.
"""
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import aiohttp
import json
from edgarito.services.retrieval.edgar_rest_client.low_level_client import EDGARLowLevelClient
from edgarito.services.cache.filesystem_cache import FileSystemCache

# Test companies across different sectors
TEST_COMPANIES = {
    "AAPL": {"sector": "Technology", "description": "Stable, profitable tech giant"},
    "TSLA": {"sector": "Automotive", "description": "High growth, volatile EV maker"},
    "JPM": {"sector": "Financials", "description": "Major bank with high leverage"},
    "JNJ": {"sector": "Healthcare", "description": "Defensive pharmaceutical company"},
    "XOM": {"sector": "Energy", "description": "Cyclical energy company"},
    "WMT": {"sector": "Retail", "description": "Low margin retail giant"},
}

async def fetch_company_data(ticker: str):
    """Fetch company facts from EDGAR API."""
    cache = FileSystemCache(root_directory="./cache")
    
    async with EDGARLowLevelClient(
        cache=cache,
        user_agent="edgarito-test-suite (test@example.com)"
    ) as edgar:
        # Get CIK from ticker
        print(f"Fetching {ticker}...")
        tickers = await edgar.get_tickers()
        company = next((t for t in tickers if t.ticker == ticker), None)
        
        if not company:
            print(f"  ❌ Ticker {ticker} not found")
            return None
        
        # Get company facts
        facts = await edgar.get_company_facts(cik=company.cik)
        
        # Save to fixtures
        fixtures_dir = Path("tests/fixtures")
        fixtures_dir.mkdir(exist_ok=True)
        
        output_file = fixtures_dir / f"{ticker.lower()}_facts.json"
        with open(output_file, "w") as f:
            json.dump(facts.model_dump(mode='json'), f, indent=2, default=str)
        
        print(f"  ✓ Saved to {output_file}")
        return facts

async def main():
    """Fetch all test company data."""
    print("Fetching test data from EDGAR API...")
    print("=" * 60)
    
    for ticker, info in TEST_COMPANIES.items():
        print(f"\n{ticker} ({info['sector']})")
        print(f"  {info['description']}")
        await fetch_company_data(ticker)
    
    print("\n" + "=" * 60)
    print("✓ Test data collection complete")

if __name__ == "__main__":
    asyncio.run(main())
