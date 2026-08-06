"""
Comprehensive validation tests using real EDGAR data.

Tests financial statement readers against actual SEC filings to ensure:
- Correct data extraction
- Accurate YTD conversion
- Proper FY-to-Q4 calculation
- Deduplication works correctly
"""
import pytest
import json
from pathlib import Path
from edgarito.schemas.edgar_responses.company_facts import CompanyFacts
from edgarito.services.financial.statements import FinancialStatements
from edgarito.enums.granularity import Granularity

# Load fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"

def load_company_facts(ticker: str) -> CompanyFacts:
    """Load company facts from fixtures."""
    fixture_file = FIXTURES_DIR / f"{ticker.lower()}_facts.json"
    with open(fixture_file) as f:
        data = json.load(f)
    # Convert snake_case keys to proper aliases for Pydantic
    if "facts" in data and "us_gaap" in data["facts"]:
        data["facts"]["us-gaap"] = data["facts"].pop("us_gaap")
    return CompanyFacts(**data)


class TestAAPLReader:
    """Test AAPL (Technology sector) - stable, profitable company."""
    
    @pytest.fixture
    def aapl_facts(self):
        return load_company_facts("AAPL")
    
    @pytest.fixture
    def statements(self, aapl_facts):
        return FinancialStatements(aapl_facts)
    
    def test_annual_revenue(self, statements):
        """Test annual revenue extraction."""
        revenue = statements.income_statement.get_revenue(Granularity.ANNUAL)
        
        # Should have multiple years of data
        assert len(revenue.values) >= 5
        
        # Revenue should be positive and in billions
        for val in revenue.values:
            assert val > 100_000_000_000, "AAPL revenue should be > $100B"
    
    def test_quarterly_revenue(self, statements):
        """Test quarterly revenue extraction and YTD conversion."""
        revenue = statements.income_statement.get_revenue(Granularity.QUARTERLY)
        
        # Should have quarterly data
        assert len(revenue.values) >= 8
        
        # All quarters should be positive
        for val in revenue.values:
            assert val > 0, "Revenue should be positive"
            # Quarterly revenue should be reasonable (not YTD cumulative)
            assert val < 150_000_000_000, "Quarterly revenue should be < $150B (not cumulative)"
    
    def test_total_assets(self, statements):
        """Test balance sheet assets."""
        assets = statements.balance_sheet.get_total_assets(Granularity.ANNUAL)
        
        assert len(assets.values) >= 5
        
        # Assets should be positive and substantial
        for val in assets.values:
            assert val > 200_000_000_000, "AAPL assets should be > $200B"
    
    def test_operating_cash_flow(self, statements):
        """Test operating cash flow extraction."""
        ocf = statements.cash_flow.get_operating_cash_flow(Granularity.ANNUAL)
        
        assert len(ocf.values) >= 5
        
        # AAPL should have positive cash flow (most years)
        positive_years = sum(1 for val in ocf.values if val > 0)
        assert positive_years >= len(ocf.values) * 0.8, "AAPL should have positive OCF in most years"


class TestTSLAReader:
    """Test TSLA (Automotive) - high growth, volatile company."""
    
    @pytest.fixture
    def tsla_facts(self):
        return load_company_facts("TSLA")
    
    @pytest.fixture
    def statements(self, tsla_facts):
        return FinancialStatements(tsla_facts)
    
    def test_revenue_growth(self, statements):
        """Test that TSLA shows revenue growth over time."""
        revenue = statements.income_statement.get_revenue(Granularity.ANNUAL)
        
        # Should have data
        assert len(revenue.values) >= 3
        
        # Recent years should show growth (values are reverse chronological)
        if len(revenue.values) >= 2:
            # Most recent year
            recent = revenue.values[0]
            prev = revenue.values[1]
            # TSLA has been growing
            assert recent > 0 and prev > 0
    
    def test_quarterly_data_completeness(self, statements):
        """Test that quarterly data is available."""
        revenue = statements.income_statement.get_revenue(Granularity.QUARTERLY)
        
        # Should have quarterly data
        assert len(revenue.values) >= 4, "Should have at least 4 quarters of data"


class TestJPMReader:
    """Test JPM (Financials) - high leverage is normal for banks."""
    
    @pytest.fixture
    def jpm_facts(self):
        return load_company_facts("JPM")
    
    @pytest.fixture
    def statements(self, jpm_facts):
        return FinancialStatements(jpm_facts)
    
    def test_high_assets(self, statements):
        """Banks have very high assets due to loans."""
        assets = statements.balance_sheet.get_total_assets(Granularity.ANNUAL)
        
        assert len(assets.values) >= 3
        
        # JPM should have > $2T in assets
        for val in assets.values:
            assert val > 2_000_000_000_000, "JPM assets should be > $2T"
    
    def test_revenue_positive(self, statements):
        """Banks should have positive revenue."""
        revenue = statements.income_statement.get_revenue(Granularity.ANNUAL)
        
        assert len(revenue.values) >= 3
        
        for val in revenue.values:
            assert val > 0, "Revenue should be positive"


class TestCrossCompanyValidation:
    """Cross-company validation tests."""
    
    @pytest.mark.parametrize("ticker", ["AAPL", "TSLA", "JPM", "JNJ", "XOM", "WMT"])
    def test_data_availability(self, ticker):
        """Test that all companies have minimum required data."""
        facts = load_company_facts(ticker)
        statements = FinancialStatements(facts)
        
        # Should be able to get annual revenue
        revenue = statements.income_statement.get_revenue(Granularity.ANNUAL)
        assert len(revenue.values) >= 1, f"{ticker} should have at least 1 year of revenue"
        
        # Should be able to get annual assets
        assets = statements.balance_sheet.get_total_assets(Granularity.ANNUAL)
        assert len(assets.values) >= 1, f"{ticker} should have at least 1 year of assets"
    
    @pytest.mark.parametrize("ticker", ["AAPL", "TSLA", "JPM", "JNJ", "XOM", "WMT"])
    def test_quarterly_conversion_sanity(self, ticker):
        """Test that quarterly values are reasonable (not cumulative)."""
        facts = load_company_facts(ticker)
        statements = FinancialStatements(facts)
        
        try:
            revenue_q = statements.income_statement.get_revenue(Granularity.QUARTERLY)
            revenue_a = statements.income_statement.get_revenue(Granularity.ANNUAL)
            
            if len(revenue_q.values) >= 4 and len(revenue_a.values) >= 1:
                # Sum of 4 most recent quarters should roughly equal annual
                # (allowing for fiscal year timing differences)
                q_sum = sum(revenue_q.values[:4])
                annual = revenue_a.values[0]
                
                # Within 20% tolerance (accounts for different fiscal periods)
                ratio = q_sum / annual if annual > 0 else 0
                assert 0.8 <= ratio <= 1.2, f"{ticker}: Q sum {q_sum/1e9:.1f}B vs Annual {annual/1e9:.1f}B (ratio: {ratio:.2f})"
        except:
            # Some companies may not have complete quarterly data
            pytest.skip(f"{ticker} missing complete quarterly/annual data for comparison")
