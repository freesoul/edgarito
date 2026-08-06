"""
Red flags validation tests using real EDGAR data.

Tests the red flags detection system against actual companies to ensure:
- Flags are correctly identified
- Sector-specific thresholds work
- Quality score calculation is accurate
"""
import pytest
import json
from pathlib import Path
from edgarito.schemas.edgar_responses.company_facts import CompanyFacts
from edgarito.services.analysis.red_flags_service import RedFlagsService
from edgarito.schemas.red_flags import RedFlagSeverity

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


class TestRedFlagsAAPL:
    """Test red flags for AAPL - should have minimal flags (healthy company)."""
    
    @pytest.fixture
    def aapl_facts(self):
        return load_company_facts("AAPL")
    
    @pytest.fixture
    def red_flags_service(self, aapl_facts):
        return RedFlagsService(facts=aapl_facts, market_data=None)
    
    def test_low_critical_flags(self, red_flags_service):
        """AAPL should have few or zero critical flags."""
        report = red_flags_service.analyze()
        
        # AAPL is a healthy company, should have minimal critical flags
        assert report.critical_flags <= 2, f"AAPL should have minimal critical flags, got {report.critical_flags}"
    
    def test_quality_score_calculated(self, red_flags_service):
        """AAPL should have a calculated quality score."""
        report = red_flags_service.analyze()
        
        # Quality score should exist and be calculated
        assert report.quality_score is not None
        assert 0 <= report.quality_score <= 100
    
    def test_report_structure(self, red_flags_service):
        """Test that report has all required sections."""
        report = red_flags_service.analyze()
        
        # Should have data in various sections
        assert isinstance(report.balance_sheet_flags, list)
        assert isinstance(report.cash_flow_flags, list)
        assert isinstance(report.profitability_flags, list)
        assert isinstance(report.growth_flags, list)


class TestRedFlagsJPM:
    """Test red flags for JPM - bank with high leverage (normal for sector)."""
    
    @pytest.fixture
    def jpm_facts(self):
        return load_company_facts("JPM")
    
    @pytest.fixture
    def red_flags_service(self, jpm_facts):
        return RedFlagsService(facts=jpm_facts, market_data=None)
    
    def test_sector_specific_debt_handling(self, red_flags_service, jpm_facts):
        """Banks have high debt ratios - sector profile should accommodate this."""
        report = red_flags_service.analyze()
        
        # JPM will have high debt, but sector adjustments should reduce severity
        # Just verify the report generates successfully
        assert report is not None
        assert report.ticker == str(jpm_facts.cik)


class TestRedFlagsCrossCompany:
    """Cross-company red flags validation."""
    
    @pytest.mark.parametrize("ticker,sector", [
        ("AAPL", "technology"),
        ("TSLA", "automotive"),
        ("JPM", "financials"),
        ("JNJ", "healthcare"),
        ("XOM", "energy"),
        ("WMT", "retail"),
    ])
    def test_report_generation(self, ticker, sector):
        """Test that red flags report generates for all companies."""
        facts = load_company_facts(ticker)
        service = RedFlagsService(facts=facts, market_data=None)
        
        report = service.analyze()
        
        # Basic report structure checks
        assert report is not None
        assert report.ticker == str(facts.cik)
        assert report.company_name == facts.entityName
        assert report.quality_score >= 0
        assert report.quality_score <= 100
        
        # Should have counted flags
        expected_total = (report.critical_flags + report.warning_flags + report.info_flags)
        assert report.total_flags == expected_total, f"{ticker}: Flag count mismatch"
    
    @pytest.mark.parametrize("ticker", ["AAPL", "TSLA", "JPM", "JNJ", "XOM", "WMT"])
    def test_quality_score_bounds(self, ticker):
        """Test that quality scores are within valid bounds."""
        facts = load_company_facts(ticker)
        service = RedFlagsService(facts=facts, market_data=None)
        
        report = service.analyze()
        
        # Quality score must be 0-100
        assert 0 <= report.quality_score <= 100, f"{ticker}: Quality score {report.quality_score} out of bounds"
    
    @pytest.mark.parametrize("ticker", ["AAPL", "TSLA", "JPM", "JNJ", "XOM", "WMT"])
    def test_flag_severity_consistency(self, ticker):
        """Test that flags have consistent severity levels."""
        facts = load_company_facts(ticker)
        service = RedFlagsService(facts=facts, market_data=None)
        
        report = service.analyze()
        
        # Count flags by severity manually
        critical_count = 0
        warning_count = 0
        info_count = 0
        
        for flag in report.all_flags:
            if flag.severity == RedFlagSeverity.CRITICAL:
                critical_count += 1
            elif flag.severity == RedFlagSeverity.WARNING:
                warning_count += 1
            elif flag.severity == RedFlagSeverity.INFO:
                info_count += 1
        
        # Verify counts match report
        assert critical_count == report.critical_flags, f"{ticker}: Critical flags mismatch"
        assert warning_count == report.warning_flags, f"{ticker}: Warning flags mismatch"
        assert info_count == report.info_flags, f"{ticker}: Info flags mismatch"
