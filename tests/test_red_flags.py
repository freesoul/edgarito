import pytest
from edgarito.services.analysis.red_flags_service import RedFlagsService
from edgarito.schemas.red_flags import RedFlagSeverity

def test_red_flags_service_initialization(mock_company_facts, mock_market_data):
    service = RedFlagsService(facts=mock_company_facts, market_data=mock_market_data)
    assert service.sector == "technology"
    assert service.facts.cik == 320193

def test_red_flags_analysis_basic(mock_company_facts, mock_market_data):
    service = RedFlagsService(facts=mock_company_facts, market_data=mock_market_data)
    report = service.analyze()
    
    assert report.ticker == "320193"
    assert report.company_name == "Apple Inc."
    assert report.total_flags >= 0
    assert report.quality_score <= 100.0
