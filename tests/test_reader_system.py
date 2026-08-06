import pytest
from datetime import date, timedelta
from edgarito.services.financial.base_reader import BaseStatementReader
from edgarito.schemas.edgar_responses.company_facts import Measurement, CompanyFacts, Facts, Fact, FactUnits
from edgarito.enums.edgar.period import FiscalPeriod
from edgarito.enums.edgar.core_filing_type import CoreFilingType
from edgarito.enums.granularity import Granularity

class TestReaderSystem:
    
    @pytest.fixture
    def base_reader(self):
        facts = CompanyFacts(
            cik=123, 
            entityName="Test Corp", 
            facts=Facts(dei={}, us_gaap={})
        )
        return BaseStatementReader(facts)

    def test_deduplicate_measurements_prefer_shorter_duration(self, base_reader):
        """Test that shorter duration (individual quarter) is preferred over YTD."""
        # Q2 Individual (Apr-Jun)
        m1 = Measurement(
            start=date(2023, 4, 1), end=date(2023, 6, 30),
            val=100, accn="1", fy=2023, fp=FiscalPeriod.Q2,
            form="10-Q", filed=date(2023, 8, 1)
        )
        # Q2 YTD (Jan-Jun)
        m2 = Measurement(
            start=date(2023, 1, 1), end=date(2023, 6, 30),
            val=200, accn="2", fy=2023, fp=FiscalPeriod.Q2,
            form="10-Q", filed=date(2023, 8, 1)
        )
        
        deduplicated = base_reader._deduplicate_measurements([m1, m2])
        assert len(deduplicated) == 1
        assert deduplicated[0].val == 100  # Should pick m1 (shorter duration)

    def test_deduplicate_measurements_prefer_newer_filing(self, base_reader):
        """Test that newer filing date is preferred when durations are equal."""
        # Original filing
        m1 = Measurement(
            start=date(2023, 1, 1), end=date(2023, 3, 31),
            val=100, accn="1", fy=2023, fp=FiscalPeriod.Q1,
            form="10-Q", filed=date(2023, 4, 15)
        )
        # Amended/Restated filing (later date)
        m2 = Measurement(
            start=date(2023, 1, 1), end=date(2023, 3, 31),
            val=150, accn="2", fy=2023, fp=FiscalPeriod.Q1,
            form="10-Q/A", filed=date(2023, 5, 1)
        )
        
        deduplicated = base_reader._deduplicate_measurements([m1, m2])
        assert len(deduplicated) == 1
        assert deduplicated[0].val == 150  # Should pick m2 (newer filed date)

    def test_convert_ytd_to_quarterly(self, base_reader):
        """Test conversion of YTD values to individual quarters."""
        # Setup mock data: Q1, Q2 YTD, Q3 YTD
        # Q1: 100
        # Q2 YTD: 250 (so Q2 individual = 150)
        # Q3 YTD: 450 (so Q3 individual = 200)
        
        # Create measurements
        m_q1 = Measurement(
            start=date(2023, 1, 1), end=date(2023, 3, 31),
            val=100, accn="1", fy=2023, fp=FiscalPeriod.Q1,
            form="10-Q", filed=date(2023, 4, 1)
        )
        m_q2_ytd = Measurement(
            start=date(2023, 1, 1), end=date(2023, 6, 30),
            val=250, accn="2", fy=2023, fp=FiscalPeriod.Q2,
            form="10-Q", filed=date(2023, 8, 1)
        )
        m_q3_ytd = Measurement(
            start=date(2023, 1, 1), end=date(2023, 9, 30),
            val=450, accn="3", fy=2023, fp=FiscalPeriod.Q3,
            form="10-Q", filed=date(2023, 11, 1)
        )
        
        # Create univariate measurements
        from edgarito.schemas.reader.measurements import UnivariateMeasurements
        univariate = UnivariateMeasurements.from_measurements(
            concept="Test",
            granularity=Granularity.QUARTERLY,
            measurements=[m_q1, m_q2_ytd, m_q3_ytd]
        )
        univariate.sort()
        
        # Run conversion
        base_reader._convert_ytd_to_quarterly(univariate)
        
        # Verify values
        values = {p.fp: v for v, p in zip(univariate.values, univariate.periods)}
        assert values[FiscalPeriod.Q1] == 100
        assert values[FiscalPeriod.Q2] == 150  # 250 - 100
        assert values[FiscalPeriod.Q3] == 200  # 450 - 250

    def test_convert_fy_to_q4(self, base_reader):
        """Test conversion of FY and Q1-Q3 to Q4."""
        # Q1: 100, Q2: 100, Q3: 100
        # FY: 500 (so Q4 = 200)
        
        m_q1 = Measurement(
            start=date(2023, 1, 1), end=date(2023, 3, 31),
            val=100, accn="1", fy=2023, fp=FiscalPeriod.Q1,
            form="10-Q", filed=date(2023, 4, 1)
        )
        m_q2 = Measurement(
            start=date(2023, 4, 1), end=date(2023, 6, 30),
            val=100, accn="2", fy=2023, fp=FiscalPeriod.Q2,
            form="10-Q", filed=date(2023, 8, 1)
        )
        m_q3 = Measurement(
            start=date(2023, 7, 1), end=date(2023, 9, 30),
            val=100, accn="3", fy=2023, fp=FiscalPeriod.Q3,
            form="10-Q", filed=date(2023, 11, 1)
        )
        m_fy = Measurement(
            start=date(2023, 1, 1), end=date(2023, 12, 31),
            val=500, accn="4", fy=2023, fp=FiscalPeriod.Year,
            form="10-K", filed=date(2024, 3, 1)
        )
        
        from edgarito.schemas.reader.measurements import UnivariateMeasurements
        univariate = UnivariateMeasurements.from_measurements(
            concept="Test",
            granularity=Granularity.QUARTERLY,
            measurements=[m_q1, m_q2, m_q3, m_fy]
        )
        univariate.sort()
        
        base_reader._convert_fy_to_q4(univariate)
        
        # Verify Q4 exists and has correct value
        q4_found = False
        for v, p in zip(univariate.values, univariate.periods):
            if p.fp == FiscalPeriod.Q4:
                assert v == 200  # 500 - 300
                q4_found = True
        assert q4_found
