import pytest
from unittest.mock import MagicMock
from edgarito.services.analysis.red_flags.earnings_quality_analyzer import EarningsQualityAnalyzer
from edgarito.schemas.reader.measurements import UnivariateMeasurements
from edgarito.enums.granularity import Granularity
from edgarito.schemas.red_flags import RedFlagSeverity

class TestEarningsQualityAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        # Mock dependencies
        facts = MagicMock()
        balance_sheet = MagicMock()
        income_statement = MagicMock()
        cash_flow = MagicMock()
        
        return EarningsQualityAnalyzer(
            facts=facts,
            balance_sheet=balance_sheet,
            income_statement=income_statement,
            cash_flow=cash_flow,
            thresholds=MagicMock(),
            sector="Technology",
            sector_profile=MagicMock(),
            market_cap=1000000000,
            market_data=MagicMock()
        )

    def test_dsri_critical(self, analyzer):
        """Test Days Sales Receivable Index (DSRI) critical flag."""
        # Setup data:
        # Current: Receivables=200, Revenue=100 -> Ratio=2.0
        # Previous: Receivables=100, Revenue=100 -> Ratio=1.0
        # DSRI = 2.0 / 1.0 = 2.0 (> 1.5 Critical)
        
        # Mock get_accounts_receivable
        rec_meas = MagicMock()
        rec_meas.values = [200, 100]
        rec_meas.periods = ["2023", "2022"]
        rec_meas.intersect.return_value = rec_meas
        analyzer.balance_sheet.get_accounts_receivable.return_value = rec_meas
        
        # Mock get_revenue
        rev_meas = MagicMock()
        rev_meas.values = [100, 100]
        rev_meas.periods = ["2023", "2022"]
        rev_meas.intersect.return_value = rev_meas
        analyzer.income_statement.get_revenue.return_value = rev_meas
        
        flags = analyzer.analyze(Granularity.ANNUAL)
        
        dsri_flags = [f for f in flags if "DSRI" in f.title]
        assert len(dsri_flags) == 1
        assert dsri_flags[0].severity == RedFlagSeverity.CRITICAL
        assert dsri_flags[0].current_value == 2.0

    def test_gmi_warning(self, analyzer):
        """Test Gross Margin Index (GMI) warning flag."""
        # Setup data:
        # Current: GrossProfit=40, Revenue=100 -> Margin=0.4
        # Previous: GrossProfit=60, Revenue=100 -> Margin=0.6
        # GMI = 0.6 / 0.4 = 1.5 (> 1.3 Warning)
        
        # Mock get_accounts_receivable (to avoid DSRI error/flag)
        analyzer.balance_sheet.get_accounts_receivable.return_value = MagicMock(values=[])
        
        # Mock get_gross_profit
        gp_meas = MagicMock()
        gp_meas.values = [40, 60]
        gp_meas.periods = ["2023", "2022"]
        gp_meas.intersect.return_value = gp_meas
        analyzer.income_statement.get_gross_profit.return_value = gp_meas
        
        # Mock get_revenue
        rev_meas = MagicMock()
        rev_meas.values = [100, 100]
        rev_meas.periods = ["2023", "2022"]
        rev_meas.intersect.return_value = rev_meas
        analyzer.income_statement.get_revenue.return_value = rev_meas
        
        flags = analyzer.analyze(Granularity.ANNUAL)
        
        gmi_flags = [f for f in flags if "GMI" in f.title]
        assert len(gmi_flags) == 1
        assert gmi_flags[0].severity == RedFlagSeverity.WARNING
        assert gmi_flags[0].current_value == pytest.approx(1.5)
