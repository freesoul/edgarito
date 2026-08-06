import pytest
import datetime
from edgarito.services.financial.balance_sheet_reader import BalanceSheetReader
from edgarito.schemas.edgar_responses.company_facts import CompanyFacts, Facts, Fact, FactUnits, Measurement
from edgarito.enums.granularity import Granularity
from edgarito.enums.edgar.period import FiscalPeriod

class TestIntegration:
    
    @pytest.fixture
    def mock_aapl_facts(self):
        """Mock AAPL data with some realistic complexity (YTD, multiple filings)."""
        return CompanyFacts(
            cik=320193,
            entityName="Apple Inc.",
            facts=Facts(
                dei={},
                **{"us-gaap": {
                    "Assets": Fact(
                        label="Assets",
                        description="Total Assets",
                        units=FactUnits(
                            USD=[
                                # 2023 Q1 (Individual)
                                Measurement(
                                    start=datetime.date(2022, 10, 1), end=datetime.date(2022, 12, 31),
                                    val=346747000000, accn="1", fy=2023, fp=FiscalPeriod.Q1,
                                    form="10-Q", filed=datetime.date(2023, 2, 3)
                                ),
                                # 2023 Q2 (Individual)
                                Measurement(
                                    start=datetime.date(2023, 1, 1), end=datetime.date(2023, 4, 1),
                                    val=332160000000, accn="2", fy=2023, fp=FiscalPeriod.Q2,
                                    form="10-Q", filed=datetime.date(2023, 5, 5)
                                ),
                                # 2023 Q3 (Individual)
                                Measurement(
                                    start=datetime.date(2023, 4, 2), end=datetime.date(2023, 7, 1),
                                    val=335038000000, accn="3", fy=2023, fp=FiscalPeriod.Q3,
                                    form="10-Q", filed=datetime.date(2023, 8, 4)
                                ),
                                # 2023 FY (Annual)
                                Measurement(
                                    start=datetime.date(2022, 9, 25), end=datetime.date(2023, 9, 30),
                                    val=352583000000, accn="4", fy=2023, fp=FiscalPeriod.Year,
                                    form="10-K", filed=datetime.date(2023, 11, 3)
                                )
                            ]
                        )
                    ),
                    "Liabilities": Fact(
                        label="Liabilities",
                        description="Total Liabilities",
                        units=FactUnits(
                            USD=[
                                Measurement(
                                    start=datetime.date(2022, 9, 25), end=datetime.date(2023, 9, 30),
                                    val=290437000000, accn="4", fy=2023, fp=FiscalPeriod.Year,
                                    form="10-K", filed=datetime.date(2023, 11, 3)
                                )
                            ]
                        )
                    )
                }}
            )
        )

    def test_balance_sheet_reader_integration(self, mock_aapl_facts):
        reader = BalanceSheetReader(mock_aapl_facts)
        
        # Test Annual Assets
        assets_annual = reader.get_total_assets(Granularity.ANNUAL)
        assert len(assets_annual.values) == 1
        assert assets_annual.values[0] == 352583000000
        
        # Test Quarterly Assets
        assets_quarterly = reader.get_total_assets(Granularity.QUARTERLY)
        # Should have Q1, Q2, Q3, and FY (which counts as Q4-ish or just FY depending on logic)
        # Balance sheet reader sets convert_fy_to_q4=False, so FY remains FY
        # But wait, get_total_assets calls _get_concept with convert_fy_to_q4=False
        # So we should see Q1, Q2, Q3, and FY
        assert len(assets_quarterly.values) == 4
        
        # Test Liabilities
        liabilities = reader.get_total_liabilities(Granularity.ANNUAL)
        assert len(liabilities.values) == 1
        assert liabilities.values[0] == 290437000000

