import pytest
import asyncio
from edgarito.schemas.edgar_responses.company_facts import CompanyFacts, Facts, Fact, FactUnits, Measurement
from edgarito.schemas.market_data import MarketData
from edgarito.enums.granularity import Granularity

@pytest.fixture
def mock_company_facts():
    """Create a mock CompanyFacts object with minimal data for testing."""
    return CompanyFacts(
        cik=320193,
        entityName="Apple Inc.",
        facts=Facts(
            dei={},
            us_gaap={
                "Assets": Fact(
                    label="Assets",
                    description="Sum of the carrying amounts as of the balance sheet date of all assets that are recognized. Assets are probable future economic benefits obtained or controlled by an entity as a result of past transactions or events.",
                    units=FactUnits(
                        USD=[
                            Measurement(
                                start="2023-01-01",
                                end="2023-12-31",
                                val=1000000000,
                                accn="0000320193-23-000106",
                                fy=2023,
                                fp="FY",
                                form="10-K",
                                filed="2023-11-03",
                                frame="CY2023"
                            ),
                            Measurement(
                                start="2023-07-01",
                                end="2023-09-30",
                                val=1000000000,
                                accn="0000320193-23-000106",
                                fy=2023,
                                fp="Q3",
                                form="10-Q",
                                filed="2023-11-03",
                                frame="CY2023Q3I"
                            )
                        ]
                    )
                ),
                "Liabilities": Fact(
                    label="Liabilities",
                    description="Sum of the carrying amounts as of the balance sheet date of all liabilities that are recognized. Liabilities are probable future sacrifices of economic benefits arising from present obligations of a particular entity to transfer assets or provide services to other entities in the future as a result of past transactions or events.",
                    units=FactUnits(
                        USD=[
                             Measurement(
                                start="2023-07-01",
                                end="2023-09-30",
                                val=500000000,
                                accn="0000320193-23-000106",
                                fy=2023,
                                fp="Q3",
                                form="10-Q",
                                filed="2023-11-03",
                                frame="CY2023Q3I"
                            )
                        ]
                    )
                ),
                 "StockholdersEquity": Fact(
                    label="Stockholders' Equity",
                    description="Total of all stockholders' equity (deficit) items, net of receivables from officers, directors, owners, and affiliates of the entity which are attributable to the parent. The amount of the economic entity's stockholders' equity. The stockholders' equity of the entity is the residual interest in the assets of the entity after deducting all its liabilities.",
                    units=FactUnits(
                        USD=[
                             Measurement(
                                start="2023-07-01",
                                end="2023-09-30",
                                val=500000000,
                                accn="0000320193-23-000106",
                                fy=2023,
                                fp="Q3",
                                form="10-Q",
                                filed="2023-11-03",
                                frame="CY2023Q3I"
                            )
                        ]
                    )
                )
            }
        )
    )

@pytest.fixture
def mock_market_data():
    return MarketData(
        ticker="AAPL",
        market_cap=2000000000,
        sector="Technology",
        industry="Consumer Electronics",
        current_price=150.0,
        currency="USD"
    )
