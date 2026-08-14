import asyncio
import datetime
import hashlib
import json
from decimal import Decimal

import pytest
from support.http import FakeResponse, FakeSession

from edgarito.schemas.market import (
    MarketDataFrequency,
    ReferenceSeriesKind,
    ReferenceValueUnit,
)
from edgarito.schemas.valuation import ReferenceDatasetRelease
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.providers.damodaran import DamodaranClient
from edgarito.services.providers.ecb import EcbClient
from edgarito.services.providers.fred import FredClient
from edgarito.services.providers.treasury import TreasuryClient

UTC = datetime.timezone.utc


TREASURY_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata">
  <updated>2026-08-05T15:44:56Z</updated>
  <entry><content type="application/xml"><m:properties>
    <d:NEW_DATE m:type="Edm.DateTime">2026-08-04T00:00:00</d:NEW_DATE>
    <d:BC_10YEAR m:type="Edm.Double">4.21</d:BC_10YEAR>
  </m:properties></content></entry>
  <entry><content type="application/xml"><m:properties>
    <d:NEW_DATE m:type="Edm.DateTime">2026-08-05T00:00:00</d:NEW_DATE>
    <d:BC_10YEAR m:type="Edm.Double">4.19</d:BC_10YEAR>
  </m:properties></content></entry>
</feed>"""


def test_treasury_normalizes_yields_and_reuses_the_raw_year_cache(tmp_path):
    session = FakeSession([FakeResponse(content=TREASURY_XML)])
    client = TreasuryClient(FileSystemCache(tmp_path), session=session)

    first = asyncio.run(client.get_par_yield(120, 2026))
    second = asyncio.run(client.get_par_yield(120, 2026))

    assert first == second
    assert first.provider == "treasury"
    assert first.kind == ReferenceSeriesKind.GOVERNMENT_YIELD
    assert first.unit == ReferenceValueUnit.PERCENTAGE_POINTS
    assert first.tenor_months == 120
    assert first.latest_observation.value == Decimal("4.19")
    assert first.source_version == "2026-08-05T15:44:56Z"
    assert len(session.calls) == 1
    assert session.calls[0]["params"]["field_tdr_date_value"] == "2026"


def test_treasury_rejects_unsupported_tenors_without_a_request(tmp_path):
    session = FakeSession([])
    client = TreasuryClient(FileSystemCache(tmp_path), session=session)

    with pytest.raises(ValueError, match="Unsupported Treasury tenor"):
        asyncio.run(client.get_par_yield(18, 2026))

    assert session.calls == []


def test_fred_normalizes_metadata_observations_and_vintage(tmp_path):
    metadata = {
        "seriess": [
            {
                "id": "DGS10",
                "title": "Market Yield on U.S. Treasury Securities at 10-Year",
                "frequency_short": "D",
                "units": "Percent",
                "last_updated": "2026-08-05 16:01:16-05",
            }
        ]
    }
    observations = {
        "observations": [
            {"date": "2026-08-04", "value": "4.21"},
            {"date": "2026-08-05", "value": "."},
            {"date": "2026-08-06", "value": "4.19"},
        ]
    }
    session = FakeSession(
        [
            FakeResponse(content=json.dumps(metadata)),
            FakeResponse(content=json.dumps(observations)),
        ]
    )
    client = FredClient(FileSystemCache(tmp_path), "free-test-key", session=session)
    vintage = datetime.date(2026, 8, 6)

    first = asyncio.run(
        client.get_series(
            "dgs10",
            kind=ReferenceSeriesKind.GOVERNMENT_YIELD,
            vintage_date=vintage,
            currency="USD",
            country="US",
        )
    )
    second = asyncio.run(
        client.get_series(
            "DGS10",
            kind=ReferenceSeriesKind.GOVERNMENT_YIELD,
            vintage_date=vintage,
            currency="USD",
            country="US",
        )
    )

    assert first == second
    assert first.frequency == MarketDataFrequency.DAILY
    assert first.unit == ReferenceValueUnit.PERCENTAGE_POINTS
    assert [item.value for item in first.observations] == [
        Decimal("4.21"),
        Decimal("4.19"),
    ]
    assert first.source_version == "vintage:2026-08-06"
    assert len(session.calls) == 2
    assert all(
        call["params"]["realtime_start"] == "2026-08-06" for call in session.calls
    )
    assert not any("free-test-key" in path.name for path in tmp_path.rglob("*"))


def test_fred_requires_an_api_key(tmp_path):
    with pytest.raises(ValueError, match="FRED API key"):
        FredClient(FileSystemCache(tmp_path), None)


ECB_CSV = """KEY,FREQ,REF_AREA,CURRENCY,TIME_PERIOD,OBS_VALUE,TITLE,UNIT
FM.D.U2.EUR.4F.KR.MRR_FR.LEV,D,U2,EUR,2026-01-01,2.15,Main refinancing operations,PCPA
FM.D.U2.EUR.4F.KR.MRR_FR.LEV,D,U2,EUR,2026-01-02,2.15,Main refinancing operations,PCPA
"""


def test_ecb_normalizes_sdmx_csv_and_preserves_http_revision(tmp_path):
    session = FakeSession(
        [FakeResponse(content=ECB_CSV, headers={"ETag": '"revision-42"'})]
    )
    client = EcbClient(FileSystemCache(tmp_path), session=session)

    result = asyncio.run(
        client.get_series(
            "FM",
            "D.U2.EUR.4F.KR.MRR_FR.LEV",
            kind=ReferenceSeriesKind.POLICY_RATE,
            start_period=datetime.date(2026, 1, 1),
        )
    )

    assert result.provider == "ecb"
    assert result.frequency == MarketDataFrequency.DAILY
    assert result.currency == "EUR"
    assert result.region == "U2"
    assert result.unit == ReferenceValueUnit.PERCENTAGE_POINTS
    assert result.source_version == '"revision-42"'
    assert result.latest_observation.value == Decimal("2.15")
    assert session.calls[0]["params"] == {
        "format": "csvdata",
        "startPeriod": "2026-01-01",
    }


def test_ecb_rejects_a_query_returning_multiple_series(tmp_path):
    content = ECB_CSV + (
        "FM.D.U2.EUR.4F.KR.DFR.LEV,D,U2,EUR,2026-01-02,2.00,Deposit facility,PCPA\n"
    )
    client = EcbClient(
        FileSystemCache(tmp_path), session=FakeSession([FakeResponse(content=content)])
    )

    with pytest.raises(RuntimeError, match="more than one series"):
        asyncio.run(client.get_series("FM", "D.U2.EUR.4F.KR.+.LEV"))


def test_ecb_normalizes_monthly_periods_to_month_end(tmp_path):
    content = """KEY,FREQ,REF_AREA,CURRENCY,TIME_PERIOD,OBS_VALUE,TITLE,UNIT
ICP.M.U2.N.000000.4.ANR,M,U2,EUR,2026-02,1.9,Euro area inflation,PC
"""
    client = EcbClient(
        FileSystemCache(tmp_path), session=FakeSession([FakeResponse(content=content)])
    )

    result = asyncio.run(
        client.get_series(
            "ICP",
            "M.U2.N.000000.4.ANR",
            kind=ReferenceSeriesKind.INFLATION_RATE,
        )
    )

    assert result.latest_observation.period_end == datetime.date(2026, 2, 28)


COUNTRY_HTML = """<html><table>
<tr><th>Country</th><th>Moody's rating</th><th>Adj. Default Spread</th>
<th>Country Risk Premium</th><th>Equity Risk Premium</th>
<th>Corporate Tax Rate</th><th>Sovereignn CDS</th>
<th>ERP based on sovereign CDSS</th></tr>
<tr><td>United States</td><td>Aa1</td><td>0.23%</td><td>0.36%</td>
<td>4.59%</td><td>25.00%</td><td>0.20%</td><td>4.53%</td></tr>
<tr><td>Germany</td><td>Aaa</td><td>0.00%</td><td>0.00%</td>
<td>4.23%</td><td>29.93%</td><td>NA</td><td>NA</td></tr>
</table></html>"""

INDUSTRY_HTML = """<html><table>
<tr><th>Industry Name</th><th>Number of firms</th><th>Beta</th><th>D/E Ratio</th>
<th>Effective Tax rate</th><th>Unlevered beta</th><th>Cash/Firm value</th>
<th>Unlevered beta corrected for cash</th><th>HiLo Risk</th>
<th>Standard deviation of equity</th>
<th>Standard deviation in operating income (last 10 years)</th></tr>
<tr><td>Aerospace/Defense</td><td>79</td><td>0.95</td><td>15.56%</td>
<td>11.58%</td><td>0.85</td><td>2.61%</td><td>0.87</td><td>0.5213</td>
<td>46.45%</td><td>21.86%</td></tr>
</table></html>"""


def _release(dataset: str, version: str, content: str, region: str):
    return ReferenceDatasetRelease(
        dataset=dataset,
        version=version,
        published_on=datetime.date(2026, 1, 5),
        source_url=f"https://example.test/{dataset}.html",
        expected_sha256=hashlib.sha256(content.encode()).hexdigest(),
        region=region,
    )


def test_damodaran_country_risk_is_typed_versioned_and_auditable(tmp_path):
    release = _release("country-risk-premiums", "2026-01-05", COUNTRY_HTML, "global")
    session = FakeSession([FakeResponse(content=COUNTRY_HTML)])
    client = DamodaranClient(FileSystemCache(tmp_path), session=session)

    first = asyncio.run(client.get_country_risk_premiums(release))
    second = asyncio.run(client.get_country_risk_premiums(release))
    germany = first.find_country(" germany ")

    assert first == second
    assert germany is not None
    assert germany.country_risk_premium == Decimal("0.00")
    assert germany.sovereign_cds_spread is None
    assert first.metadata.version == "2026-01-05"
    assert first.metadata.sha256 == release.expected_sha256
    provenance = first.metadata.assumption_provenance()
    assert provenance.dataset == "country-risk-premiums"
    assert provenance.version == "2026-01-05"
    assert len(session.calls) == 1


def test_damodaran_industry_betas_preserve_percentage_point_units(tmp_path):
    release = _release("industry-betas", "2026-01-08", INDUSTRY_HTML, "US")
    client = DamodaranClient(
        FileSystemCache(tmp_path),
        session=FakeSession([FakeResponse(content=INDUSTRY_HTML)]),
    )

    result = asyncio.run(client.get_industry_betas(release))
    aerospace = result.find_industry("aerospace/defense")

    assert aerospace is not None
    assert aerospace.number_of_firms == 79
    assert aerospace.levered_beta == Decimal("0.95")
    assert aerospace.debt_to_equity == Decimal("15.56")
    assert aerospace.cash_adjusted_unlevered_beta == Decimal("0.87")
    assert result.metadata.region == "US"


def test_damodaran_rejects_changed_content_before_caching(tmp_path):
    release = _release("country-risk-premiums", "2026-01-05", COUNTRY_HTML, "global")
    changed = COUNTRY_HTML.replace("4.59%", "4.60%")
    client = DamodaranClient(
        FileSystemCache(tmp_path), session=FakeSession([FakeResponse(content=changed)])
    )

    with pytest.raises(RuntimeError, match="checksum"):
        asyncio.run(client.get_country_risk_premiums(release))

    assert not list(tmp_path.rglob("*.json"))
