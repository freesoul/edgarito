This project provides a async client for the EDGAR REST API which structures the responses into pydantic schemas.

The following purposes include adding financial statement reader classes, computing common metrics such as FCF, and adding classes to spot investment red flags.

Finally, perhaps adding, as a fallback to the EDGAR REST API, the pipeline of downloading, extracting and loading into the pydantic schemas the data directly from edgar txt/xbrl submissions.

# Installation
```
python -m pip install edgarito
```

# Example usage of the async API

```
from edgarito.services.retrieval.edgar_rest_client.low_level_client import EDGARLowLevelClient
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.enums.edgar.core_filing_type import CoreFilingType


async def main():
    # A cache is required to comply with SEC request of optimizing requests.
    cache = FileSystemCache(root_directory="./cache")

    # We create a client to interact with the SEC API.
    async with EDGARLowLevelClient(cache=cache, user_agent="Your Name (your-email@gmail.com)") as edgar:

        # Search for Apple CIK from the tickers.
        tickers = await edgar.get_tickers()
        for ticker_info in tickers:
            if ticker_info.ticker == "AAPL":
                print(f"CIK for AAPL is {ticker_info.cik}")
                cik = ticker_info.cik
                break
        else:
            raise ValueError("Ticker not found")

        # Display some company facts
        company = await edgar.get_company_facts(cik=cik)

        for fact_name, fact_info in company.facts.us_gaap.items():

            if fact_info.units.USD:  # or fact_info.units.shares...

                print(f"{fact_name} - {fact_info.label}:")
                for measurement in fact_info.units.USD:

                    if measurement.parsed_type != CoreFilingType.FILING_10K:
                        continue
                    print(f"\t{measurement.fy}: {measurement.val}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

```

# Examples of CLI usage.

- Find submissions `python -m edgarito.cli submissions --cik 320193 --type 10-K`

```
[2025-03-01 12:37:41,330][INFO][edgarito.services.retrieval.edgar_rest_client.submissions_client] Got 1000 filings for CIK 320193
[2025-03-01 12:37:41,330][INFO][edgarito.services.retrieval.edgar_rest_client.submissions_client] Getting additional filings for name='CIK0000320193-submissions-001.json' filingCount=1125 filingFrom=datetime.date(1994, 1, 26) filingTo=datetime.date(2014, 5, 28)
[2025-03-01 12:37:41,330][INFO][edgarito.services.retrieval.edgar_rest_client.submissions_client] Got 1125 additional filings for name='CIK0000320193-submissions-001.json' filingCount=1125 filingFrom=datetime.date(1994, 1, 26) filingTo=datetime.date(2014, 5, 28)
[2025-03-01 12:37:41,346][INFO][Cli] 1994-12-13         0000320193-94-000016    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 1995-12-19         0000320193-95-000016    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 1996-12-19         0000320193-96-000023    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 1997-12-05         0001047469-97-006960    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 1999-12-22         0000912057-99-010244    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 2000-12-14         0000912057-00-053623    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 2002-12-19         0001047469-02-007674    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 2003-12-19         0001047469-03-041604    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 2004-12-03         0001047469-04-035975    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 2005-12-01         0001104659-05-058421    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 2006-12-29         0001104659-06-084288    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 2007-11-15         0001047469-07-009340    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 2008-11-05         0001193125-08-224958    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 2009-10-27         0001193125-09-214859    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 2010-10-27         0001193125-10-238044    10-K (10-K)
[2025-03-01 12:37:41,346][INFO][Cli] 2011-10-26         0001193125-11-282113    10-K (10-K)
[2025-03-01 12:37:41,347][INFO][Cli] 2012-10-31         0001193125-12-444068    10-K (10-K)
[2025-03-01 12:37:41,347][INFO][Cli] 2013-10-30         0001193125-13-416534    10-K (10-K)
[2025-03-01 12:37:41,347][INFO][Cli] 2014-10-27         0001193125-14-383437    10-K (10-K)
[2025-03-01 12:37:41,347][INFO][Cli] 2015-10-28         0001193125-15-356351    10-K (10-K)
[2025-03-01 12:37:41,347][INFO][Cli] 2016-10-26         0001628280-16-020309    10-K (10-K)
[2025-03-01 12:37:41,347][INFO][Cli] 2017-11-03         0000320193-17-000070    10-K (10-K)
[2025-03-01 12:37:41,347][INFO][Cli] 2018-11-05         0000320193-18-000145    10-K (10-K)
[2025-03-01 12:37:41,347][INFO][Cli] 2019-10-31         0000320193-19-000119    10-K (XBRL)
[2025-03-01 12:37:41,347][INFO][Cli] 2020-10-30         0000320193-20-000096    10-K (XBRL)
[2025-03-01 12:37:41,347][INFO][Cli] 2021-10-29         0000320193-21-000105    10-K (XBRL)
[2025-03-01 12:37:41,347][INFO][Cli] 2022-10-28         0000320193-22-000108    10-K (XBRL)
[2025-03-01 12:37:41,347][INFO][Cli] 2023-11-03         0000320193-23-000106    10-K (XBRL)
[2025-03-01 12:37:41,347][INFO][Cli] 2024-11-01         0000320193-24-000123    10-K (XBRL)
```

- Download last 10-K for AAPL `python -m edgarito.cli download --ticker aapl --type 10-K --limit 5`

```
[2025-03-01 12:37:03,588][INFO][Cli] CIK: 320193
[2025-03-01 12:37:03,590][INFO][edgarito.services.retrieval.edgar_rest_client.submissions_client] Got 1000 filings for CIK 320193
[2025-03-01 12:37:03,590][INFO][edgarito.services.retrieval.edgar_rest_client.submissions_client] Getting additional filings for name='CIK0000320193-submissions-001.json' filingCount=1125 filingFrom=datetime.date(1994, 1, 26) filingTo=datetime.date(2014, 5, 28)
[2025-03-01 12:37:03,592][INFO][edgarito.services.retrieval.edgar_rest_client.submissions_client] Got 1125 additional filings for name='CIK0000320193-submissions-001.json' filingCount=1125 filingFrom=datetime.date(1994, 1, 26) filingTo=datetime.date(2014, 5, 28)
[2025-03-01 12:37:03,601][INFO][Cli] 2020-10-30         0000320193-20-000096    10-K (XBRL)
[2025-03-01 12:37:03,601][INFO][Cli] 2021-10-29         0000320193-21-000105    10-K (XBRL)
[2025-03-01 12:37:03,602][INFO][Cli] 2022-10-28         0000320193-22-000108    10-K (XBRL)
[2025-03-01 12:37:03,602][INFO][Cli] 2023-11-03         0000320193-23-000106    10-K (XBRL)
[2025-03-01 12:37:03,602][INFO][Cli] 2024-11-01         0000320193-24-000123    10-K (XBRL)
[2025-03-01 12:37:03,602][INFO][DownloadService] Fetching https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/0000320193-20-000096.txt
[2025-03-01 12:37:03,604][INFO][DownloadService] Fetching https://www.sec.gov/Archives/edgar/data/320193/000032019321000105/0000320193-21-000105.txt
[2025-03-01 12:37:03,604][INFO][DownloadService] Fetching https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/0000320193-22-000108.txt
[2025-03-01 12:37:03,605][INFO][DownloadService] Fetching https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/0000320193-23-000106.txt
[2025-03-01 12:37:03,605][INFO][DownloadService] Fetching https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/0000320193-24-000123.txt
```

You can find more commands in `edgarito.cli.__main__`