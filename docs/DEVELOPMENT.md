The current objectives are:

- Provide a simple async client for the new API of EDGAR.
    - Cache the responses.
    - Retries.
    - Structure the responses into dataclasses.

- (maybe) Higher level client for the low level API/non-API routes of edgar
    - Paginate submissions (?)

- (maybe) Client for scrapping EDGAR HTML pages as fallback

- (maybe) Client for downloading FILES from EDGAR.
    - Download both original files or separate ones (eg. only XBRL xml).
    - Cache the files.
    - Extraction of full-submission.txt (see secedgar.parser)

- Parse the XBRL in order to extract financial data (as an alternative to direct API calls).
    py-xbrl

- Compute basic non-GAAP metrics

- Spot red flags on a company.






- Fetching data.

    1) API client

        Constraints
            - Cached
            - Async
            - Retry

        Docs
            - https://www.sec.gov/search-filings/edgar-application-programming-interfaces
            - https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data

        Routes
            - Ticker-CIK - OK
                https://www.sec.gov/files/company_tickers.json

            - Submissions - OK
                https://data.sec.gov/submissions/CIK0001018724.json

            - One submission files (first resp_json["filings"]["files"])
                https://data.sec.gov/submissions/CIK0001018724-submissions-001.json

            - Company facts (Everything from XBRLs)
                https://data.sec.gov/api/xbrl/companyfacts/CIK0001018724.json

            - A fact of a company
                https://data.sec.gov/api/xbrl/companyconcept/CIK0001018724/us-gaap/AccountsPayableCurrent.json

            - Frames
                https://data.sec.gov/api/xbrl/frames/us-gaap/AccountsPayableCurrent/USD/CY2019Q1I.json

        Non API routes
            - Daily filing (by range; tar file; see secedgar)
                Archives/edgar/daily-index/{year}/QTR{num}/
            
            - Quarterly (by range; tar file; see secedgar)
                Archives/edgar/full-index/{year}/QTR{num}/

            - Submissions directory
                https://www.sec.gov/Archives/edgar/data/1018724/

            - Submission files
                https://www.sec.gov/Archives/edgar/data/1018724/000101872425000004/             

            - A submission file
                https://www.sec.gov/Archives/edgar/data/1018724/000101872425000004/amzn-20241231_htm.xml


- Download files
    https://www.sec.gov/Archives/edgar/data/{cik}/{acc_num_no_dash}/{document}

- Fraud odds report.
    - Service that spots fraud based on this https://www.linkedin.com/pulse/recognising-financial-statement-red-flags-guide-entrepreneurs-jplsf/.
