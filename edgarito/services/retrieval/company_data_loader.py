"""
Company Data Loader - Orchestrates the complete data loading pipeline.

This service provides a unified interface for loading company financial data by:
1. Fetching CompanyFacts from SEC EDGAR REST API (includes CIK, entity name, XBRL data)
2. Fetching company submissions to identify available filings
3. For foreign 20-F filers: downloading and parsing 6-K quarterly reports
4. Combining all data sources into a unified response

This replaces direct usage of edgar_rest_client throughout the codebase.
"""
from typing import Optional
import json
import aiohttp

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts, Measurement, Fact, FactUnits
from edgarito.schemas.edgar_responses.submission import CompanySubmissionsResponse
from edgarito.services.retrieval.edgar_rest_client.low_level_client import EDGARLowLevelClient
from edgarito.services.retrieval.edgar_rest_client.submissions_client import SubmissionsClient
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.retrieval.downloader.download_service import DownloadService
from edgarito.services.retrieval.parser.filing_6k_parser import Filing6KParser
from edgarito.services.retrieval.merger.combiner_6k import SixKCombiner, QuarterlyDataPoint
from edgarito.enums.edgar.period import FiscalPeriod


class CompanyDataLoader:
    """
    High-level service for loading all company financial data.
    
    Orchestrates:
    - CompanyFacts API (annual/quarterly XBRL from 10-K/10-Q/20-F)
    - Submissions API (list of all filings)
    - 6-K filing downloads and parsing (quarterly data for foreign companies)
    - Data merging via SixKCombiner
    
    Usage:
        async with aiohttp.ClientSession() as session:
            loader = CompanyDataLoader(session, cache_dir="cache")
            
            # Load by ticker
            result = await loader.load_from_ticker("RACE")
            print(result.facts.entityName)  # Ferrari N.V.
            print(result.combiner.get_available_metrics())  # ['net_revenues', 'ebit', ...]
            
            # Load by CIK
            result = await loader.load_from_cik(1648416)
    """
    
    def __init__(
        self,
        session: aiohttp.ClientSession,
        cache_dir: str = "cache",
        download_dir: str = "cache/downloads",
        user_agent: str = "Edgarito Client (contact@example.com)",
        use_cache: bool = True,
        make_cache: bool = True
    ):
        """
        Initialize the data loader.
        
        Args:
            session: aiohttp session for HTTP requests
            cache_dir: Directory for caching API responses
            download_dir: Directory for downloaded filing documents
            user_agent: User agent string for SEC API requests
            use_cache: Whether to use cached data
            make_cache: Whether to cache new data
        """
        self.session = session
        self.cache_dir = cache_dir
        self.download_dir = download_dir
        self.user_agent = user_agent
        self.use_cache = use_cache
        self.make_cache = make_cache
        
        # Initialize services
        self.cache = FileSystemCache(cache_dir)
        self.edgar_client = EDGARLowLevelClient(self.cache, user_agent, session)
        self.submissions_client = SubmissionsClient(self.edgar_client)
        self.download_service = DownloadService(session, download_root_dir=download_dir, cache=self.cache)
        self.parser = Filing6KParser()
    
    async def load_from_ticker(self, ticker: str) -> 'CompanyDataResult':
        """
        Load all financial data for a company by ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (e.g., "RACE", "AAPL")
            
        Returns:
            CompanyDataResult with facts, submissions, and optional 6-K combiner
        """
        # Get CIK from ticker
        cik = await self._get_cik_from_ticker(ticker)
        
        return await self.load_from_cik(cik)
    
    async def _get_cik_from_ticker(self, ticker: str) -> int:
        """Helper method to resolve ticker to CIK."""
        tickers = await self.edgar_client.get_tickers(
            use_cache=self.use_cache, 
            make_cache=self.make_cache
        )
        for ticker_obj in tickers:
            if ticker.lower() == ticker_obj.ticker.lower():
                return ticker_obj.cik_str
        raise ValueError(f"Ticker {ticker} not found")
    
    async def load_from_cik(self, cik: int) -> 'CompanyDataResult':
        """
        Load all financial data for a company by CIK.
        
        Args:
            cik: Central Index Key (e.g., 1648416 for Ferrari)
            
        Returns:
            CompanyDataResult with facts, submissions, and optional 6-K combiner
        """
        # 1. Load CompanyFacts (XBRL data from 10-K/10-Q/20-F)
        facts = await self.edgar_client.get_company_facts(
            cik,
            use_cache=self.use_cache,
            make_cache=self.make_cache
        )
        
        # 2. Load submissions (list of all filings)
        submissions = await self.submissions_client.get_all_submissions(
            cik,
            use_cache=self.use_cache,
            make_cache=self.make_cache
        )
        
        # 3. Check if company has 6-K filings and merge into CompanyFacts
        await self._merge_6k_data_into_facts(cik, submissions, facts)
        
        return CompanyDataResult(
            facts=facts,
            submissions=submissions,
            combiner=None  # No longer needed, data is in facts
        )
    
    async def _merge_6k_data_into_facts(
        self, 
        cik: int, 
        submissions: CompanySubmissionsResponse,
        facts: CompanyFacts
    ) -> None:
        """
        Merge 6-K quarterly data directly into CompanyFacts structure.
        
        For companies filing 20-F (annual) instead of 10-K, quarterly data is NOT
        available in CompanyFacts API. These companies file 6-K reports with quarterly
        earnings in HTML press releases (Exhibit 99.1).
        
        This method downloads, parses, and injects 6-K data as synthetic Measurement
        objects into the CompanyFacts, making it transparent to downstream code.
        
        Args:
            cik: Central Index Key
            submissions: Company submissions response
            facts: CompanyFacts to modify in-place
        """
        # Find 6-K filings with quarterly pattern (q1, q2, q3, q4 in filename/description)
        # First, transpose the filings to get a list of TransposedFiling objects
        all_filings = submissions.filings.recent.transpose()
        
        six_k_filings = []
        for filing in all_filings:
            if filing.form == "6-K":
                # Check if filename/description suggests quarterly data
                filename = filing.primaryDocument.lower() if filing.primaryDocument else ""
                if any(q in filename for q in ["q1", "q2", "q3", "q4", "quarter"]):
                    six_k_filings.append(filing)
        
        if not six_k_filings:
            return None
        
        # Download and parse 6-K filings
        data_points = []
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Found {len(six_k_filings)} 6-K filings with quarterly indicators")
        
        for filing in six_k_filings[:20]:  # Limit to most recent 20 to avoid long download times
            try:
                # Check if file already exists in cache
                import pathlib
                download_dir = pathlib.Path(self.download_dir) / str(cik).zfill(10) / filing.accessionNumber
                download_path = download_dir / f"{filing.accessionNumber}.txt"
                
                if download_path.exists():
                    logger.debug(f"Using cached filing: {filing.accessionNumber}")
                else:
                    logger.info(f"Downloading filing: {filing.accessionNumber}")
                    download_path = await self.download_service.download(
                        cik=cik,
                        accession_number=filing.accessionNumber,
                        file_to_download=f"{filing.accessionNumber}.txt"
                    )
                
                # Read file
                with open(download_path, 'rb') as f:
                    txt_content = f.read().decode('utf-8', errors='ignore')
                
                # Extract Exhibit 99.x (press release)
                exhibit_html = self.parser.extract_exhibit(txt_content, exhibit_pattern="EX-99")
                if not exhibit_html:
                    logger.debug(f"No EX-99 exhibit found in {filing.accessionNumber}")
                    continue
                
                # Parse HTML tables
                tables = self.parser.parse_html_tables(exhibit_html)
                if not tables:
                    logger.debug(f"No tables found in {filing.accessionNumber}")
                    continue
                
                logger.info(f"Found {len(tables)} tables in {filing.accessionNumber}")
                
                # Extract financial metrics
                metrics = self.parser.extract_financial_metrics(tables)
                if not metrics:
                    logger.debug(f"No metrics extracted from {filing.accessionNumber}")
                    continue
                
                logger.info(f"Extracted metrics from {filing.accessionNumber}: {list(metrics.keys())}")
                
                # Create data point
                # Extract fiscal period from filename (q1, q2, q3, q4)
                filename_lower = filing.primaryDocument.lower() if filing.primaryDocument else ""
                fiscal_period = None
                for q in ["q1", "q2", "q3", "q4"]:
                    if q in filename_lower:
                        fiscal_period = q.upper()
                        break
                
                if not fiscal_period:
                    logger.debug(f"Could not determine fiscal period for {filing.accessionNumber}")
                    continue
                
                # Extract fiscal year from filing date
                fiscal_year = filing.filingDate.year if hasattr(filing.filingDate, 'year') else int(str(filing.filingDate)[:4])
                
                # Create data points for each metric
                for metric_name, values in metrics.items():
                    # values should be a list of Decimal values
                    if isinstance(values, list) and len(values) > 0:
                        # Use the first value (current quarter)
                        filing_date_str = str(filing.filingDate) if hasattr(filing.filingDate, 'year') else filing.filingDate
                        dp = QuarterlyDataPoint(
                            filing_date=filing_date_str,
                            period_end_date=filing_date_str,
                            fiscal_year=fiscal_year,
                            fiscal_period=fiscal_period,
                            accession_number=filing.accessionNumber,
                            metric_name=metric_name,
                            value=values[0],
                            currency="EUR"
                        )
                        data_points.append(dp)
                        logger.debug(f"Created data point: {metric_name}={values[0]} for {fiscal_period} {fiscal_year}")
                    
            except Exception as e:
                # Skip filings that fail to download/parse
                logger.warning(f"Error processing 6-K filing {filing.accessionNumber}: {e}")
                continue
        
        if not data_points:
            logger.info("No 6-K data points extracted")
            return
        
        # Use SixKCombiner to merge data into CompanyFacts
        logger.info(f"Merging {len(data_points)} data points into CompanyFacts")
        combiner = SixKCombiner(facts)
        combiner.add_6k_quarterly_data(data_points)
        logger.info(f"Successfully merged 6-K data into CompanyFacts")
        
        # Save merged facts back to cache if caching is enabled
        if self.make_cache:
            cache_path = f"edgar_rest/api/xbrl/companyfacts/CIK{cik:010d}.json"
            logger.info(f"Saving merged CompanyFacts to cache: {cache_path}")
            self.cache.save(cache_path, json.dumps(facts.model_dump(mode='json', by_alias=True)))


class CompanyDataResult:
    """
    Container for all loaded company data.
    
    Attributes:
        facts: CompanyFacts from SEC EDGAR API (XBRL data)
        submissions: Complete list of company filings
        combiner: Optional SixKCombiner with merged quarterly data from 6-K filings
    """
    
    def __init__(
        self,
        facts: CompanyFacts,
        submissions: CompanySubmissionsResponse,
        combiner: Optional[SixKCombiner] = None
    ):
        self.facts = facts
        self.submissions = submissions
        self.combiner = combiner
    
    def has_6k_data(self) -> bool:
        """Check if 6-K quarterly data is available."""
        return self.combiner is not None and self.combiner.has_quarterly_data()
