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
import aiohttp

from edgarito.schemas.edgar_responses.company_facts import CompanyFacts
from edgarito.schemas.edgar_responses.submission import CompanySubmissionsResponse
from edgarito.services.edgar_rest_client.low_level_client import EDGARLowLevelClient
from edgarito.services.edgar_rest_client.submissions_client import SubmissionsClient
from edgarito.services.cache.filesystem_cache import FileSystemCache
from edgarito.services.downloader.download_service import DownloadService
from edgarito.services.parser.filing_6k_parser import Filing6KParser
from edgarito.services.merger.combiner_6k import SixKCombiner, QuarterlyDataPoint


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
        use_cache: bool = True,
        make_cache: bool = True
    ):
        """
        Initialize the data loader.
        
        Args:
            session: aiohttp session for HTTP requests
            cache_dir: Directory for caching API responses
            download_dir: Directory for downloaded filing documents
            use_cache: Whether to use cached data
            make_cache: Whether to cache new data
        """
        self.session = session
        self.cache_dir = cache_dir
        self.download_dir = download_dir
        self.use_cache = use_cache
        self.make_cache = make_cache
        
        # Initialize services
        self.cache = FileSystemCache(cache_dir)
        self.edgar_client = EDGARLowLevelClient(session, self.cache)
        self.submissions_client = SubmissionsClient(self.edgar_client)
        self.download_service = DownloadService(session, download_root_dir=download_dir)
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
        cik = await self.edgar_client.get_cik_from_ticker(
            ticker, 
            use_cache=self.use_cache, 
            make_cache=self.make_cache
        )
        
        return await self.load_from_cik(cik)
    
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
        
        # 3. Check if company has 6-K filings (foreign company indicator)
        combiner = await self._try_load_6k_data(cik, submissions, facts)
        
        return CompanyDataResult(
            facts=facts,
            submissions=submissions,
            combiner=combiner
        )
    
    async def _try_load_6k_data(
        self, 
        cik: int, 
        submissions: CompanySubmissionsResponse,
        facts: CompanyFacts
    ) -> Optional[SixKCombiner]:
        """
        Attempt to load and parse 6-K quarterly data for foreign companies.
        
        For companies filing 20-F (annual) instead of 10-K, quarterly data is NOT
        available in CompanyFacts API. These companies file 6-K reports with quarterly
        earnings in HTML press releases (Exhibit 99.1).
        
        Args:
            cik: Central Index Key
            submissions: Company submissions response
            facts: Company facts to add to combiner
            
        Returns:
            SixKCombiner with quarterly data, or None if no 6-K filings found
        """
        # Find 6-K filings with quarterly pattern (q1, q2, q3, q4 in filename/description)
        six_k_filings = []
        for filing in submissions.filings.recent:
            if filing.form == "6-K":
                # Check if filename/description suggests quarterly data
                filename = filing.primaryDocument.lower() if filing.primaryDocument else ""
                if any(q in filename for q in ["q1", "q2", "q3", "q4", "quarter"]):
                    six_k_filings.append(filing)
        
        if not six_k_filings:
            return None
        
        # Download and parse 6-K filings
        data_points = []
        for filing in six_k_filings[:20]:  # Limit to most recent 20 to avoid long download times
            try:
                # Download submission file
                download_path = await self.download_service.download_submission(
                    cik=cik,
                    accession_number=filing.accessionNumber,
                    primary_document=f"{filing.accessionNumber}.txt"
                )
                
                # Read file
                with open(download_path, 'rb') as f:
                    txt_content = f.read().decode('utf-8', errors='ignore')
                
                # Extract Exhibit 99.x (press release)
                exhibit_html = self.parser.extract_exhibit(txt_content, exhibit_pattern="EX-99")
                if not exhibit_html:
                    continue
                
                # Parse HTML tables
                tables = self.parser.parse_html_tables(exhibit_html)
                if not tables:
                    continue
                
                # Extract financial metrics
                metrics = self.parser.extract_financial_metrics(tables)
                if not metrics:
                    continue
                
                # Create data point
                # Extract fiscal period from filename (q1, q2, q3, q4)
                filename_lower = filing.primaryDocument.lower() if filing.primaryDocument else ""
                fiscal_period = None
                for q in ["q1", "q2", "q3", "q4"]:
                    if q in filename_lower:
                        fiscal_period = q.upper()
                        break
                
                if not fiscal_period:
                    continue
                
                # Extract fiscal year from filing date (YYYY-MM-DD)
                fiscal_year = int(filing.filingDate[:4])
                
                # Create data points for each metric
                for metric_name, value in metrics.items():
                    dp = QuarterlyDataPoint(
                        metric_name=metric_name,
                        value=value,
                        fiscal_year=fiscal_year,
                        fiscal_period=fiscal_period,
                        filing_date=filing.filingDate,
                        accession_number=filing.accessionNumber
                    )
                    data_points.append(dp)
                    
            except Exception:
                # Skip filings that fail to download/parse
                continue
        
        if not data_points:
            return None
        
        # Create combiner and add data
        combiner = SixKCombiner()
        combiner.add_company_facts(facts)
        combiner.add_6k_quarterly_data(data_points)
        
        return combiner


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
