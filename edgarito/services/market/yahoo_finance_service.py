"""
Service for fetching market data from Yahoo Finance.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    yf = None

from edgarito.schemas.market_data import MarketData
from edgarito.services.cache.filesystem_cache import FileSystemCache

logger = logging.getLogger(__name__)


class YahooFinanceService:
    """Service to fetch stock market data from Yahoo Finance."""
    
    def __init__(
        self,
        cache: Optional[FileSystemCache] = None,
        cache_expiry_hours: int = 24
    ):
        """
        Initialize the Yahoo Finance service.
        
        Args:
            cache: Optional FileSystemCache for caching market data
            cache_expiry_hours: Hours before cached data is considered stale (default: 24)
        """
        if yf is None:
            raise ImportError("yfinance package not installed")
        
        self._cache = cache
        self._cache_expiry_hours = cache_expiry_hours
        
    def _get_cache_path(self, ticker: str) -> str:
        """
        Get the cache file path for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Cache file path string
        """
        return f"yahoo_finance/{ticker.upper()}.json"
    
    def _is_cache_valid(self, cached_data: MarketData) -> bool:
        """
        Check if cached data is still valid based on expiry time.
        
        Args:
            cached_data: Previously cached MarketData
            
        Returns:
            True if cache is still valid, False if expired
        """
        if not cached_data.timestamp:
            return False
        
        age = datetime.now() - cached_data.timestamp
        return age < timedelta(hours=self._cache_expiry_hours)
    
    async def get_market_data(
        self,
        ticker: str,
        use_cache: bool = True,
        make_cache: bool = True
    ) -> Optional[MarketData]:
        """
        Fetch market data for a given ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'TSLA', 'AAPL')
            use_cache: Whether to use cached data if available
            make_cache: Whether to cache the fetched data
            
        Returns:
            MarketData object if successful, None if data unavailable or error
        """
        if yf is None:
            logger.error("yfinance package not available")
            return None
        
        ticker = ticker.strip().upper()
        
        # Try to load from cache
        if use_cache and self._cache:
            cache_path = self._get_cache_path(ticker)
            cached_content = self._cache.read(cache_path)
            
            if cached_content:
                try:
                    cached_data = MarketData(**json.loads(cached_content))
                    if self._is_cache_valid(cached_data):
                        logger.info(
                            f"Using cached market data for {ticker} "
                            f"(age: {(datetime.now() - cached_data.timestamp).seconds // 3600}h)"
                        )
                        return cached_data
                    else:
                        logger.info(f"Cached data for {ticker} expired, fetching fresh data")
                except Exception as e:
                    logger.warning(f"Failed to load cached data for {ticker}: {e}")
        
        # Fetch from Yahoo Finance
        try:
            logger.info(f"Fetching market data for ticker: {ticker}")
            
            # Fetch stock info from Yahoo Finance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract market cap (handle various possible keys)
            market_cap = info.get('marketCap') or info.get('market_cap')
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            # Extract additional metrics
            enterprise_value = info.get('enterpriseValue')
            ev_to_ebitda = info.get('enterpriseToEbitda')
            peg_ratio = info.get('trailingPegRatio') or info.get('pegRatio')
            short_percent = info.get('shortPercentOfFloat')
            insider_percent = info.get('heldPercentInsiders')
            
            # Convert percentages to 0-100 scale if needed
            short_percent_formatted = short_percent * 100 if short_percent and short_percent < 1 else short_percent
            insider_percent_formatted = insider_percent * 100 if insider_percent and insider_percent < 1 else insider_percent
            
            if market_cap is None:
                logger.warning(
                    f"No market cap data found for ticker {ticker}. "
                    "This might be an invalid ticker or delisted company."
                )
                # Still return a MarketData object with None values
                market_data = MarketData(
                    ticker=ticker,
                    market_cap=None,
                    current_price=current_price,
                    enterprise_value=float(enterprise_value) if enterprise_value else None,
                    ev_to_ebitda=float(ev_to_ebitda) if ev_to_ebitda else None,
                    peg_ratio=float(peg_ratio) if peg_ratio else None,
                    short_percent_float=float(short_percent_formatted) if short_percent_formatted else None,
                    insider_ownership_percent=float(insider_percent_formatted) if insider_percent_formatted else None
                )
            else:
                logger.info(
                    f"Successfully fetched market data for {ticker}: "
                    f"market_cap=${market_cap:,.0f}, price=${current_price or 0:.2f}"
                )
                
                market_data = MarketData(
                    ticker=ticker,
                    market_cap=float(market_cap) if market_cap else None,
                    current_price=float(current_price) if current_price else None,
                    enterprise_value=float(enterprise_value) if enterprise_value else None,
                    ev_to_ebitda=float(ev_to_ebitda) if ev_to_ebitda else None,
                    peg_ratio=float(peg_ratio) if peg_ratio else None,
                    short_percent_float=float(short_percent_formatted) if short_percent_formatted else None,
                    insider_ownership_percent=float(insider_percent_formatted) if insider_percent_formatted else None
                )
            
            # Cache the result
            if make_cache and self._cache and market_data.market_cap is not None:
                cache_path = self._get_cache_path(ticker)
                self._cache.save(cache_path, market_data.model_dump_json())
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}", exc_info=True)
            return None
    
    def get_market_data_sync(
        self,
        ticker: str,
        use_cache: bool = True,
        make_cache: bool = True
    ) -> Optional[MarketData]:
        """
        Synchronous version of get_market_data for non-async contexts.
        
        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached data if available
            make_cache: Whether to cache the fetched data
            
        Returns:
            MarketData object if successful, None otherwise
        """
        if yf is None:
            logger.error("yfinance package not available")
            return None
        
        ticker = ticker.strip().upper()
        
        # Try to load from cache
        if use_cache and self._cache:
            cache_path = self._get_cache_path(ticker)
            cached_content = self._cache.read(cache_path)
            
            if cached_content:
                try:
                    cached_data = MarketData(**json.loads(cached_content))
                    if self._is_cache_valid(cached_data):
                        logger.info(
                            f"Using cached market data for {ticker} "
                            f"(age: {(datetime.now() - cached_data.timestamp).seconds // 3600}h)"
                        )
                        return cached_data
                    else:
                        logger.info(f"Cached data for {ticker} expired, fetching fresh data")
                except Exception as e:
                    logger.warning(f"Failed to load cached data for {ticker}: {e}")
        
        # Fetch from Yahoo Finance
        try:
            logger.info(f"Fetching market data (sync) for ticker: {ticker}")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            market_cap = info.get('marketCap') or info.get('market_cap')
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            # Extract additional metrics
            enterprise_value = info.get('enterpriseValue')
            ev_to_ebitda = info.get('enterpriseToEbitda')
            peg_ratio = info.get('trailingPegRatio') or info.get('pegRatio')
            short_percent = info.get('shortPercentOfFloat')
            insider_percent = info.get('heldPercentInsiders')
            
            # Convert percentages to 0-100 scale if needed
            short_percent_formatted = short_percent * 100 if short_percent and short_percent < 1 else short_percent
            insider_percent_formatted = insider_percent * 100 if insider_percent and insider_percent < 1 else insider_percent
            
            if market_cap is None:
                logger.warning(f"No market cap data found for ticker {ticker}")
                market_data = MarketData(
                    ticker=ticker,
                    market_cap=None,
                    current_price=current_price,
                    enterprise_value=float(enterprise_value) if enterprise_value else None,
                    ev_to_ebitda=float(ev_to_ebitda) if ev_to_ebitda else None,
                    peg_ratio=float(peg_ratio) if peg_ratio else None,
                    short_percent_float=float(short_percent_formatted) if short_percent_formatted else None,
                    insider_ownership_percent=float(insider_percent_formatted) if insider_percent_formatted else None
                )
            else:
                logger.info(
                    f"Successfully fetched market data for {ticker}: "
                    f"market_cap=${market_cap:,.0f}"
                )
                
                market_data = MarketData(
                    ticker=ticker,
                    market_cap=float(market_cap) if market_cap else None,
                    current_price=float(current_price) if current_price else None,
                    enterprise_value=float(enterprise_value) if enterprise_value else None,
                    ev_to_ebitda=float(ev_to_ebitda) if ev_to_ebitda else None,
                    peg_ratio=float(peg_ratio) if peg_ratio else None,
                    short_percent_float=float(short_percent_formatted) if short_percent_formatted else None,
                    insider_ownership_percent=float(insider_percent_formatted) if insider_percent_formatted else None
                )
            
            # Cache the result
            if make_cache and self._cache and market_data.market_cap is not None:
                cache_path = self._get_cache_path(ticker)
                self._cache.save(cache_path, market_data.model_dump_json())
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}", exc_info=True)
            return None
