"""
Service for fetching market data from Yahoo Finance.
"""
import logging
from typing import Optional

try:
    import yfinance as yf
except ImportError:
    yf = None

from .market_data_schemas import MarketData

logger = logging.getLogger(__name__)


class YahooFinanceService:
    """Service to fetch stock market data from Yahoo Finance."""
    
    def __init__(self):
        """Initialize the Yahoo Finance service."""
        if yf is None:
            raise ImportError("yfinance package not installed")
        
    async def get_market_data(self, ticker: str) -> Optional[MarketData]:
        """
        Fetch market data for a given ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'TSLA', 'AAPL')
            
        Returns:
            MarketData object if successful, None if data unavailable or error
        """
        if yf is None:
            logger.error("yfinance package not available")
            return None
        
        try:
            ticker = ticker.strip().upper()
            logger.info(f"Fetching market data for ticker: {ticker}")
            
            # Fetch stock info from Yahoo Finance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract market cap (handle various possible keys)
            market_cap = info.get('marketCap') or info.get('market_cap')
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if market_cap is None:
                logger.warning(
                    f"No market cap data found for ticker {ticker}. "
                    "This might be an invalid ticker or delisted company."
                )
                # Still return a MarketData object with None values
                return MarketData(
                    ticker=ticker,
                    market_cap=None,
                    current_price=current_price
                )
            
            logger.info(
                f"Successfully fetched market data for {ticker}: "
                f"market_cap=${market_cap:,.0f}, price=${current_price or 0:.2f}"
            )
            
            return MarketData(
                ticker=ticker,
                market_cap=float(market_cap) if market_cap else None,
                current_price=float(current_price) if current_price else None
            )
            
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}", exc_info=True)
            return None
    
    def get_market_data_sync(self, ticker: str) -> Optional[MarketData]:
        """
        Synchronous version of get_market_data for non-async contexts.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            MarketData object if successful, None otherwise
        """
        if yf is None:
            logger.error("yfinance package not available")
            return None
        
        try:
            ticker = ticker.strip().upper()
            logger.info(f"Fetching market data (sync) for ticker: {ticker}")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            market_cap = info.get('marketCap') or info.get('market_cap')
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if market_cap is None:
                logger.warning(f"No market cap data found for ticker {ticker}")
                return MarketData(
                    ticker=ticker,
                    market_cap=None,
                    current_price=current_price
                )
            
            logger.info(
                f"Successfully fetched market data for {ticker}: "
                f"market_cap=${market_cap:,.0f}"
            )
            
            return MarketData(
                ticker=ticker,
                market_cap=float(market_cap) if market_cap else None,
                current_price=float(current_price) if current_price else None
            )
            
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}", exc_info=True)
            return None
