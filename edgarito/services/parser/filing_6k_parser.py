"""Parser for 6-K filings to extract quarterly financial data from HTML press releases.

This parser extracts Exhibit 99.x documents (typically earnings press releases) from
6-K submission .txt files and parses HTML tables to extract structured financial data.

Pattern: Foreign companies filing 20-F (annual) use 6-K for quarterly earnings reports.
These 6-K filings contain Exhibit 99.1 or similar with HTML press releases containing
structured financial tables.

Tested with: Ferrari (RACE), Stellantis (STLA), Novo Nordisk (NVO), TSMC (TSM), SAP
"""

import re
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup
from decimal import Decimal


class Filing6KParser:
    """Parser for 6-K filings with HTML press release exhibits."""

    @staticmethod
    def extract_exhibit(txt_content: str, exhibit_pattern: str = "EX-99") -> Optional[str]:
        """Extract exhibit document from SEC .txt submission.
        
        Args:
            txt_content: Full text content of SEC submission .txt file
            exhibit_pattern: Pattern to match exhibit type (e.g., "EX-99" for EX-99.1, EX-99.2)
            
        Returns:
            HTML content of the exhibit, or None if not found
            
        Example:
            Typical 6-K structure:
            <DOCUMENT><TYPE>6-K</TYPE>...cover page...</DOCUMENT>
            <DOCUMENT><TYPE>EX-99.1</TYPE>...press release...</DOCUMENT>
        """
        # Pattern to find document blocks with matching exhibit type
        # SEC format: <TYPE>EX-99.1 (no closing tag, on separate line)
        # Look for <TYPE>EX-99.x followed by <TEXT> tag, extract until </TEXT> or </DOCUMENT>
        
        # Pattern: Find <TYPE>EX-99.x line, then find <TEXT> tag, extract content until closing tag
        # The format looks like:
        # <TYPE>EX-99.1
        # <SEQUENCE>2
        # <FILENAME>...
        # <TEXT>
        # ...content...
        # </TEXT>
        # </DOCUMENT>
        
        pattern = rf'<TYPE>({exhibit_pattern}(?:\.\d+)?)\s+.*?<TEXT>\s*(.*?)(?:</TEXT>|</DOCUMENT>)'
        
        matches = re.findall(pattern, txt_content, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Return content of first matching exhibit (usually EX-99.1)
            exhibit_content = matches[0][1].strip()
            return exhibit_content
        
        return None

    @staticmethod
    def parse_html_tables(html_content: str) -> List[BeautifulSoup]:
        """Parse all HTML tables from press release content.
        
        Args:
            html_content: HTML content of press release exhibit
            
        Returns:
            List of BeautifulSoup table objects
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        return tables

    @staticmethod
    def extract_table_data(table: BeautifulSoup) -> List[List[str]]:
        """Extract raw data from HTML table as 2D list.
        
        Args:
            table: BeautifulSoup table object
            
        Returns:
            2D list where each inner list is a row of cell text values
        """
        rows = []
        for tr in table.find_all('tr'):
            cells = []
            for cell in tr.find_all(['td', 'th']):
                # Get text and clean whitespace
                text = cell.get_text(strip=True)
                cells.append(text)
            if cells:  # Only add non-empty rows
                rows.append(cells)
        return rows

    @staticmethod
    def _clean_numeric(value: str) -> Optional[Decimal]:
        """Clean and convert numeric string to Decimal.
        
        Handles:
        - Thousands separators (1,644)
        - Negative values in parentheses (332) or with minus -332
        - Percentage signs 28.4%
        - Currency symbols €1,644
        - Empty or non-numeric values
        
        Args:
            value: String representation of number
            
        Returns:
            Decimal value or None if not parseable
        """
        if not value or value == '-' or value == '—':
            return None
        
        # Remove common non-numeric characters
        cleaned = value.replace(',', '').replace('%', '').replace('€', '').replace('$', '').strip()
        
        # Handle parentheses as negative (accounting format)
        if cleaned.startswith('(') and cleaned.endswith(')'):
            cleaned = '-' + cleaned[1:-1]
        
        try:
            return Decimal(cleaned)
        except:
            return None

    @staticmethod
    def extract_financial_metrics(tables: List[BeautifulSoup], simplify=True) -> Dict[str, Any]:
        """Extract financial metrics from parsed HTML tables.
        
        Looks for common financial statement line items:
        - Net revenues / Revenue / Sales
        - EBIT / Operating income
        - EBITDA
        - Net profit / Net income / Net earnings
        - EPS / Earnings per share
        - Operating cash flow / Cash flow from operations
        
        Args:
            tables: List of parsed HTML table objects
            simplify: If True, returns only first 4 numeric values per metric (current/prior period)
            
        Returns:
            Dictionary mapping metric names to values
            
        Example return structure (simplified):
            {
                "net_revenues": [Decimal("1644"), Decimal("1544"), Decimal("4941"), Decimal("4447")],
                "ebit": [Decimal("467"), Decimal("423"), Decimal("1420"), Decimal("1245")],
                ...
            }
        """
        metrics = {}
        
        # Common metric keywords to search for (case-insensitive)
        # Order matters: check more specific patterns first to avoid substring matches
        metric_patterns = {
            'net_revenues': ['net revenues', 'total revenues', 'net sales', 'total revenue'],
            'ebitda': ['ebitda', 'adjusted ebitda'],  # Check before EBIT to avoid substring match
            'ebit': ['ebit /', 'ebit/', 'adj. ebit', 'operating income', 'operating profit'],  # More specific patterns
            'net_profit': ['net profit', 'net income', 'net earnings', 'profit for the period'],
            'eps': ['basic eps', 'diluted eps', 'earnings per share'],
            'operating_cash_flow': ['operating cash flow', 'cash flow from operations', 
                                   'cash from operating activities']
        }
        
        for table_idx, table in enumerate(tables):
            rows = Filing6KParser.extract_table_data(table)
            
            if not rows or len(rows) < 2:
                continue
            
            # Search for metric rows - check ALL cells for metric labels
            for row in rows:
                if not row or len(row) < 2:
                    continue
                
                # Check all cells in the row for metric labels
                matched_metric = None
                metric_cell_idx = None
                
                for cell_idx, cell in enumerate(row):
                    cell_lower = cell.lower().strip()
                    
                    # Match against known metric patterns
                    for metric_key, patterns in metric_patterns.items():
                        if any(pattern in cell_lower for pattern in patterns):
                            matched_metric = metric_key
                            metric_cell_idx = cell_idx
                            break
                    
                    if matched_metric:
                        break
                
                if not matched_metric:
                    continue
                
                # Extract numeric values from the row (excluding the label cell)
                values = []
                for cell_idx, cell in enumerate(row):
                    if cell_idx == metric_cell_idx:
                        continue  # Skip the label cell
                    
                    value = Filing6KParser._clean_numeric(cell)
                    if value is not None and value > 0:  # Only positive meaningful values
                        values.append(value)
                
                # Store metric (only if not already found to avoid duplicates)
                if matched_metric not in metrics:
                    if simplify:
                        # Take first 4 values (typically: Q current, Q prior, YTD current, YTD prior)
                        metrics[matched_metric] = values[:4]
                    else:
                        metrics[matched_metric] = values
        
        return metrics

    @staticmethod
    def parse_6k_filing(txt_content: str) -> Dict[str, Any]:
        """Complete parsing pipeline for 6-K filing.
        
        Args:
            txt_content: Full text content of SEC 6-K submission .txt file
            
        Returns:
            Dictionary with extracted financial metrics
            
        Example usage:
            from edgarito.services.downloader.download_service import DownloadService
            
            downloader = DownloadService()
            txt_content = downloader.fetch(cik="0001648416", 
                                          accession_number="0001648416-24-000194",
                                          document_name="0001648416-24-000194.txt")
            
            parser = Filing6KParser()
            metrics = parser.parse_6k_filing(txt_content)
        """
        # Step 1: Extract Exhibit 99.x (press release)
        exhibit_html = Filing6KParser.extract_exhibit(txt_content, exhibit_pattern="EX-99")
        
        if not exhibit_html:
            return {"error": "No Exhibit 99.x found in filing"}
        
        # Step 2: Parse HTML tables
        tables = Filing6KParser.parse_html_tables(exhibit_html)
        
        if not tables:
            return {"error": "No HTML tables found in exhibit"}
        
        # Step 3: Extract financial metrics
        metrics = Filing6KParser.extract_financial_metrics(tables)
        
        return {
            "success": True,
            "tables_found": len(tables),
            "metrics": metrics
        }
