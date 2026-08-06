"""
Reference module for comprehensive SEC filing types.

This module contains ALL SEC filing types for reference purposes.
For day-to-day use, prefer CoreFilingType from enums.edgar.core_filing_type
which contains only the essential filing types needed for investment analysis.

NOTE: This comprehensive enum is maintained for:
- Edge case handling
- Research and exploration
- Validating obscure filing types
- Historical reference

Most applications should use CoreFilingType instead.
"""

# Re-export from the reference implementation
from edgarito.enums.edgar.reference.all_filing_types import FilingType

__all__ = ["FilingType"]
