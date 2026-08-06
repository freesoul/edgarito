# Edgarito

**A comprehensive Python toolkit for SEC EDGAR data analysis and financial statement extraction.**

Edgarito provides an async client for the SEC EDGAR REST API with structured Pydantic schemas, powerful financial statement readers with robust data deduplication, computed financial metrics, and automated red flag detection for investment analysis.

├── enums/                  # Enumerations (filing types, periods, granularity)
├── schemas/               # Pydantic models for API responses
├── services/
│   ├── retrieval/         # EDGAR REST API clients
│   ├── cache/             # FileSystem caching
│   ├── financial/         # Financial statement readers
│   └── analysis/          # Red flags and metrics
└── examples/              # Usage examples
```

## Contributing

Contributions welcome! Areas for improvement:

- Additional financial metrics
- More red flag detection rules  
- Support for IFRS (currently US-GAAP only)
- Direct XBRL parsing from filings (as fallback to REST API)

## License

MIT License - see LICENSE file for details

## Disclaimer

This tool is for informational purposes only. Not financial advice. Always verify data with official SEC filings at https://www.sec.gov/edgar.