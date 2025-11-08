from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedFlagsThresholds(BaseModel):
    """Configurable thresholds for red flag detection"""
    
    # Balance Sheet Health - Tiered thresholds
    debt_to_equity_ratio_warning: float = Field(default=1.0, description="D/E WARNING threshold")
    debt_to_equity_ratio_critical: float = Field(default=2.0, description="D/E CRITICAL threshold")
    
    current_ratio_critical: float = Field(default=1.0, description="Current ratio CRITICAL threshold")
    current_ratio_warning: float = Field(default=1.5, description="Current ratio WARNING threshold")
    
    quick_ratio_warning: float = Field(default=1.0, description="Quick ratio WARNING threshold")
    quick_ratio_critical: float = Field(default=0.5, description="Quick ratio CRITICAL threshold")
    
    interest_coverage_critical: float = Field(default=1.5, description="Interest coverage CRITICAL threshold")
    interest_coverage_warning: float = Field(default=3.0, description="Interest coverage WARNING threshold")
    
    # Cash Flow Quality
    stock_comp_percent_ocf: float = Field(default=10.0, description="Stock comp as % of OCF")
    
    # Profitability & Income Quality
    net_margin_percent: float = Field(default=3.0, description="Net margin % threshold")
    roe_percent: float = Field(default=10.0, description="ROE % threshold")
    roic_percent: float = Field(default=7.0, description="ROIC % threshold")
    gross_margin_std_dev: float = Field(default=5.0, description="Gross margin volatility")
    
    # Growth & Sustainability
    revenue_cagr_inflation: float = Field(default=3.0, description="Revenue CAGR threshold")
    sga_percent_revenue_warning: float = Field(default=30.0, description="SG&A WARNING threshold")
    sga_percent_revenue_info: float = Field(default=25.0, description="SG&A INFO threshold")
    sga_increase_threshold: float = Field(default=3.0, description="SG&A increase %")
    revenue_growth_for_rd_check: float = Field(default=5.0, description="Revenue growth for R&D check")
    rd_decline_threshold: float = Field(default=-5.0, description="R&D decline threshold")
    
    # Valuation Concerns
    price_to_sales: float = Field(default=10.0, description="P/S ratio threshold")
    pe_ratio_low: float = Field(default=5.0, description="Low P/E threshold")
    price_to_book: float = Field(default=5.0, description="P/B ratio threshold")
    roe_for_pb_check: float = Field(default=15.0, description="ROE needed for high P/B")
    dividend_yield: float = Field(default=8.0, description="Dividend yield threshold")
    peg_ratio: float = Field(default=2.0, description="PEG ratio threshold")
    ev_to_ebitda: float = Field(default=15.0, description="EV/EBITDA threshold")
    short_interest_percent: float = Field(default=10.0, description="Short interest threshold")
    insider_ownership_percent: float = Field(default=2.0, description="Insider ownership threshold")


class Settings(BaseSettings):
    # Used by pydantic_settings
    model_config = SettingsConfigDict(
        cli_parse_args=False, 
        env_file=".cli.env", 
        extra="ignore", 
        cli_ignore_unknown_args=True,
        env_nested_delimiter="__"
    )

    # Common cli settings
    log_level: int = 20
    user_agent: str
    cache_path: str = "./cache"
    taxonomy_url: Optional[str] = None
    
    # Red Flags Thresholds (nested model, can override with red_flags__field_name in env)
    red_flags: RedFlagsThresholds = Field(default_factory=RedFlagsThresholds)


settings = Settings()