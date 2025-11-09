from typing import Optional, Dict
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SectorThresholdAdjustments(BaseModel):
    """
    Sector-specific threshold multipliers and overrides.
    Values are multipliers applied to base thresholds (e.g., 0.8 = 80% of base threshold).
    """
    
    # OCF/FCF Conversion Thresholds (for cash flow quality)
    ocf_ni_ratio_multiplier: float = Field(default=1.0, description="Multiplier for OCF/NI ratio threshold")
    fcf_ni_ratio_multiplier: float = Field(default=1.0, description="Multiplier for FCF/NI ratio threshold")
    
    # Debt tolerance (for capital-intensive sectors)
    debt_to_equity_multiplier: float = Field(default=1.0, description="Multiplier for D/E thresholds")
    
    # Liquidity expectations
    current_ratio_multiplier: float = Field(default=1.0, description="Multiplier for current ratio thresholds")
    quick_ratio_multiplier: float = Field(default=1.0, description="Multiplier for quick ratio thresholds")
    
    # Profitability expectations
    net_margin_multiplier: float = Field(default=1.0, description="Multiplier for net margin threshold")
    roe_multiplier: float = Field(default=1.0, description="Multiplier for ROE threshold")


def _load_sector_profiles() -> Dict[str, SectorThresholdAdjustments]:
    """
    Load sector profiles from YAML file in settings/ folder.
    Falls back to empty dict if file not found or invalid.
    """
    # Look for sector_profiles.yaml in settings/ folder (3 levels up from this file, then into settings/)
    yaml_path = Path(__file__).parent.parent.parent / "settings" / "sector_profiles.yaml"
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert YAML dict to SectorThresholdAdjustments models
        profiles = {}
        for sector_name, multipliers in data.items():
            if isinstance(multipliers, dict):
                profiles[sector_name] = SectorThresholdAdjustments(**multipliers)
        
        return profiles
    except FileNotFoundError:
        print(f"Warning: sector_profiles.yaml not found at {yaml_path}, using defaults")
        return {}
    except Exception as e:
        print(f"Warning: Error loading sector_profiles.yaml: {e}, using defaults")
        return {}


def _load_red_flags_thresholds() -> 'RedFlagsThresholds':
    """
    Load red flags thresholds from YAML file in settings/ folder.
    Falls back to defaults if file not found or invalid.
    """
    yaml_path = Path(__file__).parent.parent.parent / "settings" / "red_flags.yaml"
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create RedFlagsThresholds from YAML data
        if isinstance(data, dict):
            return RedFlagsThresholds(**data)
        else:
            print(f"Warning: red_flags.yaml has invalid format, using defaults")
            return RedFlagsThresholds()
    except FileNotFoundError:
        print(f"Warning: red_flags.yaml not found at {yaml_path}, using defaults")
        return RedFlagsThresholds()
    except Exception as e:
        print(f"Warning: Error loading red_flags.yaml: {e}, using defaults")
        return RedFlagsThresholds()


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
    # taxonomy_url: Optional[str] = None
    
    # Red Flags Thresholds (loaded from settings/red_flags.yaml, can still override with red_flags__field_name in env)
    red_flags: RedFlagsThresholds = Field(default_factory=_load_red_flags_thresholds)
    
    # Sector Profiles (loaded from settings/sector_profiles.yaml)
    sector_profiles: Dict[str, SectorThresholdAdjustments] = Field(default_factory=_load_sector_profiles)


settings = Settings()