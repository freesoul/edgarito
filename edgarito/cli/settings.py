from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

from edgarito.enums.cli.actions import Action


class Settings(BaseSettings):
    # Used by pydantic_settings
    model_config = SettingsConfigDict(cli_parse_args=True, env_file=".env")

    # Shared params
    cik: Optional[int] = None
    ticker: Optional[str] = None

    # Cli settings
    action: Action
    user_agent: str
    cache_path: str = "./cache"
    use_cache: bool = True
    make_cache: bool = True

    # Action FIND_TICKER_CIK
    ticker: Optional[str] = None


settings = Settings()
