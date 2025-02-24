from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

from edgarito.enums.cli.actions import Action


class Settings(BaseSettings):
    # Used by pydantic_settings
    model_config = SettingsConfigDict(cli_parse_args=True, env_file=".env")

    # Shared params
    user_agent: str
    cache_path: str = "./cache"
    taxonomy_url: Optional[str] = None
    use_cache: bool = True
    make_cache: bool = True
    cik: Optional[int] = None
    ticker: Optional[str] = None
    limit: Optional[int] = None  # Used for reading submissions and downloading

    # Actions
    action: Action

    # Action FIND_TICKER_CIK
    ticker: Optional[str] = None

    # Action DOWNLOAD
    # ranges ...


settings = Settings()

def configure_edgarito(custom_settings: dict):
    """
    Useful when using this project as a package
    """
    global settings
    settings = Settings(**custom_settings)