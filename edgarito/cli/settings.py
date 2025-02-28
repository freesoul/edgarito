from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Used by pydantic_settings
    model_config = SettingsConfigDict(cli_parse_args=True, env_file=".cli.env", extra="ignore", cli_ignore_unknown_args=True)

    # Common cli settings
    log_level: int = 20
    user_agent: str
    cache_path: str = "./cache"
    taxonomy_url: Optional[str] = None


settings = Settings()