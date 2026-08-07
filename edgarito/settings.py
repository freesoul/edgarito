import os

from dotenv import find_dotenv, load_dotenv

from edgarito.config.providers import (
    ClassificationProviderConfiguration,
    ProviderConfiguration,
)

# Keep compatibility with the existing CLI configuration, then load any
# additional values from the conventional .env file. Existing shell variables
# are never overwritten.
for dotenv_name in (".cli.env", ".env"):
    dotenv_path = find_dotenv(dotenv_name, usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

EDGARITO_CACHE_DIR = os.getenv("EDGARITO_CACHE_DIR") or os.getenv("cache_path", "cache")
EDGARITO_USER_AGENT = os.getenv("EDGARITO_USER_AGENT") or os.getenv("user_agent")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv(
    "alphavantage_api_key"
)
FMP_API_KEY = os.getenv("FMP_API_KEY") or os.getenv("fmp_key")
FRED_API_KEY = os.getenv("FRED_API_KEY") or os.getenv("fred_api_key")
MASSIVE_API_KEY = (
    os.getenv("MASSIVE_API_KEY")
    or os.getenv("massive_api_key")
    or os.getenv("POLYGON_API_KEY")
)
OPENFIGI_API_KEY = os.getenv("OPENFIGI_API_KEY") or os.getenv("openfigi_api_key")
OPENAI_API_KEY = os.getenv("openai_secret_api_key") or os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")
PROVIDER_CONFIGURATION = ProviderConfiguration.from_environment(os.environ)
CLASSIFICATION_PROVIDER_CONFIGURATION = (
    ClassificationProviderConfiguration.from_environment(os.environ)
)
