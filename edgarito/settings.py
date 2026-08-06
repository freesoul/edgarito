import os

from dotenv import find_dotenv, load_dotenv


# Keep compatibility with the existing CLI configuration, then load any
# additional values from the conventional .env file. Existing shell variables
# are never overwritten.
for dotenv_name in (".cli.env", ".env"):
    dotenv_path = find_dotenv(dotenv_name, usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

EDGARITO_CACHE_DIR = os.getenv("EDGARITO_CACHE_DIR") or os.getenv(
    "cache_path", "cache"
)
EDGARITO_USER_AGENT = os.getenv("EDGARITO_USER_AGENT") or os.getenv("user_agent")
