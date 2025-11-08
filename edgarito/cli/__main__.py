"""CLI entry point for edgarito."""

import logging

from edgarito.cli.logger import configure_logger
from edgarito.cli.settings import settings
from edgarito.cli.commands import create_app


if __name__ == "__main__":
    configure_logger(settings.log_level)
    logging.debug(f"Using log level {settings.log_level}")
    
    app = create_app()
    app()
