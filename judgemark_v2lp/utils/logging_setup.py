import os
import sys
# from loguru import logger
from loguru import logger

def setup_logging(verbosity: str):
    """Set up logging based on verbosity level."""
    logger.remove()
    logger.add(sys.stderr, level=verbosity)

def get_verbosity(args_verbosity: str) -> str:
    """Determine the verbosity level from command-line or environment."""
    if args_verbosity:
        return args_verbosity
    return os.getenv("LOG_VERBOSITY", "INFO")
