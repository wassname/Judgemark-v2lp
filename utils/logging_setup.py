import os
import logging

def setup_logging(verbosity: str):
    """Set up logging based on verbosity level."""
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = log_levels.get(verbosity.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_verbosity(args_verbosity: str) -> str:
    """Determine the verbosity level from command-line or environment."""
    if args_verbosity:
        return args_verbosity
    return os.getenv("LOG_VERBOSITY", "INFO")