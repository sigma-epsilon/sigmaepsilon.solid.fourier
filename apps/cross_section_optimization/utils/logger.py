import logging
from functools import lru_cache


@lru_cache(maxsize=1, typed=False)
def get_logger() -> logging.Logger:
    """Returns a logger instance for cross_section_optimization."""
    logger = logging.getLogger("cross_section_optimization")
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_log_level(level: str | int) -> None:
    """Sets log level for the project logger."""
    logger = get_logger()
    logger.setLevel(level)
    logger.info(f"Log level set to {logging.getLevelName(level)}")