"""
Logging utilities for the Global Autonomous Knowledge Engine.
Provides structured logging with rotating file handlers.
"""

import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logger(
    log_level: str = "INFO",
    log_file: str = "logs/gake.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    Configure the root logger with both console and file handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        max_bytes: Maximum size of each log file
        backup_count: Number of backup files to keep
    """
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy_logger in ["urllib3", "asyncio", "aiohttp", "httpx"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logging.getLogger("gake").info("GAKE logger initialized")


class StructuredLogger:
    """Logger that outputs structured JSON logs for production use."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def info(self, msg: str, **kwargs):
        self._logger.info(self._format(msg, kwargs))

    def warning(self, msg: str, **kwargs):
        self._logger.warning(self._format(msg, kwargs))

    def error(self, msg: str, **kwargs):
        self._logger.error(self._format(msg, kwargs))

    def debug(self, msg: str, **kwargs):
        self._logger.debug(self._format(msg, kwargs))

    def _format(self, msg: str, extra: dict) -> str:
        if extra:
            extra_str = " | ".join(f"{k}={v}" for k, v in extra.items())
            return f"{msg} | {extra_str}"
        return msg
