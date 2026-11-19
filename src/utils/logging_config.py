"""Logging configuration for the Quantum Architecture Optimizer.

Provides structured, configurable logging with file and console output.
All modules should use get_logger() to obtain their logger instance.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Module-level state
_initialized = False
_log_dir: Optional[Path] = None


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
    log_file: str | None = None,
    console: bool = True,
) -> None:
    """Set up the logging system for the project.

    Should be called once at application startup. Subsequent calls
    update the root logger's level but do not add duplicate handlers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. Defaults to experiments/logs/.
        log_file: Log filename. Defaults to 'qao_YYYYMMDD_HHMMSS.log'.
        console: Whether to output to console (stderr).
    """
    global _initialized, _log_dir

    # Resolve log level
    env_level = os.environ.get("LOG_LEVEL", "").upper()
    log_level = getattr(logging, env_level or level.upper(), logging.INFO)

    # Root logger for the project
    root_logger = logging.getLogger("qao")
    root_logger.setLevel(log_level)

    if _initialized:
        # Just update the level, don't add new handlers
        root_logger.setLevel(log_level)
        return

    # Formatter
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(fmt)
        root_logger.addHandler(console_handler)

    # File handler
    if log_dir is not None or log_file is not None:
        if log_dir is None:
            from src.utils.config_loader import get_project_root
            log_dir = get_project_root() / "experiments" / "logs"

        _log_dir = Path(log_dir)
        _log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"qao_{timestamp}.log"

        file_handler = logging.FileHandler(_log_dir / log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # File always gets DEBUG
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    All loggers are children of the 'qao' root logger.

    Args:
        name: Logger name, typically __name__ of the calling module.
              Prefix 'qao.' is added automatically if not present.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    if not name.startswith("qao."):
        # Convert module paths like 'src.utils.config_loader' to 'qao.utils.config_loader'
        parts = name.split(".")
        if parts[0] == "src":
            parts = parts[1:]
        name = "qao." + ".".join(parts)

    return logging.getLogger(name)
