"""
Centralized logging configuration for Ring AI Readiness Score system.
Provides dual output:
  1. Console / Command prompt: Colorized logs showing DEBUG, INFO, WARNING, ERROR levels.
  2. File: Persistent structured logs saved to logs/app.log.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.config import LOG_FILE, LOGS_DIR


class ConsoleColorFormatter(logging.Formatter):
    """Custom formatter adding ANSI color escapes for command prompt output."""

    GREY = "\x1b[38;20m"
    CYAN = "\x1b[36m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    FORMAT = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    LEVEL_COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, self.GREY)
        formatted_time = self.formatTime(record, self.DATEFMT)
        level_name = f"{color}{record.levelname:<5}{self.RESET}"
        return f"{self.GREY}{formatted_time}{self.RESET} [{level_name}] [{self.CYAN}{record.name}{self.RESET}]: {record.getMessage()}"


def setup_logging(
    log_file_path: Optional[Path] = None,
    console_level: int = logging.DEBUG,
    file_level: int = logging.DEBUG,
) -> None:
    """
    Initialize root logging with console and rotating file handlers.
    Can be configured via LOG_LEVEL environment variable (e.g. LOG_LEVEL=INFO).
    """
    env_level_name = os.getenv("LOG_LEVEL", "").upper()
    if env_level_name in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        console_level = getattr(logging, env_level_name)
        file_level = getattr(logging, env_level_name)

    target_log_file = log_file_path or LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(min(console_level, file_level))

    # Prevent duplicate handlers if setup_logging is called multiple times
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # 1. Console Handler (Command Prompt)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ConsoleColorFormatter())
    root_logger.addHandler(console_handler)

    # 2. File Handler (Persistent Log File in logs/app.log)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-5s] [%(name)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        filename=str(target_log_file),
        maxBytes=10 * 1024 * 1024,  # 10 MB per log file
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)


def get_logger(name: str = "AS_UH_Project") -> logging.Logger:
    """
    Retrieve a named logger instance configured with console and file output.
    Initializes setup_logging() automatically if not already configured.
    """
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        setup_logging()
    return logging.getLogger(name)
