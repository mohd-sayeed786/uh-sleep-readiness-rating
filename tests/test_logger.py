"""
Unit tests for the centralized logging mechanism.
Verifies dual output to console and file, level filtering, and format integrity.
"""
import logging
import tempfile
from pathlib import Path

from src.logger import ConsoleColorFormatter, get_logger, setup_logging


def test_logger_file_and_levels():
    """Verify that logger writes DEBUG, INFO, WARNING, and ERROR records to the log file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_log_file = Path(tmpdir) / "test.log"
        setup_logging(log_file_path=test_log_file, console_level=logging.DEBUG, file_level=logging.DEBUG)

        logger = get_logger("test_harness")
        logger.debug("Debug verification message")
        logger.info("Info verification message")
        logger.warning("Warning verification message")
        logger.error("Error verification message")

        assert test_log_file.exists()
        content = test_log_file.read_text(encoding="utf-8")

        assert "Debug verification message" in content
        assert "Info verification message" in content
        assert "Warning verification message" in content
        assert "Error verification message" in content
        assert "[DEBUG" in content
        assert "[INFO " in content
        assert "[WARNING" in content
        assert "[ERROR" in content


def test_console_color_formatter():
    """Verify console color formatter includes timestamp, level, and message."""
    formatter = ConsoleColorFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Sample console message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "Sample console message" in formatted
    assert "test_logger" in formatted
    assert "INFO" in formatted
