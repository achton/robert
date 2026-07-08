"""Tests for logger configuration."""

import logging

from robot.logger import get_logger


class TestGetLogger:
    """Test suite for get_logger()."""

    def test_returns_logger_with_correct_name(self):
        """Should return a logger with the requested name."""
        logger = get_logger("TestService")
        assert logger.name == "TestService"

    def test_logger_has_handler(self):
        """Should configure the logger with a stream handler."""
        logger = get_logger("HandlerTest")
        assert len(logger.handlers) >= 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_logger_does_not_propagate(self):
        """Should disable propagation to avoid duplicate messages."""
        logger = get_logger("PropagateTest")
        assert logger.propagate is False

    def test_logger_level_is_debug(self):
        """Should set logger level to DEBUG."""
        logger = get_logger("LevelTest")
        assert logger.level == logging.DEBUG
