"""Tests for configuration helpers."""

from unittest.mock import mock_open, patch

from robot.config import is_raspberry_pi


class TestIsRaspberryPi:
    """Test suite for is_raspberry_pi()."""

    def test_returns_true_on_raspberry_pi(self):
        """Should return True when /proc/device-tree/model contains 'raspberry pi'."""
        fake_model = "Raspberry Pi 4 Model B Rev 1.4\x00"
        with patch("builtins.open", mock_open(read_data=fake_model)):
            assert is_raspberry_pi() is True

    def test_returns_false_on_other_hardware(self):
        """Should return False for non-Pi hardware."""
        fake_model = "NVIDIA Jetson Nano\x00"
        with patch("builtins.open", mock_open(read_data=fake_model)):
            assert is_raspberry_pi() is False

    def test_returns_false_when_file_missing(self):
        """Should return False when /proc/device-tree/model doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert is_raspberry_pi() is False
