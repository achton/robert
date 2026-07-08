"""Tests for hardware detection utilities."""

from unittest.mock import patch

from robot.hardware import detect_display


class TestDetectDisplay:
    """Test suite for detect_display()."""

    def test_detects_x11(self):
        """Should detect X11 via DISPLAY env var."""
        with patch.dict("os.environ", {"DISPLAY": ":0"}, clear=True):
            assert detect_display() is True

    def test_detects_wayland(self):
        """Should detect Wayland via WAYLAND_DISPLAY env var."""
        with patch.dict(
            "os.environ", {"WAYLAND_DISPLAY": "wayland-0"}, clear=True
        ):
            assert detect_display() is True

    def test_detects_framebuffer(self):
        """Should detect framebuffer at /dev/fb0."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("os.path.exists", return_value=True),
        ):
            assert detect_display() is True

    def test_returns_false_when_no_display(self):
        """Should return False when no display is available."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("os.path.exists", return_value=False),
        ):
            assert detect_display() is False
