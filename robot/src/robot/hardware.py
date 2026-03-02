"""
Hardware detection utilities.

Provides functions to detect available hardware (audio devices, displays).
Services use these to gracefully disable themselves when required hardware
is missing.
"""

import logging
import os

logger = logging.getLogger("Hardware")


def detect_audio_device(device_name: str) -> bool:
    """
    Check if a specific audio device is available.

    Args:
        device_name: Name or partial name of the audio device to find

    Returns:
        True if device is found, False otherwise
    """
    try:
        import sounddevice as sd

        devices = sd.query_devices()

        for device in devices:
            if device_name.lower() in device.get("name", "").lower():
                logger.info(f"Found audio device: {device.get('name')}")
                return True

        logger.warning(f"Audio device '{device_name}' not found")
        return False

    except Exception as e:
        logger.warning(f"Failed to query audio devices: {e}")
        return False


def detect_display() -> bool:
    """
    Check if a display is available.

    On the laptop this checks for X11 or Wayland. On the Pi (headless)
    this checks for the framebuffer device at /dev/fb0.

    Returns:
        True if display is available, False otherwise.
    """
    # Check for X11 display
    if "DISPLAY" in os.environ:
        logger.info("X11 display detected")
        return True

    # Check for Wayland display
    if "WAYLAND_DISPLAY" in os.environ:
        logger.info("Wayland display detected")
        return True

    # Check for framebuffer (Pi with DSI touchscreen)
    if os.path.exists("/dev/fb0"):
        logger.info("Framebuffer display detected (/dev/fb0)")
        return True

    logger.warning("No display detected")
    return False
