"""
Configuration helpers.

Base configuration utilities. Service-specific config dataclasses are added
in their respective phases.
"""

from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env file if it exists (for API keys, secrets, etc.)
load_dotenv()


@dataclass
class GUIConfig:
    """Configuration for the GUI service (display and expressions)."""

    width: int = 800
    height: int = 480
    fps: int = 60
    default_expression: str = "neutral"


def is_raspberry_pi() -> bool:
    """
    Detect if running on a Raspberry Pi.

    Reads /proc/device-tree/model which is present on all Pi models.

    Returns:
        True if running on Raspberry Pi, False otherwise
    """
    try:
        with open("/proc/device-tree/model") as f:
            model = f.read().lower()
            return "raspberry pi" in model
    except FileNotFoundError:
        return False
