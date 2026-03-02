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


@dataclass
class AudioConfig:
    """Configuration for the AudioService."""

    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    channels: int = 1
    chunk_duration_ms: int = 40

    # Devices searched in order during init. Falls back to system default.
    preferred_input_devices: tuple[str, ...] = (
        "echo_cancel_source",
        "seeed-2mic-voicecard",
    )

    # Playback gain (0.0–1.0). Slightly below unity to avoid USB speaker
    # clipping, same as robotv3.
    playback_gain: float = 0.8

    @property
    def input_chunk_size(self) -> int:
        """Number of samples per mic chunk (e.g. 640 at 16 kHz / 40 ms)."""
        return int(self.input_sample_rate * self.chunk_duration_ms / 1000)

    @property
    def output_chunk_size(self) -> int:
        """Number of samples per speaker chunk (e.g. 960 at 24 kHz / 40 ms)."""
        return int(self.output_sample_rate * self.chunk_duration_ms / 1000)


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
