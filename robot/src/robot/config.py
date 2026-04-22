"""
Configuration helpers.

Base configuration utilities. Service-specific config dataclasses are added
in their respective phases.
"""

import os
from dataclasses import dataclass, field

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

    # Devices searched in order during init. Falls back to system default,
    # which — on the Pi — is PipeWire routing to the WM8960 sound card.
    # `echo_cancel_source` stays listed for the future AEC setup; it is
    # currently disabled because the PipeWire echo-cancel module breaks
    # PortAudio output on the Pi 4 (see docs/pi-audio-issues.md).
    preferred_input_devices: tuple[str, ...] = ("echo_cancel_source",)

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


@dataclass
class RealtimeConfig:
    """Configuration for the RealtimeService (Gemini Live voice)."""

    # API key — loaded from GEMINI_API_KEY env var. If empty, service
    # disables itself gracefully.
    api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )

    model: str = "gemini-2.5-flash-native-audio-preview-12-2025"
    voice_name: str = "Kore"

    system_instruction: str = (
        "Du er Roberta, en venlig og hjælpsom kontor-robot. "
        "Du taler dansk. Hold dine svar korte og naturlige, "
        "som i en almindelig samtale. Du har en glad og positiv "
        "personlighed."
    )

    # Sent as a hidden text message on connect to make the bot speak first.
    greeting_prompt: str = "Sig hej og præsentér dig selv kort."

    # When False, mic audio is NOT forwarded to the Gemini session.
    # Useful for testing mic hardware without spending API tokens.
    mic_forwarding_enabled: bool = True

    # Sample rates — must match AudioConfig
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000

    # Optional Gemini features
    enable_proactive_audio: bool = True
    enable_affective_dialog: bool = True

    # VAD (Voice Activity Detection) tuning.
    # Lower sensitivity = less likely to trigger on background noise.
    vad_start_sensitivity: str = "START_SENSITIVITY_LOW"
    vad_end_sensitivity: str = "END_SENSITIVITY_LOW"
    vad_silence_duration_ms: int = 500


@dataclass
class DashboardConfig:
    """Configuration for the DashboardService (SSE event stream)."""

    # Turn the dashboard off completely by setting ROBOTA_DASHBOARD=0.
    enabled: bool = field(
        default_factory=lambda: os.getenv("ROBOTA_DASHBOARD", "1") != "0"
    )

    # 0.0.0.0 so a laptop on the same LAN can connect; change to
    # "127.0.0.1" for local-only access.
    host: str = field(
        default_factory=lambda: os.getenv("ROBOTA_DASHBOARD_HOST", "0.0.0.0")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("ROBOTA_DASHBOARD_PORT", "8765"))
    )


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
