"""
Configuration helpers.

Base configuration utilities. Service-specific config dataclasses are added
in their respective phases.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if it exists (for API keys, secrets, etc.)
load_dotenv()

# Roberta's persona prompt lives in the shared assets directory as editable
# markdown, so tweaking her personality doesn't mean editing code.
PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "assets" / "prompt.md"
)

# Used only if prompt.md is missing, so she still has a sane persona.
_FALLBACK_SYSTEM_INSTRUCTION = (
    "Du er Roberta, en venlig kontor-robot. Du taler dansk og holder dine "
    "svar korte og naturlige."
)


def _load_system_instruction() -> str:
    """Read Roberta's persona from assets/prompt.md, or fall back."""
    try:
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return _FALLBACK_SYSTEM_INSTRUCTION


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
class VADConfig:
    """Client-side Silero VAD speech gate.

    Runs the Silero VAD model locally so we only forward *speech* to Gemini.
    Non-speech (HVAC, keyboards, silence) never reaches the model, which
    otherwise hallucinates transcripts from noise. See
    docs/voice-gating-research.md.
    """

    # Master switch. When False, all mic audio is forwarded (old behavior).
    enabled: bool = True

    # Silero VAD ONNX model file, found in the shared assets directory.
    model_filename: str = "silero_vad.onnx"

    # Speech probability threshold (0.0–1.0). A window scoring above this
    # counts as speech. Silero's default is 0.5, but that clipped real
    # speech in the office; 0.3 is more sensitive (opens more readily).
    threshold: float = 0.3

    # Continuous silence required before a speech segment is considered
    # finished (hangover). We keep forwarding audio through this window, so
    # it must exceed the server VAD's silence_duration_ms (500 ms) for
    # Gemini to detect end-of-turn from the trailing silence we send. That
    # is how a turn ends now that we no longer send audio_stream_end.
    min_silence_ms: int = 700

    # Audio kept from just before speech starts, so word onsets aren't
    # clipped. Above Silero's 30 ms default; raised to 300 ms after onsets
    # were still getting clipped in the office.
    speech_pad_ms: int = 300


@dataclass
class RealtimeConfig:
    """Configuration for the RealtimeService (Gemini Live voice)."""

    # API key — loaded from GEMINI_API_KEY env var. If empty, service
    # disables itself gracefully.
    api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )

    model: str = "gemini-2.5-flash-native-audio-latest"
    voice_name: str = "Aoede"

    system_instruction: str = field(default_factory=_load_system_instruction)

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

    # Reconnect backoff (seconds). A dropped or failed Gemini connection is
    # retried instead of stopping the service. The wait grows from min to
    # max on repeated failures and resets after a successful connection.
    # This also recovers from the stale-clock TLS failure at boot: once NTP
    # corrects the clock, the next connection attempt's handshake succeeds.
    reconnect_min_backoff_seconds: float = 2.0
    reconnect_max_backoff_seconds: float = 30.0

    # Client-side speech gate (Silero VAD). Keeps automatic server VAD on
    # as a backstop; we simply don't forward non-speech audio.
    vad: VADConfig = field(default_factory=VADConfig)


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
