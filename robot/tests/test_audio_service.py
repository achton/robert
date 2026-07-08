"""Tests for AudioService."""

import base64
from unittest.mock import patch

import numpy as np

from robot.audio_service import AudioService
from robot.config import AudioConfig
from robot.event_bus import EventBus

# ── AudioConfig ──────────────────────────────────────────────────────


class TestAudioConfig:
    """Test AudioConfig dataclass defaults and properties."""

    def test_defaults(self):
        cfg = AudioConfig()
        assert cfg.input_sample_rate == 16000
        assert cfg.output_sample_rate == 24000
        assert cfg.channels == 1
        assert cfg.chunk_duration_ms == 40
        assert cfg.playback_gain == 0.8

    def test_input_chunk_size(self):
        """16 kHz * 40 ms = 640 samples."""
        cfg = AudioConfig()
        assert cfg.input_chunk_size == 640

    def test_output_chunk_size(self):
        """24 kHz * 40 ms = 960 samples."""
        cfg = AudioConfig()
        assert cfg.output_chunk_size == 960

    def test_custom_chunk_duration(self):
        cfg = AudioConfig(chunk_duration_ms=20)
        assert cfg.input_chunk_size == 320
        assert cfg.output_chunk_size == 480

    def test_preferred_input_devices(self):
        cfg = AudioConfig()
        assert "echo_cancel_source" in cfg.preferred_input_devices


# ── AudioService init ────────────────────────────────────────────────


class TestAudioServiceInit:
    """Test AudioService instantiation."""

    def test_can_instantiate(self):
        bus = EventBus()
        service = AudioService(bus)
        assert service.running is False
        assert service.recording_enabled is False

    def test_custom_config(self):
        bus = EventBus()
        config = AudioConfig(input_sample_rate=48000, playback_gain=0.5)
        service = AudioService(bus, config)
        assert service.config.input_sample_rate == 48000
        assert service.config.playback_gain == 0.5


# ── AudioService.initialize() ───────────────────────────────────────


class TestAudioServiceInitialize:
    """Test hardware detection during initialize()."""

    async def test_disables_when_no_device(self):
        """Service sets running=False when no audio input is found."""
        bus = EventBus()
        service = AudioService(bus)

        with (
            patch(
                "robot.audio_service.detect_audio_device",
                return_value=False,
            ),
            patch(
                "sounddevice.query_devices",
                side_effect=Exception("no device"),
            ),
        ):
            await service.initialize()

        assert service.running is False

    async def test_enables_with_preferred_device(self):
        """Service enables when echo_cancel_source is found."""
        bus = EventBus()
        service = AudioService(bus)

        def fake_detect(name):
            return name == "echo_cancel_source"

        with patch(
            "robot.audio_service.detect_audio_device",
            side_effect=fake_detect,
        ):
            await service.initialize()

        assert service.running is True
        assert service._input_device_name == "echo_cancel_source"

    async def test_falls_back_to_system_default(self):
        """Falls back to system default when no preferred device found."""
        bus = EventBus()
        service = AudioService(bus)

        with (
            patch(
                "robot.audio_service.detect_audio_device",
                return_value=False,
            ),
            patch(
                "sounddevice.query_devices",
                return_value={"name": "Built-in Mic", "max_input_channels": 2},
            ),
        ):
            await service.initialize()

        assert service.running is True
        # None means system default
        assert service._input_device_name is None


# ── Recording gate ───────────────────────────────────────────────────


class TestRecordingGate:
    """Test start/stop recording event handlers."""

    async def test_start_recording(self):
        bus = EventBus()
        service = AudioService(bus)
        assert service.recording_enabled is False

        await service._handle_start_recording({})
        assert service.recording_enabled is True

    async def test_stop_recording(self):
        bus = EventBus()
        service = AudioService(bus)
        service.recording_enabled = True

        await service._handle_stop_recording({})
        assert service.recording_enabled is False


# ── Play chunk handling ──────────────────────────────────────────────


class TestHandlePlayChunk:
    """Test audio.play_chunk event handler."""

    def _make_pcm16_b64(self, num_samples: int = 100) -> str:
        """Create a base64-encoded PCM16 test signal."""
        samples = np.zeros(num_samples, dtype=np.int16)
        return base64.b64encode(samples.tobytes()).decode("ascii")

    async def test_queues_valid_audio(self):
        bus = EventBus()
        service = AudioService(bus)

        audio_b64 = self._make_pcm16_b64(100)
        await service._handle_play_chunk({"audio": audio_b64})

        assert not service._speaker_queue.empty()

    async def test_ignores_missing_audio(self):
        bus = EventBus()
        service = AudioService(bus)

        await service._handle_play_chunk({})
        assert service._speaker_queue.empty()

    async def test_ignores_non_dict(self):
        bus = EventBus()
        service = AudioService(bus)

        await service._handle_play_chunk("not a dict")
        assert service._speaker_queue.empty()

    async def test_ignores_bad_base64(self):
        bus = EventBus()
        service = AudioService(bus)

        await service._handle_play_chunk({"audio": "!!!invalid!!!"})
        assert service._speaker_queue.empty()

    async def test_applies_gain(self):
        """Playback gain should reduce sample amplitude."""
        bus = EventBus()
        config = AudioConfig(playback_gain=0.5)
        service = AudioService(bus, config)

        # Create a signal with known amplitude
        samples = np.full(100, 10000, dtype=np.int16)
        audio_b64 = base64.b64encode(samples.tobytes()).decode("ascii")

        await service._handle_play_chunk({"audio": audio_b64})

        # Read back from queue
        queued_bytes = service._speaker_queue.get_nowait()
        result = np.frombuffer(queued_bytes, dtype=np.int16)
        assert np.all(result == 5000)


# ── Apply gain ───────────────────────────────────────────────────────


class TestApplyGain:
    """Test the _apply_gain static method."""

    def test_unity_gain_passthrough(self):
        """Gain of 1.0 should return input unchanged."""
        samples = np.array([100, -200, 32767], dtype=np.int16)
        pcm = samples.tobytes()
        result = AudioService._apply_gain(pcm, 1.0)
        assert result == pcm

    def test_half_gain(self):
        samples = np.array([10000, -10000], dtype=np.int16)
        pcm = samples.tobytes()
        result = AudioService._apply_gain(pcm, 0.5)
        out = np.frombuffer(result, dtype=np.int16)
        assert np.all(out == np.array([5000, -5000], dtype=np.int16))

    def test_clipping(self):
        """Gain > 1 should clip to int16 range."""
        samples = np.array([30000, -30000], dtype=np.int16)
        pcm = samples.tobytes()
        result = AudioService._apply_gain(pcm, 2.0)
        out = np.frombuffer(result, dtype=np.int16)
        assert out[0] == 32767
        assert out[1] == -32768


# ── Stop playback ────────────────────────────────────────────────────


class TestStopPlayback:
    """Test the stop playback handler."""

    async def test_clears_queue_and_buffer(self):
        bus = EventBus()
        service = AudioService(bus)

        # Fill queue and buffer
        service._speaker_queue.put_nowait(b"\x00" * 100)
        service._speaker_queue.put_nowait(b"\x00" * 100)
        service._playback_buffer = b"\x00" * 500

        await service._handle_stop_playback({})

        assert service._speaker_queue.empty()
        assert service._playback_buffer == b""


# ── Event bus round-trip ─────────────────────────────────────────────


class TestPlayChunkViaEventBus:
    """Test audio.play_chunk through the full event bus flow."""

    async def test_event_queues_audio(self):
        bus = EventBus()
        service = AudioService(bus)

        # Subscribe the handler (normally done in run())
        bus.subscribe("audio.play_chunk", service._handle_play_chunk)

        # Create test audio
        samples = np.zeros(100, dtype=np.int16)
        audio_b64 = base64.b64encode(samples.tobytes()).decode("ascii")

        await bus.publish("audio.play_chunk", {"audio": audio_b64})

        assert not service._speaker_queue.empty()

    async def test_recording_gate_via_bus(self):
        """Start/stop recording via bus toggles the flag."""
        bus = EventBus()
        service = AudioService(bus)

        bus.subscribe("audio.start_recording", service._handle_start_recording)
        bus.subscribe("audio.stop_recording", service._handle_stop_recording)

        await bus.publish("audio.start_recording", {})
        assert service.recording_enabled is True

        await bus.publish("audio.stop_recording", {})
        assert service.recording_enabled is False
