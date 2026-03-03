"""Tests for RealtimeService."""

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from robot.config import RealtimeConfig
from robot.event_bus import EventBus
from robot.realtime_service import RealtimeService


def _async_collector(target_list, *, tag=None):
    """Return an async callback that appends to a list.

    If tag is set, appends (tag, data) tuples. Otherwise appends data.
    """

    async def _handler(data):
        if tag is not None:
            target_list.append((tag, data))
        else:
            target_list.append(data)

    return _handler


# ── RealtimeConfig ────────────────────────────────────────────────────


class TestRealtimeConfig:
    """Test RealtimeConfig dataclass defaults and env loading."""

    def test_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            cfg = RealtimeConfig()
        assert cfg.api_key == ""
        assert cfg.model == "gemini-2.5-flash-native-audio-preview-12-2025"
        assert cfg.voice_name == "Kore"
        assert cfg.input_sample_rate == 16000
        assert cfg.output_sample_rate == 24000
        assert cfg.enable_proactive_audio is True
        assert cfg.enable_affective_dialog is True
        assert cfg.vad_start_sensitivity == "START_SENSITIVITY_LOW"
        assert cfg.vad_end_sensitivity == "END_SENSITIVITY_LOW"
        assert cfg.vad_silence_duration_ms == 500

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key-123"}):
            cfg = RealtimeConfig()
        assert cfg.api_key == "test-key-123"

    def test_custom_values(self):
        cfg = RealtimeConfig(
            api_key="custom-key",
            model="gemini-2.0-flash-live-001",
            voice_name="Puck",
        )
        assert cfg.api_key == "custom-key"
        assert cfg.model == "gemini-2.0-flash-live-001"
        assert cfg.voice_name == "Puck"


# ── RealtimeService init ─────────────────────────────────────────────


class TestRealtimeServiceInit:
    """Test RealtimeService instantiation."""

    def test_can_instantiate(self):
        bus = EventBus()
        service = RealtimeService(bus)
        assert service.running is False
        assert service._session is None
        assert service._response_in_progress is False
        assert service._mic_paused is False

    def test_custom_config(self):
        bus = EventBus()
        config = RealtimeConfig(api_key="my-key", voice_name="Puck")
        service = RealtimeService(bus, config)
        assert service.config.api_key == "my-key"
        assert service.config.voice_name == "Puck"


# ── RealtimeService.initialize() ─────────────────────────────────────


class TestRealtimeServiceInitialize:
    """Test API key check during initialize."""

    async def test_disables_without_api_key(self):
        """Service disables itself when no API key is set."""
        bus = EventBus()
        config = RealtimeConfig(api_key="")
        service = RealtimeService(bus, config)

        await service.initialize()
        assert service.running is False

    async def test_enables_with_api_key(self):
        """Service enables when API key is present."""
        bus = EventBus()
        config = RealtimeConfig(api_key="test-key")
        service = RealtimeService(bus, config)

        await service.initialize()
        assert service.running is True


# ── Mic chunk handling ────────────────────────────────────────────────


class TestHandleMicChunk:
    """Test audio.mic_chunk event handler."""

    async def test_ignores_when_no_session(self):
        """Does nothing when session is not connected."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service.running = True
        # _session is None — should return silently
        audio_b64 = base64.b64encode(b"\x00" * 100).decode("ascii")
        await service._handle_mic_chunk(
            {"audio": audio_b64, "sample_rate": 16000}
        )

    async def test_ignores_when_not_running(self):
        """Does nothing when service is not running."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service._session = AsyncMock()
        service.running = False

        audio_b64 = base64.b64encode(b"\x00" * 100).decode("ascii")
        await service._handle_mic_chunk(
            {"audio": audio_b64, "sample_rate": 16000}
        )
        service._session.send_realtime_input.assert_not_called()

    async def test_sends_audio_to_session(self):
        """Forwards decoded audio bytes to the Gemini session."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service.running = True

        mock_session = AsyncMock()
        service._session = mock_session

        raw_audio = b"\x01\x02" * 50
        audio_b64 = base64.b64encode(raw_audio).decode("ascii")

        await service._handle_mic_chunk(
            {"audio": audio_b64, "sample_rate": 16000}
        )

        mock_session.send_realtime_input.assert_called_once()
        call_kwargs = mock_session.send_realtime_input.call_args
        sent_audio = call_kwargs.kwargs["audio"]
        assert sent_audio["data"] == audio_b64
        assert sent_audio["mime_type"] == "audio/pcm;rate=16000"

    async def test_handles_send_error_gracefully(self):
        """Logs error but does not crash on send failure."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service.running = True

        mock_session = AsyncMock()
        mock_session.send_realtime_input.side_effect = RuntimeError(
            "connection lost"
        )
        service._session = mock_session

        audio_b64 = base64.b64encode(b"\x00" * 100).decode("ascii")
        # Should not raise
        await service._handle_mic_chunk(
            {"audio": audio_b64, "sample_rate": 16000}
        )

    async def test_ignores_non_dict(self):
        """Ignores non-dict payloads."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service.running = True
        service._session = AsyncMock()

        await service._handle_mic_chunk("not a dict")
        service._session.send_realtime_input.assert_not_called()

    async def test_ignores_missing_audio(self):
        """Ignores payloads without 'audio' key."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service.running = True
        service._session = AsyncMock()

        await service._handle_mic_chunk({"sample_rate": 16000})
        service._session.send_realtime_input.assert_not_called()


# ── Server response handling ──────────────────────────────────────────


def _make_server_content(
    *,
    model_turn=None,
    interrupted=False,
    turn_complete=False,
    input_transcription=None,
    output_transcription=None,
):
    """Create a mock server_content object."""
    return SimpleNamespace(
        model_turn=model_turn,
        interrupted=interrupted,
        turn_complete=turn_complete,
        input_transcription=input_transcription,
        output_transcription=output_transcription,
    )


def _make_audio_part(audio_bytes):
    """Create a mock part with inline audio data."""
    return SimpleNamespace(
        inline_data=SimpleNamespace(data=audio_bytes),
        text=None,
    )


class TestHandleServerResponse:
    """Test _handle_server_response with various message types."""

    async def test_model_turn_publishes_audio(self):
        """Audio data in model_turn is published as audio.play_chunk."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))

        published = []
        bus.subscribe(
            "audio.play_chunk",
            _async_collector(published, tag="audio.play_chunk"),
        )

        raw_audio = b"\x00\x01" * 100
        part = _make_audio_part(raw_audio)
        model_turn = SimpleNamespace(parts=[part])
        content = _make_server_content(model_turn=model_turn)

        await service._handle_server_response(content)

        assert len(published) == 1
        event_name, payload = published[0]
        assert event_name == "audio.play_chunk"
        assert payload["sample_rate"] == 24000
        # Verify the audio round-trips correctly
        decoded = base64.b64decode(payload["audio"])
        assert decoded == raw_audio

    async def test_first_model_turn_pauses_mic(self):
        """First model_turn chunk pauses mic and sets happy expression."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        assert service._mic_paused is False

        events = []
        for name in [
            "realtime.response_started",
            "gui.set_expression",
            "audio.stop_recording",
            "audio.play_chunk",
        ]:
            bus.subscribe(name, _async_collector(events, tag=name))

        part = _make_audio_part(b"\x00" * 10)
        model_turn = SimpleNamespace(parts=[part])
        content = _make_server_content(model_turn=model_turn)

        await service._handle_server_response(content)

        assert service._response_in_progress is True
        assert service._mic_paused is True
        event_names = [e[0] for e in events]
        assert "realtime.response_started" in event_names
        assert "gui.set_expression" in event_names
        assert "audio.stop_recording" in event_names

    async def test_subsequent_model_turn_skips_pause(self):
        """Second model_turn chunk does not re-pause mic."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        # Simulate already in a response
        service._response_in_progress = True
        service._mic_paused = True

        events = []
        bus.subscribe(
            "realtime.response_started",
            _async_collector(events, tag="response_started"),
        )

        part = _make_audio_part(b"\x00" * 10)
        model_turn = SimpleNamespace(parts=[part])
        content = _make_server_content(model_turn=model_turn)

        await service._handle_server_response(content)

        # Should NOT publish response_started again
        assert len(events) == 0

    async def test_turn_complete_resumes_mic(self):
        """Turn complete resumes mic and sets neutral expression."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service._response_in_progress = True
        service._mic_paused = True

        events = []
        for name in [
            "gui.set_expression",
            "audio.start_recording",
            "realtime.response_completed",
        ]:
            bus.subscribe(name, _async_collector(events, tag=name))

        content = _make_server_content(turn_complete=True)
        await service._handle_server_response(content)

        assert service._response_in_progress is False
        assert service._mic_paused is False
        event_names = [e[0] for e in events]
        assert "audio.start_recording" in event_names
        assert "realtime.response_completed" in event_names
        # Check expression was set to neutral
        expr_events = [e for e in events if e[0] == "gui.set_expression"]
        assert expr_events[0][1] == {"expression": "neutral"}

    async def test_interrupted_stops_playback(self):
        """Interruption clears speaker queue and resumes mic."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service._response_in_progress = True
        service._mic_paused = True

        events = []
        for name in [
            "audio.stop_playback",
            "gui.set_expression",
            "audio.start_recording",
            "realtime.interrupted",
        ]:
            bus.subscribe(name, _async_collector(events, tag=name))

        content = _make_server_content(interrupted=True)
        await service._handle_server_response(content)

        assert service._response_in_progress is False
        assert service._mic_paused is False
        event_names = [e[0] for e in events]
        assert "audio.stop_playback" in event_names
        assert "audio.start_recording" in event_names
        assert "realtime.interrupted" in event_names

    async def test_user_transcript(self):
        """Input transcription is published as realtime.user_transcript."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))

        published = []
        bus.subscribe(
            "realtime.user_transcript",
            _async_collector(published),
        )

        content = _make_server_content(
            input_transcription=SimpleNamespace(text="Hej Roberta")
        )
        await service._handle_server_response(content)

        assert len(published) == 1
        assert published[0] == {"text": "Hej Roberta"}

    async def test_model_transcript_accumulated(self):
        """Output transcripts are accumulated and published at turn_complete."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))

        published = []
        bus.subscribe(
            "realtime.model_transcript",
            _async_collector(published),
        )

        # Simulate word-by-word transcript fragments
        for word in ["Hej! ", "Hvordan ", "kan jeg ", "hjælpe?"]:
            content = _make_server_content(
                output_transcription=SimpleNamespace(text=word)
            )
            await service._handle_server_response(content)

        # Nothing published yet — still accumulating
        assert len(published) == 0

        # Turn complete triggers the full transcript
        content = _make_server_content(turn_complete=True)
        await service._handle_server_response(content)

        assert len(published) == 1
        assert published[0] == {"text": "Hej! Hvordan kan jeg hjælpe?"}


# ── Mic pause/resume ─────────────────────────────────────────────────


class TestMicPauseResume:
    """Test idempotent mic pause/resume helpers."""

    async def test_pause_publishes_stop_recording(self):
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))

        events = []
        bus.subscribe(
            "audio.stop_recording",
            _async_collector(events, tag="stop"),
        )

        await service._pause_mic()
        assert service._mic_paused is True
        assert events == [("stop", {})]

    async def test_pause_is_idempotent(self):
        """Second pause does not publish another event."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))

        events = []
        bus.subscribe(
            "audio.stop_recording",
            _async_collector(events, tag="stop"),
        )

        await service._pause_mic()
        await service._pause_mic()
        assert len(events) == 1  # only one event

    async def test_resume_publishes_start_recording(self):
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service._mic_paused = True  # simulate paused state

        events = []
        bus.subscribe(
            "audio.start_recording",
            _async_collector(events, tag="start"),
        )

        await service._resume_mic()
        assert service._mic_paused is False
        assert events == [("start", {})]

    async def test_resume_is_idempotent(self):
        """Resume when not paused does nothing."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))

        events = []
        bus.subscribe(
            "audio.start_recording",
            _async_collector(events, tag="start"),
        )

        await service._resume_mic()
        assert events == []  # no event — was not paused


# ── Shutdown ──────────────────────────────────────────────────────────


class TestShutdown:
    """Test RealtimeService shutdown."""

    async def test_closes_session(self):
        """Shutdown closes the Gemini session."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service.running = True

        mock_session = AsyncMock()
        service._session = mock_session

        await service.shutdown()

        mock_session.close.assert_called_once()
        assert service._session is None
        assert service.running is False

    async def test_unsubscribes_from_events(self):
        """Shutdown removes the mic_chunk subscription."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service.running = True

        # Manually subscribe (normally done in run())
        bus.subscribe("audio.mic_chunk", service._handle_mic_chunk)
        assert service._handle_mic_chunk in bus.subscribers.get(
            "audio.mic_chunk", []
        )

        await service.shutdown()

        assert service._handle_mic_chunk not in bus.subscribers.get(
            "audio.mic_chunk", []
        )

    async def test_resumes_mic_if_paused(self):
        """Shutdown resumes mic capture if it was paused."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service.running = True
        service._mic_paused = True

        events = []
        bus.subscribe(
            "audio.start_recording",
            _async_collector(events, tag="start"),
        )

        await service.shutdown()

        assert service._mic_paused is False
        assert len(events) == 1

    async def test_publishes_disconnected(self):
        """Shutdown publishes realtime.disconnected."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service.running = True

        events = []
        bus.subscribe(
            "realtime.disconnected",
            _async_collector(events),
        )

        await service.shutdown()

        assert len(events) == 1
        assert events[0] == {"provider": "gemini"}

    async def test_handles_session_close_error(self):
        """Shutdown handles errors when closing session gracefully."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service.running = True

        mock_session = AsyncMock()
        mock_session.close.side_effect = RuntimeError("already closed")
        service._session = mock_session

        # Should not raise
        await service.shutdown()
        assert service._session is None

    async def test_shutdown_without_session(self):
        """Shutdown works fine when no session was ever created."""
        bus = EventBus()
        service = RealtimeService(bus, RealtimeConfig(api_key="key"))
        service.running = True

        # Should not raise
        await service.shutdown()
        assert service.running is False
