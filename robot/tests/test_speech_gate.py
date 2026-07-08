"""Tests for the Silero VAD speech gate (state machine only).

These stub out the ONNX inference (_infer) so they run without onnxruntime
or the model file — they verify the gating/buffering/hangover logic, which
is where the risk lives.
"""

import logging

import numpy as np

from robot.config import VADConfig
from robot.speech_gate import WINDOW_SAMPLES, SpeechGate

_LOG = logging.getLogger("test")


def _window(value: int) -> np.ndarray:
    """A 512-sample int16 window filled with a constant."""
    return np.full(WINDOW_SAMPLES, value, dtype=np.int16)


def _make_gate(probabilities, **config_kwargs) -> SpeechGate:
    """Build an enabled gate whose inference returns scripted probabilities."""
    gate = SpeechGate(VADConfig(**config_kwargs), _LOG)
    gate.enabled = True
    gate.reset()
    scripted = iter(probabilities)
    gate._infer = lambda _window: next(scripted)  # type: ignore[method-assign]
    return gate


class TestVADConfigDefaults:
    def test_defaults(self):
        cfg = VADConfig()
        assert cfg.enabled is True
        assert cfg.model_filename == "silero_vad.onnx"
        assert cfg.threshold == 0.3
        assert cfg.min_silence_ms == 700
        assert cfg.speech_pad_ms == 300


class TestSpeechGate:
    def test_disabled_gate_passes_audio_through(self):
        """With the gate disabled, input is returned unchanged."""
        gate = SpeechGate(VADConfig(enabled=False), _LOG)
        # load() not called, so enabled stays False.
        payload = b"\x01\x02" * 100
        assert gate.process(payload) == payload

    def test_non_speech_is_dropped(self):
        """Windows below threshold produce no output and don't trigger."""
        gate = _make_gate(
            [0.0, 0.1], threshold=0.5, speech_pad_ms=0, min_silence_ms=0
        )
        pcm = np.concatenate([_window(0), _window(0)]).tobytes()
        assert gate.process(pcm) == b""
        assert gate.is_speaking is False

    def test_speech_is_forwarded_and_triggers(self):
        """A window above threshold is forwarded and opens the gate."""
        gate = _make_gate(
            [0.9], threshold=0.5, speech_pad_ms=0, min_silence_ms=1000
        )
        pcm = _window(123).tobytes()
        out = gate.process(pcm)
        assert gate.is_speaking is True
        assert out == pcm  # no pre-roll (speech_pad_ms=0)

    def test_preroll_emitted_on_onset(self):
        """Idle audio just before speech is prepended as pre-roll."""
        # speech_pad_ms=32 → 512 samples of pre-roll (one window).
        gate = _make_gate(
            [0.0, 0.9], threshold=0.5, speech_pad_ms=32, min_silence_ms=1000
        )
        pcm = np.concatenate([_window(50), _window(200)]).tobytes()
        out = gate.process(pcm)
        assert gate.is_speaking is True
        # pre-roll window + speech window = 2 windows forwarded.
        assert len(out) == WINDOW_SAMPLES * 2 * 2  # samples * 2 windows * 2 B

    def test_hangover_then_close(self):
        """Gate stays open through the hangover, then closes after silence."""
        # min_silence_ms=32 → 512 samples → one sub-threshold window closes it.
        gate = _make_gate(
            [0.9, 0.0, 0.0],
            threshold=0.5,
            speech_pad_ms=0,
            min_silence_ms=32,
        )
        gate.process(_window(10).tobytes())  # speech → open
        assert gate.is_speaking is True

        out2 = gate.process(_window(10).tobytes())  # hangover → closes
        assert gate.is_speaking is False
        assert len(out2) == WINDOW_SAMPLES * 2  # hangover audio still sent

        out3 = gate.process(_window(10).tobytes())  # idle → dropped
        assert out3 == b""

    def test_buffers_partial_windows(self):
        """Chunks smaller than a window accumulate until a full window fits."""
        gate = _make_gate(
            [0.9], threshold=0.5, speech_pad_ms=0, min_silence_ms=1000
        )
        half = np.zeros(WINDOW_SAMPLES // 2, dtype=np.int16).tobytes()
        assert gate.process(half) == b""  # not enough yet
        out = gate.process(half)  # now a full window is available
        assert gate.is_speaking is True
        assert len(out) == WINDOW_SAMPLES * 2
