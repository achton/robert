"""
SpeechGate — client-side voice-activity detection (Silero VAD, ONNX).

RealtimeService uses this to forward only *speech* to Gemini. Non-speech
(HVAC, keyboards, silence) never reaches the model, which otherwise
hallucinates transcripts from noise (see docs/voice-gating-research.md).

The Silero model is fixed at 16 kHz and scores exactly 512 samples per call.
It also needs 64 samples of context from the previous window, so the tensor
handed to ONNX is 576 samples. We buffer the incoming mic stream into
512-sample windows and carry both the recurrent state and the context
forward between calls.

Runtime: onnxruntime (no torch). If onnxruntime or the model file is
missing, the gate disables itself and passes all audio through unchanged,
so the laptop and CI degrade gracefully.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from robot.config import VADConfig

# The shared assets directory — same place GUIService loads images from.
ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"

# Silero VAD is fixed at these sizes for 16 kHz audio.
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512  # samples scored per inference call
CONTEXT_SAMPLES = 64  # previous-window samples prepended before inference
_INT16_FULL_SCALE = 32768.0


class SpeechGate:
    """Streaming speech detector built on the Silero VAD ONNX model.

    Feed it raw PCM16 mic bytes with process(); it returns the subset of
    the audio that is speech (with a short pre-roll so word onsets aren't
    clipped) and exposes is_speaking so the caller can tell when a segment
    ends.
    """

    def __init__(self, config: VADConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

        # Set once the ONNX model has loaded. Until then (and if loading
        # fails) the gate passes all audio through untouched.
        self.enabled = False
        self._session: Any = None

        # Thresholds converted to sample counts.
        self._min_silence_samples = int(
            SAMPLE_RATE * config.min_silence_ms / 1000
        )
        self._speech_pad_samples = int(
            SAMPLE_RATE * config.speech_pad_ms / 1000
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

        # Streaming buffers and model state — initialized by reset().
        self._input_buffer = np.zeros(0, dtype=np.int16)
        self._preroll: deque[int] = deque(maxlen=self._speech_pad_samples)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self._triggered = False
        self._silence_samples = 0

    def load(self) -> None:
        """Load the ONNX model. Leaves the gate disabled on any failure."""
        if not self.config.enabled:
            self.logger.info("SpeechGate: disabled by config")
            return

        model_path = ASSETS_DIR / self.config.model_filename
        if not model_path.exists():
            self.logger.warning(
                f"SpeechGate: model not found at {model_path} — "
                "forwarding all audio"
            )
            return

        try:
            import onnxruntime

            # VAD is tiny; one CPU thread is plenty and avoids contending
            # with the audio and render threads on the Pi.
            options = onnxruntime.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            self._session = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self.reset()
            self.enabled = True
            self.logger.info("SpeechGate: Silero VAD loaded (ONNX)")
        except Exception as err:
            self.logger.warning(
                f"SpeechGate: failed to load, forwarding all audio: {err}"
            )
            self._session = None
            self.enabled = False

    @property
    def is_speaking(self) -> bool:
        """True while inside a detected speech segment (incl. hangover)."""
        return self._triggered

    def reset(self) -> None:
        """Clear streaming state — call when a new audio stream starts."""
        self._input_buffer = np.zeros(0, dtype=np.int16)
        self._preroll = deque(maxlen=self._speech_pad_samples)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self._triggered = False
        self._silence_samples = 0

    def process(self, pcm_bytes: bytes) -> bytes:
        """Return the speech portion of pcm_bytes (PCM16, 16 kHz mono).

        When the gate is disabled the input is returned unchanged. When
        enabled, non-speech is dropped and speech is returned with a short
        pre-roll of the audio that preceded it.
        """
        if not self.enabled:
            return pcm_bytes

        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        self._input_buffer = np.concatenate([self._input_buffer, samples])

        forwarded: list[np.ndarray] = []
        while len(self._input_buffer) >= WINDOW_SAMPLES:
            window = self._input_buffer[:WINDOW_SAMPLES]
            self._input_buffer = self._input_buffer[WINDOW_SAMPLES:]
            self._process_window(window, forwarded)

        if not forwarded:
            return b""
        return np.concatenate(forwarded).astype(np.int16).tobytes()

    def _process_window(
        self, window: np.ndarray, forwarded: list[np.ndarray]
    ) -> None:
        """Score one 512-sample window and update the speech state."""
        probability = self._infer(window)
        is_speech = probability >= self.config.threshold

        if is_speech:
            if not self._triggered:
                # Speech just started — emit the buffered pre-roll first so
                # the word onset isn't clipped.
                self._triggered = True
                if self._preroll:
                    forwarded.append(np.array(self._preroll, dtype=np.int16))
                    self._preroll.clear()
            self._silence_samples = 0
            forwarded.append(window)
            return

        # Sub-threshold window.
        if self._triggered:
            # Keep forwarding through the hangover so trailing words and the
            # sentence-final silence still reach Gemini's turn detector.
            self._silence_samples += WINDOW_SAMPLES
            forwarded.append(window)
            if self._silence_samples >= self._min_silence_samples:
                self._triggered = False
                self._silence_samples = 0
        else:
            # Idle — keep recent audio for the next onset's pre-roll.
            self._preroll.extend(window.tolist())

    def _infer(self, window: np.ndarray) -> float:
        """Run the Silero model on a 512-sample window, return P(speech)."""
        audio = window.astype(np.float32) / _INT16_FULL_SCALE
        # Prepend the previous window's 64-sample context → 576 samples.
        model_input = np.concatenate([self._context, audio])[np.newaxis, :]
        self._context = audio[-CONTEXT_SAMPLES:]

        output, self._state = self._session.run(
            None,
            {
                "input": model_input,
                "state": self._state,
                "sr": self._sr,
            },
        )
        return float(np.ravel(output)[0])
