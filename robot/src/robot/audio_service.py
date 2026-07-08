"""
AudioService — Microphone capture and speaker playback.

Captures mic audio in small chunks and publishes them as base64-encoded
PCM16 events. Subscribes to playback events to output audio through the
speaker.

Audio I/O runs in a dedicated worker thread (same pattern as GUIService).
The async run() loop bridges the thread and the event bus.

Platform behaviour:
- Pi: PipeWire routes to the system default speaker. Note: the
  PipeWire echo-cancel module must be disabled — it breaks PortAudio
  callbacks entirely.
- Laptop: system default mic and speaker.
"""

import asyncio
import base64
import threading
import time
from queue import Empty, Queue
from typing import Any

import numpy as np

from robot.base_service import BaseService
from robot.config import AudioConfig
from robot.event_bus import EventBus
from robot.hardware import detect_audio_device


class AudioService(BaseService):
    """
    Audio service for microphone capture and speaker playback.

    Publishes:
        audio.mic_chunk — Base64-encoded PCM16 from mic,
                          payload: {audio: str, sample_rate: int}
        audio.mic_level — Mic input RMS level at ~10 Hz,
                          payload: {level: float}  (0.0–1.0)

    Subscribes:
        audio.play_chunk    — Base64-encoded PCM16 to play,
                              payload: {audio: str, sample_rate: int}
        audio.start_recording — Resume mic streaming, payload: {}
        audio.stop_recording  — Pause mic streaming, payload: {}
        audio.stop_playback   — Clear playback queue, payload: {}
    """

    def __init__(
        self, event_bus: EventBus, config: AudioConfig | None = None
    ) -> None:
        super().__init__(event_bus)
        self.config = config or AudioConfig()

        # The input device name selected during initialize().
        # None means "use system default".
        self._input_device_name: str | None = None

        # Recording gate — toggled by start/stop events. Single bool,
        # GIL makes reads/writes atomic. Belt-and-suspenders on top of
        # PipeWire AEC: the RealtimeService (Phase 4) will pause mic
        # capture while the model speaks.
        self.recording_enabled = False

        # Thread-safe queues bridging the audio worker and the async loop.
        # mic_queue: raw PCM16 bytes from the input callback.
        # speaker_queue: raw PCM16 bytes to feed to the output callback.
        self._mic_queue: Queue[bytes] = Queue(maxsize=100)
        self._speaker_queue: Queue[bytes] = Queue(maxsize=100)

        # Playback buffer — accumulates speaker chunks between output
        # callbacks. Written by the output callback, cleared by
        # stop_playback. The reset (= b"") races with the callback but
        # is safe under GIL (atomic reference swap).
        self._playback_buffer = b""

        # Mic input level (RMS, normalised to 0.0–1.0). Written by the
        # input callback thread, read by the async loop for publishing.
        # Single float — GIL makes reads/writes atomic.
        self.mic_level: float = 0.0

        # Threading control
        self._audio_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    async def initialize(self) -> None:
        """Detect audio input device. Disables service if none found."""
        # Try each preferred device in order
        for device_name in self.config.preferred_input_devices:
            if detect_audio_device(device_name):
                self._input_device_name = device_name
                self.logger.info(f"Using input device: {device_name}")
                self.running = True
                return

        # No preferred device — try system default
        try:
            import sounddevice as sd

            default_info = sd.query_devices(kind="input")
            if default_info:
                name = (
                    default_info.get("name", "Unknown")
                    if isinstance(default_info, dict)
                    else str(default_info)
                )
                self.logger.info(f"Using default input device: {name}")
                self._input_device_name = None  # system default
                self.running = True
                return
        except Exception as err:
            self.logger.warning(f"Failed to query default device: {err}")

        self.logger.warning("No audio input device found. Service disabled.")
        self.running = False

    async def run(self) -> None:
        """Start the audio worker thread and bridge mic data to events."""
        if not self.running:
            return

        # Subscribe to events
        self.event_bus.subscribe("audio.play_chunk", self._handle_play_chunk)
        self.event_bus.subscribe(
            "audio.start_recording", self._handle_start_recording
        )
        self.event_bus.subscribe(
            "audio.stop_recording", self._handle_stop_recording
        )
        self.event_bus.subscribe(
            "audio.stop_playback", self._handle_stop_playback
        )
        self.event_bus.subscribe("audio.wait_drain", self._handle_wait_drain)

        # Start the worker thread
        self._audio_thread = threading.Thread(
            target=self._audio_worker,
            daemon=True,
            name="AudioWorker",
        )
        self._audio_thread.start()

        # Drain the mic queue and publish chunks as events.
        # This runs in the async loop so we can use the event bus directly.
        last_level_time = 0.0

        while self.running and not self._stop_event.is_set():
            # Publish mic level at ~10 Hz for the GUI indicator
            now = time.monotonic()
            if now - last_level_time >= 0.1:
                await self.event_bus.publish(
                    "audio.mic_level",
                    {"level": self.mic_level},
                    async_dispatch=True,
                )
                last_level_time = now

            try:
                mic_bytes = self._mic_queue.get_nowait()
            except Empty:
                await asyncio.sleep(0.005)
                continue

            audio_b64 = base64.b64encode(mic_bytes).decode("ascii")
            await self.event_bus.publish(
                "audio.mic_chunk",
                {
                    "audio": audio_b64,
                    "sample_rate": self.config.input_sample_rate,
                },
                async_dispatch=True,
            )

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _audio_worker(self) -> None:
        """
        Audio I/O loop — runs in a dedicated thread.

        Opens separate InputStream (16 kHz) and OutputStream (24 kHz) with
        sounddevice callbacks. The callbacks push/pull raw PCM16 bytes
        through the queues.
        """
        import sounddevice as sd

        def input_callback(
            indata: np.ndarray,
            _frames: int,
            _time: Any,
            status: Any,
        ) -> None:
            """Capture mic audio and push raw PCM16 bytes to the queue."""
            if status:
                self.logger.debug(f"Input status: {status}")

            # Take channel 0 only (mono). indata shape is (frames, channels).
            mono = indata[:, 0]

            # Always compute RMS for the mic level indicator, even when
            # not recording. This lets the GUI show whether the mic
            # hardware is picking up sound at all.
            rms = np.sqrt(np.mean(mono.astype(np.float32) ** 2))
            self.mic_level = min(rms / 8000.0, 1.0)

            if not self.recording_enabled:
                return

            pcm_bytes = mono.tobytes()

            if not self._mic_queue.full():
                self._mic_queue.put_nowait(pcm_bytes)
            else:
                self.logger.debug("Mic queue full, dropping chunk")

        def output_callback(
            outdata: np.ndarray,
            frames: int,
            _time: Any,
            status: Any,
        ) -> None:
            """Fill outdata from the playback buffer, silence if empty."""
            if status:
                self.logger.debug(f"Output status: {status}")

            # Drain speaker queue into the playback buffer
            while not self._speaker_queue.empty():
                try:
                    chunk = self._speaker_queue.get_nowait()
                    self._playback_buffer += chunk
                except Empty:
                    break

            # Each frame is one int16 sample = 2 bytes
            needed = frames * 2

            if len(self._playback_buffer) >= needed:
                outdata[:, 0] = np.frombuffer(
                    self._playback_buffer[:needed], dtype=np.int16
                )
                self._playback_buffer = self._playback_buffer[needed:]
            else:
                # Not enough data — output silence
                outdata.fill(0)
                # If the queue is also empty, this is a trailing partial
                # frame at the end of playback. Discard it so that
                # wait_drain can detect that playback has finished.
                if self._speaker_queue.empty():
                    self._playback_buffer = b""

        try:
            input_stream = sd.InputStream(
                samplerate=self.config.input_sample_rate,
                channels=self.config.channels,
                dtype="int16",
                blocksize=self.config.input_chunk_size,
                device=self._input_device_name,
                callback=input_callback,
            )
            output_stream = sd.OutputStream(
                samplerate=self.config.output_sample_rate,
                channels=self.config.channels,
                dtype="int16",
                blocksize=self.config.output_chunk_size,
                device=None,  # always system default output
                callback=output_callback,
            )

            with input_stream, output_stream:
                self.logger.info("Audio streams started")

                # Block until stop is requested
                while not self._stop_event.is_set():
                    self._stop_event.wait(timeout=0.1)

        except Exception as err:
            self.logger.error(f"Audio thread error: {err}")
        finally:
            self.logger.info("Audio thread stopped")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_play_chunk(self, data: Any) -> None:
        """Handle audio.play_chunk — decode, apply gain, queue for output."""
        if not isinstance(data, dict):
            return

        audio_b64 = data.get("audio")
        if not audio_b64:
            return

        try:
            pcm_bytes = base64.b64decode(audio_b64)
        except Exception:
            self.logger.warning("play_chunk: invalid base64")
            return

        # Apply playback gain before queueing
        pcm_bytes = self._apply_gain(pcm_bytes, self.config.playback_gain)

        if not self._speaker_queue.full():
            self._speaker_queue.put_nowait(pcm_bytes)

    async def _handle_start_recording(self, _data: Any) -> None:
        """Resume microphone streaming."""
        self.recording_enabled = True
        self.logger.info("Recording enabled")

    async def _handle_stop_recording(self, _data: Any) -> None:
        """Pause microphone streaming."""
        self.recording_enabled = False
        self.logger.info("Recording disabled")

    async def _handle_wait_drain(self, _data: Any) -> None:
        """Wait until all queued audio has been played through the speaker.

        This blocks the caller (via the event bus) until the speaker queue
        and playback buffer are both empty. Used by RealtimeService to wait
        for audio to finish before resuming the mic.
        """
        while (
            not self._speaker_queue.empty() or len(self._playback_buffer) > 0
        ):
            await asyncio.sleep(0.05)

    async def _handle_stop_playback(self, _data: Any) -> None:
        """Clear playback queue and buffer (for interruptions)."""
        # Drain the queue
        while not self._speaker_queue.empty():
            try:
                self._speaker_queue.get_nowait()
            except Empty:
                break

        # Clear the buffer (atomic reference swap under GIL)
        self._playback_buffer = b""
        self.logger.info("Playback stopped")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_gain(pcm_bytes: bytes, gain: float) -> bytes:
        """Scale PCM16 samples by gain, clipping to int16 range."""
        if gain == 1.0:
            return pcm_bytes

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        samples *= gain
        np.clip(samples, -32768, 32767, out=samples)
        return samples.astype(np.int16).tobytes()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Stop the audio thread and clean up."""
        self._stop_event.set()

        # Unsubscribe from events
        self.event_bus.unsubscribe("audio.play_chunk", self._handle_play_chunk)
        self.event_bus.unsubscribe(
            "audio.start_recording", self._handle_start_recording
        )
        self.event_bus.unsubscribe(
            "audio.stop_recording", self._handle_stop_recording
        )
        self.event_bus.unsubscribe(
            "audio.stop_playback", self._handle_stop_playback
        )
        self.event_bus.unsubscribe("audio.wait_drain", self._handle_wait_drain)

        # Wait for the audio thread to finish
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=2.0)

        await super().shutdown()
