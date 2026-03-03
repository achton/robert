"""
RealtimeService — Live voice conversation via Gemini.

Connects to Google Gemini Live for real-time speech-to-speech conversation.
Streams mic audio from the event bus to Gemini, and plays back Gemini's
audio responses through the speaker.

Echo cancellation strategy (two layers):
1. PipeWire AEC (hardware) — already configured on the Pi. The
   echo_cancel_source virtual device subtracts speaker output from mic.
2. Mic pause during model speech (software) — prevents residual echo
   from triggering Gemini's VAD. Trade-off: no vocal barge-in while
   the model speaks. True barge-in can be tested later by toggling
   this off and relying on AEC alone.
"""

import base64
from typing import Any

import websockets.exceptions

from robot.base_service import BaseService
from robot.config import RealtimeConfig
from robot.event_bus import EventBus


class RealtimeService(BaseService):
    """
    Real-time voice conversation service using Gemini Live.

    Publishes:
        realtime.connected        — Session established
        realtime.response_started — Model starts speaking
        realtime.response_completed — Model done speaking
        realtime.interrupted      — User interrupted model
        realtime.user_transcript  — User speech transcript
        realtime.model_transcript — Model speech transcript
        realtime.error            — Error occurred
        realtime.disconnected     — Session closed
        gui.set_expression        — Change face expression
        audio.play_chunk          — Audio to play through speaker
        audio.start_recording     — Resume mic capture
        audio.stop_recording      — Pause mic capture
        audio.stop_playback       — Clear speaker queue

    Subscribes:
        audio.mic_chunk — Mic audio from AudioService
    """

    def __init__(
        self, event_bus: EventBus, config: RealtimeConfig | None = None
    ) -> None:
        super().__init__(event_bus)
        self.config = config or RealtimeConfig()

        # Session handle — set when connected, cleared on disconnect.
        self._session: Any = None

        # True while the model is actively generating a response.
        self._response_in_progress = False

        # True when we've paused the mic (to prevent echo feedback).
        self._mic_paused = False

        # Accumulates transcript fragments during a model turn.
        # Logged as a single line when the turn completes.
        self._model_transcript_buffer: list[str] = []

    async def initialize(self) -> None:
        """Check for API key. Disables service if not set."""
        if not self.config.api_key:
            self.logger.warning(
                "No GEMINI_API_KEY set. RealtimeService disabled."
            )
            self.running = False
            return

        self.running = True

    async def run(self) -> None:
        """Connect to Gemini Live and stream audio bidirectionally."""
        if not self.running:
            return

        # Deferred import — avoids ImportError when google-genai is not
        # installed (e.g. in CI or on machines without the dependency).
        from google import genai
        from google.genai import types

        # Subscribe to mic audio from AudioService
        self.event_bus.subscribe("audio.mic_chunk", self._handle_mic_chunk)

        # Build the Gemini client and session config
        client = genai.Client(
            api_key=self.config.api_key,
            http_options={"api_version": "v1alpha"},
        )

        live_config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=self.config.system_instruction,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.config.voice_name,
                    )
                )
            ),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(  # noqa: E501
                    start_of_speech_sensitivity=types.StartSensitivity[
                        self.config.vad_start_sensitivity
                    ],
                    end_of_speech_sensitivity=types.EndSensitivity[
                        self.config.vad_end_sensitivity
                    ],
                    silence_duration_ms=self.config.vad_silence_duration_ms,
                )
            ),
            enable_affective_dialog=self.config.enable_affective_dialog,
            # Enable transcription so we can log what's being said
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

        try:
            async with client.aio.live.connect(
                model=self.config.model,
                config=live_config,
            ) as session:
                self._session = session
                self.logger.info(f"Connected to {self.config.model}")

                await self.event_bus.publish(
                    "realtime.connected",
                    {
                        "provider": "gemini",
                        "model": self.config.model,
                    },
                )

                # Start capturing mic audio
                await self.event_bus.publish("audio.start_recording", {})

                # Send a greeting prompt so the bot speaks first
                await session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part(text=self.config.greeting_prompt)],
                    ),
                    turn_complete=True,
                )

                # Process responses until the session ends
                await self._response_loop()

        except websockets.exceptions.ConnectionClosedOK:
            self.logger.info("Gemini session closed normally")
        except Exception as err:
            self.logger.error(f"Gemini session error: {err}")
            await self.event_bus.publish("realtime.error", {"error": str(err)})
        finally:
            self._session = None
            self.running = False

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

    async def _response_loop(self) -> None:
        """Iterate over server messages until the session closes."""
        async for message in self._session.receive():
            if not self.running:
                break

            server_content = message.server_content
            if server_content is None:
                continue

            await self._handle_server_response(server_content)

    async def _handle_server_response(self, server_content: Any) -> None:
        """Handle a single server content message from Gemini."""

        # Interruption — user spoke while model was responding.
        # Stop playback immediately and resume mic.
        if server_content.interrupted:
            self.logger.info("Interrupted by user")
            self._model_transcript_buffer.clear()
            await self.event_bus.publish("audio.stop_playback", {})
            await self.event_bus.publish(
                "gui.set_expression", {"expression": "neutral"}
            )
            await self._resume_mic()
            self._response_in_progress = False
            await self.event_bus.publish("realtime.interrupted", {})
            return

        # Model audio turn — stream audio to speaker
        if server_content.model_turn:
            # First chunk of a new response: pause mic, change face
            if not self._response_in_progress:
                self._response_in_progress = True
                await self.event_bus.publish("realtime.response_started", {})
                await self.event_bus.publish(
                    "gui.set_expression", {"expression": "happy"}
                )
                await self._pause_mic()

            # Forward audio chunks to the speaker
            for part in server_content.model_turn.parts:
                if part.inline_data and part.inline_data.data:
                    audio_b64 = base64.b64encode(part.inline_data.data).decode(
                        "ascii"
                    )
                    await self.event_bus.publish(
                        "audio.play_chunk",
                        {
                            "audio": audio_b64,
                            "sample_rate": self.config.output_sample_rate,
                        },
                    )

        # Transcripts (if transcription is enabled in the config).
        # The transcription objects have a .text attribute.
        # User transcripts arrive as complete sentences, so log immediately.
        input_tx = getattr(server_content, "input_transcription", None)
        if input_tx and getattr(input_tx, "text", None):
            self.logger.info(f"User: {input_tx.text}")
            await self.event_bus.publish(
                "realtime.user_transcript",
                {"text": input_tx.text},
            )

        # Model transcripts arrive word-by-word — accumulate them and
        # log the full sentence when the turn completes.
        output_tx = getattr(server_content, "output_transcription", None)
        if output_tx and getattr(output_tx, "text", None):
            self._model_transcript_buffer.append(output_tx.text)

        # Turn complete — model finished speaking
        if server_content.turn_complete:
            # Log and publish the accumulated model transcript
            if self._model_transcript_buffer:
                full_text = "".join(self._model_transcript_buffer).strip()
                self._model_transcript_buffer.clear()
                if full_text:
                    self.logger.info(f"Model: {full_text}")
                    await self.event_bus.publish(
                        "realtime.model_transcript",
                        {"text": full_text},
                    )

            await self.event_bus.publish(
                "gui.set_expression", {"expression": "neutral"}
            )
            await self._resume_mic()
            self._response_in_progress = False
            await self.event_bus.publish("realtime.response_completed", {})

    # ------------------------------------------------------------------
    # Mic audio forwarding
    # ------------------------------------------------------------------

    async def _handle_mic_chunk(self, data: Any) -> None:
        """Forward mic audio to the Gemini session."""
        if self._session is None or not self.running:
            return

        if not isinstance(data, dict):
            return

        audio_b64 = data.get("audio")
        if not audio_b64:
            return

        try:
            await self._session.send_realtime_input(
                audio={
                    "data": audio_b64,
                    "mime_type": f"audio/pcm;rate={self.config.input_sample_rate}",
                },
            )
        except Exception as err:
            self.logger.debug(f"Failed to send mic chunk: {err}")

    # ------------------------------------------------------------------
    # Mic pause/resume (echo cancellation layer 2)
    # ------------------------------------------------------------------

    async def _pause_mic(self) -> None:
        """Pause mic capture. Idempotent — safe to call multiple times."""
        if self._mic_paused:
            return
        self._mic_paused = True
        await self.event_bus.publish("audio.stop_recording", {})

    async def _resume_mic(self) -> None:
        """Resume mic capture. Idempotent — safe to call multiple times."""
        if not self._mic_paused:
            return
        self._mic_paused = False
        await self.event_bus.publish("audio.start_recording", {})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Close the Gemini session and clean up."""
        # Close the session — this causes _response_loop to exit
        if self._session is not None:
            try:
                await self._session.close()
            except Exception as err:
                self.logger.debug(f"Error closing session: {err}")
            self._session = None

        # Unsubscribe from events
        self.event_bus.unsubscribe("audio.mic_chunk", self._handle_mic_chunk)

        # Belt-and-suspenders: make sure mic is resumed
        if self._mic_paused:
            self._mic_paused = False
            await self.event_bus.publish("audio.start_recording", {})

        await self.event_bus.publish(
            "realtime.disconnected", {"provider": "gemini"}
        )

        await super().shutdown()
