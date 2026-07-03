# Roberta — Implementation Plan

This document is the roadmap for building Roberta feature by feature. Each
phase produces a working, testable increment. We move to the next phase only
when the current one is solid.

This plan is distilled from the architecture and design work in the previous
prototypes (`.OLD/robotv3/docs/PLAN.md`, `.OLD/robotv3/docs/ARCHITECTURE.md`,
etc.). **When implementing a phase, always consult the corresponding code and
docs in `.OLD/robotv3/` and `.OLD/robotv2/` first.** Lift working code
directly when it fits — don't rewrite something that already works. Adapt
only where the new structure or findings require it.

## Architecture Overview

Event-driven. A central **EventBus** coordinates loosely-coupled **services**.
Each service owns one concern (audio, display, LLM, vision, etc.) and
communicates only through events. Services detect available hardware at startup
and disable themselves gracefully when something is missing.

```text
                        ┌───────────┐
                        │  EventBus │
                        └─────┬─────┘
          ┌─────────┬─────────┼─────────┬──────────┐
          ▼         ▼         ▼         ▼          ▼
     ┌────────┐ ┌───────┐ ┌────────┐ ┌───────┐ ┌──────┐
     │ Audio  │ │  GUI  │ │Realtime│ │Vision │ │ ...  │
     │Service │ │Service│ │Service │ │Service│ │      │
     └────────┘ └───────┘ └────────┘ └───────┘ └──────┘
```

## Phases

### Phase 0 — Project scaffolding and Pi infrastructure

Set up the Python project, tooling, and basic structure. Also establish a
reproducible Pi setup so we can wipe the SD card and get back to a working
Roberta quickly.

The `Taskfile.yml` lives at the project root (it covers the whole project).
The Python venv and `pyproject.toml` live inside `robot/`.

- [x] `robot/pyproject.toml` with uv, black, ruff, mypy, pytest
- [x] `Taskfile.yml` at project root (format, lint, test, run, deploy)
- [x] Minimal `robot/` package structure
- [x] CI-ready: `task check` runs all quality gates
- [x] GitHub Actions CI workflow (`.github/workflows/check.yml`)
- [x] Pi bootstrap script (idempotent: OS packages, overlays, PipeWire, etc.)
- [x] Deploy task (rsync `robot/` to Pi, install deps)
- [x] systemd user service for auto-start on boot (`scripts/robot.service`)
- [x] Document the Pi setup in a changelog or similar

Reference: `.OLD/robotv3/scripts/bootstrap-pi.sh`,
`.OLD/robotv3/DEPLOY_CHANGELOG.md`, `.OLD/robotv3/Taskfile.yml`

### Phase 1 — Core infrastructure

The EventBus, BaseService, configuration, logging, and hardware detection.
No user-visible features yet, but everything else depends on this.

- [x] EventBus (async pub/sub)
- [x] BaseService (lifecycle: init → run → shutdown)
- [x] Configuration (dataclasses, .env for secrets)
- [x] Logging (structured, module-level loggers)
- [x] Hardware detection (Pi vs laptop, available peripherals)
- [x] Main entry point (`__main__.py`)

Reference: `.OLD/robotv3/src/common/` (event_bus.py, hardware.py, config.py,
logger.py), `.OLD/robotv3/src/services/base.py`

### Phase 2 — Display

Show something on the 7" touchscreen. On Pi: dummy SDL driver with
framebuffer blit (see `docs/display-rendering-research.md`). On laptop:
normal pygame window.

- [x] GUIService with platform-aware rendering
- [x] Basic face/expression rendering (static images or simple shapes)
- [x] Desktop mouse/keyboard input (click → `gui.touch`, ESC → `gui.quit`)

Reference: `.OLD/robotv3/src/services/gui.py`,
`docs/display-rendering-research.md`

### Phase 3 — Audio

Microphone capture and speaker playback. Echo cancellation is handled by
PipeWire at the OS level (already configured on the Pi).

- [x] AudioService: mic capture → event bus → speaker playback
- [x] Hardware detection for ReSpeaker HAT vs laptop mic
- [x] Audio format: PCM16, 16 kHz mono (mic), 24 kHz mono (playback)

Reference: `.OLD/robotv3/src/services/audio.py`,
`.OLD/robotv2/services/audio.py`

### Phase 4 — Realtime voice (LLM)

Connect to a speech-to-speech LLM for live conversation. Gemini Flash Live
is the primary provider; architecture should allow swapping providers.

- [x] RealtimeService: WebSocket connection to Gemini Live API
- [x] Bidirectional audio streaming (mic → LLM → speaker)
- [x] Text injection into session (event bus + FIFO for CLI testing)
- [ ] Session resilience (see Phase 4a below) — do this first
- [ ] Tool calling support (LLM can trigger events) — requires stable sessions
- [ ] Provider abstraction (Gemini now, OpenAI later)

Reference: `.OLD/robotv3/src/services/realtime/`,
`.OLD/robotv2/services/realtime/`

#### Tool calling notes (Gemini Live API)

The Live API supports **function calling** in live sessions. Key details for
when we implement this:

- Tools are declared in `LiveConnectConfig` via a `tools` array with standard
  function definitions (name, description, parameters).
- The model sends `BidiGenerateContentToolCall` messages. Unlike the standard
  `generateContent` API, **tool responses must be handled manually** — no
  automatic execution.
- Send results back via `session.send_tool_response()` with `FunctionResponse`
  objects.
- **Async / non-blocking tools**: Set `behavior: "NON_BLOCKING"` on a function
  definition, then use the `scheduling` parameter in the response:
  - `INTERRUPT` — report result immediately, interrupting model speech.
  - `WHEN_IDLE` — wait until the model finishes speaking, then deliver.
  - `SILENT` — inject knowledge without triggering a spoken response.
- `SILENT` mode is useful for injecting vision context (camera observations)
  without interrupting conversation.
- **Google Search grounding** is also available via `tools: [{'google_search': {}}]`.
- Multiple tools can be combined in a single session.

Sources:
- https://ai.google.dev/gemini-api/docs/live-tools

#### Context injection notes

Text injection already works via `send_client_content()`. Additional API
capabilities for future use:

- Conversation history can be sent as `turns` (list of user/model Content
  objects) with `turn_complete=False` for incremental context building.
- For long contexts, the docs recommend sending single-message summaries
  rather than full conversation history.
- `send_realtime_input()` is for streaming (responsive); `send_client_content()`
  is for deterministic ordering (context injection).

Sources:
- https://ai.google.dev/gemini-api/docs/live-guide

### Phase 4a — Session resilience

Make RealtimeService survive connection drops and long sessions. Without this,
Roberta goes silent after ~10–15 minutes with no recovery.

**Problem**: WebSocket connections have a ~10 minute lifetime. Audio-only
sessions max at 15 minutes. The 128k token context window fills up over time.
Currently, if the connection drops, the service stops entirely.

- [x] Reconnect with backoff after a dropped or failed connection.
      Retries with capped exponential backoff instead of stopping the
      service, so it recovers from network drops, Gemini hiccups, and the
      stale-clock-at-boot TLS failure (once NTP corrects the clock, the next
      attempt succeeds). Greets once; context is *not* preserved across
      reconnects yet — that needs the resumption handle below.
- [ ] Session resumption handle (preserve context across reconnects via
      `SessionResumptionUpdate`)
- [ ] Context window compression (sliding window for long sessions)
- [ ] GoAway message handling (graceful reconnect before forced disconnect)
- [ ] Token usage tracking (monitor context window pressure)

#### Session resumption

Enable `sessionResumption` in `LiveConnectConfig`. The server sends
`SessionResumptionUpdate` messages containing a handle. Store the latest
handle and pass it as `SessionResumptionConfig.handle` when reconnecting.
Handles are valid for **2 hours** after disconnection.

Implementation approach:
1. Add `session_resumption` config to `LiveConnectConfig`.
2. Store the latest resumption handle in `self._resumption_handle`.
3. On disconnect (non-shutdown), reconnect with the stored handle.
4. Wrap the session in a reconnection loop with backoff.

#### Context window compression

Enable `contextWindowCompression` in `LiveConnectConfig` with a sliding-window
approach. Configure a token threshold that triggers compression. This allows
sessions to run indefinitely.

#### GoAway handling

The server sends a `GoAway` message with `timeLeft` before disconnecting.
Listen for this and initiate a graceful reconnect (using session resumption)
before the connection is severed. Also handle `generationComplete` which
signals the model's response is done before connection loss.

#### Token usage tracking

Monitor `message.usage_metadata` (available on server messages) to track
`total_token_count` and per-modality breakdowns. Publish as events for
debugging and the debug overlay (Phase 8).

Sources:
- https://ai.google.dev/gemini-api/docs/live-session
- https://ai.google.dev/gemini-api/docs/live-guide

### Phase 5 — Vision

Camera capture and face recognition. Not needed for basic conversation but
adds awareness and personality.

- [ ] VisionService: camera capture with throttling
- [ ] Face detection (OpenCV DNN or similar)
- [ ] Face recognition with vector store (ChromaDB)
- [ ] Publish visual context to LLM

Reference: `.OLD/robotv2/services/gui.py` (face detection code),
`.OLD/robotv2/faces/`

### Phase 6 — Plugins and extensions

A simple plugin system for adding capabilities without modifying core code.

- [ ] Plugin discovery and lifecycle
- [ ] PluginAPI for event access and tool registration
- [ ] Example plugins (greeter, time, weather)

Reference: `.OLD/robotv2/plugins/`, `.OLD/robotv3/src/plugins/`

### Phase 7 — Polish and hardware integration

Follow-up features that refine existing services with Pi-specific hardware
support and quality-of-life improvements.

- [ ] Pi touchscreen input via evdev (SDL dummy driver ignores input devices)
- [ ] Audio waveform visualization on the display (GUIService)

### Phase 8 — Debug overlay

A toggleable debug overlay rendered directly on the 7" touchscreen (or pygame
window on desktop). Streams live session telemetry so we can diagnose issues
without SSH or log tailing.

- [ ] Debug overlay toggle (touch gesture, key press, or event)
- [ ] Token usage display (input/output/total from `usage_metadata`)
- [ ] Thinking summaries (if `include_thoughts=True` is enabled)
- [ ] Tool call log (function name, args, result, scheduling mode)
- [ ] Session status (connected/reconnecting, uptime, resumption handle)
- [ ] Transcript log (scrolling user/model transcript)
- [ ] VAD state indicator (speaking/listening/paused)

Implementation: GUIService subscribes to debug events published by
RealtimeService (`realtime.debug_tokens`, `realtime.debug_tool_call`, etc.)
and renders a semi-transparent overlay on top of the face. Toggled via a
`gui.toggle_debug` event.

---

## Future (not planned yet)

- **Power management** — Witty Pi integration for scheduled wake/sleep
- **Motor control** — ESP32 communication for Wild Thumper chassis
- **Web UI** — local FastAPI dashboard for configuration and monitoring

## Reference

- `.OLD/robotv3/` — previous prototype with working EventBus, AudioService,
  GUIService, RealtimeService (Gemini + OpenAI)
- `.OLD/robotv2/` — earlier prototype with plugin system and face recognition
- `docs/display-rendering-research.md` — findings on Pi display rendering
