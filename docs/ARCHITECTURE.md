# Roberta — Architecture

A living document describing Roberta's architecture. Updated as new services
are added. For the implementation roadmap, see `PLAN.md`.

## Core Concepts

### EventBus

All inter-service communication goes through a central async pub/sub
EventBus. Services never call each other directly.

```
Service A  ──publish("gui.set_expression", data)──►  EventBus
                                                        │
EventBus  ──handler(data)──►  Service B (subscriber)
```

- Handlers are async functions scheduled concurrently via `asyncio.gather`.
- By default, `publish()` awaits handler completion. Pass
  `async_dispatch=True` for fire-and-forget (useful for high-cadence events
  like audio chunks).

### BaseService

Every service inherits from `BaseService` and follows a standard lifecycle:

1. **`__init__(event_bus)`** — Store references and config. No I/O here.
2. **`initialize()`** — Detect hardware, open resources. Set
   `self.running = False` to disable gracefully.
3. **`run()`** — Main async loop, runs until `self.running` is False.
4. **`shutdown()`** — Release resources, join threads, unsubscribe events.

The `Robot` class in `__main__.py` orchestrates this: it calls `initialize()`
on each service, then runs all enabled services concurrently with
`asyncio.gather()`.

### Hardware Detection

Services detect available hardware in `initialize()` and disable themselves
when something is missing. This lets the same code run on both the Pi
(production) and a laptop (development) without conditional imports or
feature flags.

Detection functions live in `hardware.py`:

| Function          | Checks                                  |
|-------------------|-----------------------------------------|
| `detect_display`  | X11, Wayland, or `/dev/fb0` framebuffer |
| `detect_audio_device` | Named audio device via sounddevice  |

`config.is_raspberry_pi()` reads `/proc/device-tree/model` to distinguish
Pi from laptop.

## Threading Model

```
Main thread (asyncio event loop)
├── Robot orchestrator
├── EventBus
├── Service.run() coroutines
│
└── Worker threads (spawned by services that need them)
    ├── GUIWorker — pygame event loop + rendering
    └── AudioWorker — sounddevice input/output streams
```

**Why threads?** Some libraries (pygame, sounddevice) have blocking event
loops or callbacks that would stall the async event loop. These run in
dedicated daemon threads.

**Rule:** The async event loop is the source of truth. Worker threads
communicate back to it via `asyncio.run_coroutine_threadsafe()` to publish
events.

### Cross-Thread Communication

There are two patterns for passing data between the async main thread and
worker threads:

**1. Events (thread → async):** Worker threads use
`asyncio.run_coroutine_threadsafe(bus.publish(...), loop)` to schedule event
publications on the main loop. This is used for gui.quit, gui.touch, etc.

**2. Shared state (async → thread):** When the main thread needs to push
state into a worker thread (e.g. changing the current expression), the
service stores it in a shared attribute protected by a `threading.Lock`.
The worker thread reads it on its next iteration.

**3. Queues (bidirectional):** `queue.Queue` is inherently thread-safe and
works well for streaming data. AudioService uses `_mic_queue` (thread →
async) and `_speaker_queue` (async → thread) to pass PCM audio bytes
without locks.

**Lock discipline:** Any attribute accessed by both threads must be read and
written under the same lock. Group related fields into a single lock
acquisition to keep them consistent. See GUIService `_state_lock` for the
reference implementation.

## Services

### GUIService

Renders facial expressions on the 7" touchscreen (Pi) or a pygame window
(laptop).

**Events:**

| Direction   | Event                | Payload               |
|-------------|----------------------|-----------------------|
| Publishes   | `gui.quit`           | `{}`                  |
| Publishes   | `gui.touch`          | `{x: int, y: int}`   |
| Subscribes  | `gui.set_expression` | `{expression: str}`   |

**Platform rendering:**

| Environment          | SDL driver | Output              |
|----------------------|------------|---------------------|
| Laptop (X11/Wayland) | auto       | `pygame.display.flip()` |
| Pi (headless)        | `dummy`    | RGB565 blit to `/dev/fb0` |

See `docs/display-rendering-research.md` for the Pi rendering approach.

**Threading:** Pygame runs in a `GUIWorker` daemon thread. The async
`run()` method just sleeps in a loop until the stop event is set.

**Shared state and `_state_lock`:** The `current_expression` and
`_needs_redraw` fields are written by the async event handler
(`_handle_set_expression`) and read by the pygame thread. Both sides
acquire `_state_lock` before accessing these fields. Blink state
(`_is_blinking`, `_blink_end_ms`, etc.) is only touched by the pygame
thread and does not need the lock.

**Dirty-flag rendering:** The pygame thread only re-renders when
`_needs_redraw` is True (expression change or blink). When idle it polls at
10 FPS for event/blink checks; during a blink it runs at the configured FPS
(default 60).

### AudioService

Microphone capture and speaker playback via sounddevice. Foundation for
Phase 4 (realtime voice with Gemini Live).

**Events:**

| Direction   | Event                  | Payload                            |
|-------------|------------------------|------------------------------------|
| Publishes   | `audio.mic_chunk`      | `{audio: str, sample_rate: int}`   |
| Subscribes  | `audio.play_chunk`     | `{audio: str, sample_rate: int}`   |
| Subscribes  | `audio.start_recording`| `{}`                               |
| Subscribes  | `audio.stop_recording` | `{}`                               |
| Subscribes  | `audio.stop_playback`  | `{}`                               |

Audio payloads are base64-encoded PCM16 (int16 little-endian). Mic chunks
are 16 kHz mono, playback chunks are 24 kHz mono (matching Gemini Live).

**Device selection priority:**

| Priority | Device                   | Environment       |
|----------|--------------------------|-------------------|
| 1        | `echo_cancel_source`     | Pi (PipeWire AEC) |
| 2        | `seeed-2mic-voicecard`   | Pi (raw mic)      |
| 3        | System default input     | Laptop / fallback |

Output always uses system default — on Pi, PipeWire routes to
`echo_cancel_sink` automatically.

**Threading:** The `AudioWorker` daemon thread opens separate
`sd.InputStream` (16 kHz, int16) and `sd.OutputStream` (24 kHz, int16)
with callbacks. Mic data flows through `_mic_queue` to the async `run()`
loop, which base64-encodes and publishes `audio.mic_chunk`. Playback data
flows through `_speaker_queue` to the output callback.

**Recording gate:** `recording_enabled` bool toggled by start/stop events.
Single bool — GIL makes reads/writes atomic. This is belt-and-suspenders on
top of PipeWire AEC: the RealtimeService (Phase 4) will pause mic capture
while the model speaks.

**Playback gain:** Applied in `_apply_gain()` when handling `play_chunk`,
before queueing. Keeps the output callback simple and avoids per-frame
multiplication.

### RealtimeService

Real-time voice conversation via Google Gemini Live. Connects over WebSocket,
streams mic audio to the model, and plays back model audio through the speaker.

**Events:**

| Direction   | Event                       | Payload                            |
|-------------|-----------------------------|------------------------------------|
| Publishes   | `realtime.connected`        | `{provider: str, model: str}`      |
| Publishes   | `realtime.disconnected`     | `{provider: str}`                  |
| Publishes   | `realtime.response_started` | `{}`                               |
| Publishes   | `realtime.response_completed`| `{}`                              |
| Publishes   | `realtime.interrupted`      | `{}`                               |
| Publishes   | `realtime.user_transcript`  | `{text: str}`                      |
| Publishes   | `realtime.model_transcript` | `{text: str}`                      |
| Publishes   | `realtime.error`            | `{error: str}`                     |
| Subscribes  | `audio.mic_chunk`           | `{audio: str, sample_rate: int}`   |
| Subscribes  | `realtime.inject_text`      | `{text: str}`                      |

**Session lifecycle:**

1. `initialize()` checks for `GEMINI_API_KEY` — disables gracefully if absent.
2. `run()` connects via `client.aio.live.connect()`, sends a greeting prompt,
   then enters `_response_loop()`.
3. `_response_loop()` calls `session.receive()` in an outer `while` loop.
   Each `receive()` yields messages for one turn and stops at `turn_complete`;
   the outer loop starts a new `receive()` for the next turn.
4. `shutdown()` cancels background tasks, closes the session, and unsubscribes.

**Echo cancellation (two layers):**

| Layer    | Mechanism                     | Trade-off                         |
|----------|-------------------------------|-----------------------------------|
| Hardware | PipeWire AEC on the Pi        | Best quality, requires Pi config  |
| Software | Mic paused during model speech| No barge-in while model speaks    |

The software layer (`_pause_mic` / `_resume_mic`) is belt-and-suspenders on
top of PipeWire AEC. It can be disabled later to test true barge-in.

**Text injection:** Any service can publish `realtime.inject_text` with
`{text: "..."}` to send context to the model (e.g. face detections, CLI
testing). A background FIFO reader at `/tmp/roberta.fifo` bridges the CLI:
`task inject -- "message"` writes to the FIFO, which publishes the event.

**Deferred imports:** `google-genai` is imported inside `run()` to avoid
`ImportError` in CI / environments without the dependency. The `types` module
is stored on `self._types` so event handlers can build API objects outside
`run()`'s scope.

**Transcripts:** User transcripts arrive as complete sentences and are logged
immediately. Model transcripts arrive word-by-word and are accumulated in
`_model_transcript_buffer`, then logged and published as a single string at
`turn_complete`.
