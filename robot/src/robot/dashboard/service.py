"""
DashboardService - Streams every event-bus event to external observers.

Clients connect over HTTP and receive a Server-Sent Events (SSE) stream
where each event is a single JSON line of the form::

    {"t": 1713350000.123, "type": "audio.mic_level", "data": {"level": 0.42}}

The service is read-only: it observes the bus via a wiretap and never
publishes anything itself. It uses the stdlib only (``asyncio``) so no
new runtime dependency is added.

Large audio payloads (``audio.mic_chunk``, ``audio.play_chunk``) would
dominate the stream, so their base64 bodies are replaced with a short
summary before sending.
"""

import asyncio
import contextlib
import json
import time
from typing import Any

from robot.base_service import BaseService
from robot.config import DashboardConfig
from robot.event_bus import EventBus

# Events whose ``audio`` field is a base64 blob — redacted so the stream
# stays lightweight. Everything else goes out as-is.
_REDACT_AUDIO_EVENTS = frozenset({"audio.mic_chunk", "audio.play_chunk"})

# Per-client backpressure: if a slow client falls further behind than
# this, we drop events for that client rather than stall the bus.
_CLIENT_QUEUE_MAX = 256

# Keep-alive interval. Browsers/proxies may close idle SSE connections,
# so we emit an SSE comment every so often.
_KEEPALIVE_SECONDS = 15.0


class DashboardService(BaseService):
    """
    Streams every event-bus event to connected SSE clients.

    Lifecycle:
        initialize() registers a wiretap on the event bus.
        run()       starts an asyncio TCP server on config.host:config.port.
        shutdown()  closes the server and drops the wiretap.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: DashboardConfig | None = None,
    ) -> None:
        super().__init__(event_bus)
        self.config = config or DashboardConfig()
        self._server: asyncio.Server | None = None
        self._clients: set[asyncio.Queue[str]] = set()

    async def initialize(self) -> None:
        if not self.config.enabled:
            self.logger.info("Dashboard disabled by config")
            self.running = False
            return

        self.event_bus.add_wiretap(self._on_event)
        self.running = True

    async def run(self) -> None:
        if not self.running:
            return

        self._server = await asyncio.start_server(
            self._handle_client, self.config.host, self.config.port
        )
        self.logger.info(
            "Dashboard SSE listening on http://%s:%d/events",
            self.config.host,
            self.config.port,
        )
        async with self._server:
            await self._server.serve_forever()

    async def shutdown(self) -> None:
        self.event_bus.remove_wiretap(self._on_event)
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        self.running = False
        self.logger.info("Shutdown complete")

    async def _on_event(self, event_type: str, data: Any) -> None:
        """Fan an event out to every connected client."""
        if not self._clients:
            return

        payload = json.dumps(
            {
                "t": time.time(),
                "type": event_type,
                "data": _render(event_type, data),
            },
            default=repr,
        )

        for queue in list(self._clients):
            # Never block the bus. If a client is too slow, drop events
            # for it — the TUI will still see everything else.
            if queue.full():
                continue
            queue.put_nowait(payload)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")

        # Consume the HTTP request (we don't route on path — every
        # request gets the event stream).
        try:
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
        except ConnectionError:
            writer.close()
            return

        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Connection: keep-alive\r\n"
            b"Access-Control-Allow-Origin: *\r\n"
            b"\r\n"
        )
        try:
            await writer.drain()
        except ConnectionError:
            writer.close()
            return

        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=_CLIENT_QUEUE_MAX)
        self._clients.add(queue)
        self.logger.info("Dashboard client connected: %s", peer)

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(
                        queue.get(), timeout=_KEEPALIVE_SECONDS
                    )
                    writer.write(f"data: {msg}\n\n".encode())
                except TimeoutError:
                    # SSE comment line — keeps the connection warm.
                    writer.write(b": keepalive\n\n")
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            self._clients.discard(queue)
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()
            self.logger.info("Dashboard client disconnected: %s", peer)


def _render(event_type: str, data: Any) -> Any:
    """
    Turn event data into something compact and JSON-safe.

    Redacts base64 audio on the hot events and falls back to ``repr``
    for anything unserialisable.
    """
    if (
        event_type in _REDACT_AUDIO_EVENTS
        and isinstance(data, dict)
        and "audio" in data
    ):
        redacted = dict(data)
        audio = redacted.pop("audio")
        # base64 → ~4/3 of byte length; report byte length for clarity.
        if isinstance(audio, str):
            redacted["audio_bytes"] = (len(audio) * 3) // 4
        return redacted
    return data
