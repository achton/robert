"""
EventBus - Central pub/sub coordination hub.

The EventBus is the backbone of all inter-service communication. It is kept
intentionally simple: a dictionary mapping event types to lists of async callbacks.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any


class EventBus:
    """
    Async event bus for coordinating services.

    Services communicate exclusively through events. No direct coupling between
    services is allowed. This keeps the architecture simple and extensible.
    """

    def __init__(self) -> None:
        self.subscribers: dict[str, list[Callable[[Any], Coroutine]]] = (
            defaultdict(list)
        )
        # Wiretaps receive *every* event (type + data). Used for observers
        # like the dashboard that should not be coupled to specific event
        # types. Wiretaps must never block the bus: they are dispatched
        # fire-and-forget and their exceptions are logged, not raised.
        self._wiretaps: list[Callable[[str, Any], Coroutine]] = []
        self.logger = logging.getLogger("EventBus")
        self._pending_tasks: set[asyncio.Task[Any]] = set()

    def _make_done_callback(
        self, event_type: str, handler: Callable[[Any], Coroutine]
    ) -> Callable[[asyncio.Task[Any]], None]:
        """Create a callback that logs handler failures."""

        def _callback(task: asyncio.Task[Any]) -> None:
            self._pending_tasks.discard(task)
            if task.cancelled():
                return

            try:
                exception = task.exception()
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.error(
                    "Error retrieving result for %s handler %s: %s",
                    event_type,
                    getattr(handler, "__name__", repr(handler)),
                    exc,
                )
                return

            if exception:
                self.logger.error(
                    "Handler %s failed for event %s: %s",
                    getattr(handler, "__name__", repr(handler)),
                    event_type,
                    exception,
                )

        return _callback

    def subscribe(
        self, event_type: str, callback: Callable[[Any], Coroutine]
    ) -> None:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: The event type to subscribe to (e.g., "audio.mic_chunk")
            callback: Async function to call when event is published
        """
        self.subscribers[event_type].append(callback)
        self.logger.debug(f"Subscribed to {event_type}: {callback.__name__}")

    def add_wiretap(self, callback: Callable[[str, Any], Coroutine]) -> None:
        """
        Register a callback that receives every published event.

        The callback is invoked as ``callback(event_type, data)``. It is
        dispatched fire-and-forget; slow or failing wiretaps do not block
        or crash the bus.
        """
        self._wiretaps.append(callback)

    def remove_wiretap(
        self, callback: Callable[[str, Any], Coroutine]
    ) -> None:
        """Remove a previously registered wiretap."""
        if callback in self._wiretaps:
            self._wiretaps.remove(callback)

    def unsubscribe(
        self, event_type: str, callback: Callable[[Any], Coroutine]
    ) -> None:
        """
        Unsubscribe a handler from an event type.

        Args:
            event_type: The event type to unsubscribe from
            callback: The callback function to remove
        """
        if callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            self.logger.debug(
                f"Unsubscribed from {event_type}: {callback.__name__}"
            )

    async def publish(
        self,
        event_type: str,
        data: Any = None,
        *,
        async_dispatch: bool = False,
    ) -> None:
        """
        Publish an event to all registered handlers.

        Handlers are always scheduled concurrently. By default this method awaits
        their completion to preserve deterministic behavior in most services and
        tests. Set async_dispatch=True for high-cadence publishers that must
        return immediately.

        Args:
            event_type: The event type to publish (e.g., "audio.mic_chunk")
            data: Optional data to pass to handlers
            async_dispatch: Whether to skip awaiting handler completion
        """
        # Fan out to wiretaps first, fire-and-forget. Done before the early
        # return so observers see events even when nothing is subscribed.
        for tap in self._wiretaps:
            try:
                task = asyncio.create_task(tap(event_type, data))
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.error(
                    "Wiretap %s failed for event %s: %s",
                    getattr(tap, "__name__", repr(tap)),
                    event_type,
                    exc,
                )

        handlers = self.subscribers.get(event_type, [])

        if not handlers:
            self.logger.debug(
                f"Event {event_type} published with no subscribers"
            )
            return

        self.logger.debug(
            f"Publishing {event_type} to {len(handlers)} handler(s)"
        )

        tasks: list[asyncio.Task[Any]] = []
        for handler in handlers:
            task = asyncio.create_task(handler(data))
            self._pending_tasks.add(task)
            task.add_done_callback(
                self._make_done_callback(event_type, handler)
            )
            tasks.append(task)

        if not async_dispatch:
            await asyncio.gather(*tasks, return_exceptions=True)
