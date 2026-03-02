"""Tests for the EventBus pub/sub system."""

import asyncio

import pytest

from robot.event_bus import EventBus


class TestEventBus:
    """Test suite for EventBus."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        """Test basic subscribe and publish functionality."""
        bus = EventBus()
        received = []

        async def handler(data):
            received.append(data)

        bus.subscribe("test.event", handler)
        await bus.publish("test.event", {"message": "hello"})

        assert len(received) == 1
        assert received[0]["message"] == "hello"

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribe functionality."""
        bus = EventBus()
        received = []

        async def handler(data):
            received.append(data)

        bus.subscribe("test.event", handler)
        await bus.publish("test.event", {"message": "first"})

        bus.unsubscribe("test.event", handler)
        await bus.publish("test.event", {"message": "second"})

        assert len(received) == 1
        assert received[0]["message"] == "first"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        """Test multiple subscribers to the same event."""
        bus = EventBus()
        received_a = []
        received_b = []

        async def handler_a(data):
            received_a.append(data)

        async def handler_b(data):
            received_b.append(data)

        bus.subscribe("test.event", handler_a)
        bus.subscribe("test.event", handler_b)
        await bus.publish("test.event", {"message": "broadcast"})

        assert len(received_a) == 1
        assert len(received_b) == 1
        assert received_a[0]["message"] == "broadcast"
        assert received_b[0]["message"] == "broadcast"

    @pytest.mark.asyncio
    async def test_publish_with_no_subscribers(self):
        """Publishing to an event with no subscribers should not error."""
        bus = EventBus()
        await bus.publish("no.subscribers", {"message": "hello"})

    @pytest.mark.asyncio
    async def test_multiple_events(self):
        """Subscribers only receive their own event type."""
        bus = EventBus()
        received_a = []
        received_b = []

        async def handler_a(data):
            received_a.append(data)

        async def handler_b(data):
            received_b.append(data)

        bus.subscribe("event.a", handler_a)
        bus.subscribe("event.b", handler_b)

        await bus.publish("event.a", {"message": "A"})
        await bus.publish("event.b", {"message": "B"})

        assert len(received_a) == 1
        assert len(received_b) == 1
        assert received_a[0]["message"] == "A"
        assert received_b[0]["message"] == "B"

    @pytest.mark.asyncio
    async def test_async_dispatch_delivery(self):
        """Async-dispatch mode still delivers events."""
        bus = EventBus()
        event_received = asyncio.Event()

        async def handler(_data):
            event_received.set()

        bus.subscribe("fast.event", handler)
        await bus.publish("fast.event", {}, async_dispatch=True)

        await asyncio.wait_for(event_received.wait(), timeout=0.1)
