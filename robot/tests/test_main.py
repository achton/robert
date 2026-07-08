"""Tests for the main entry point (Robot class, signal handling, shutdown)."""

import asyncio
import os
import signal
from unittest.mock import AsyncMock, patch

from robot.__main__ import Robot, main
from robot.base_service import BaseService
from robot.event_bus import EventBus

# -- Helpers ---------------------------------------------------------


class FakeService(BaseService):
    """Minimal service that runs until stopped."""

    def __init__(self, event_bus: EventBus) -> None:
        super().__init__(event_bus)
        self._stop = asyncio.Event()

    async def initialize(self) -> None:
        self.running = True

    async def run(self) -> None:
        await self._stop.wait()

    async def shutdown(self) -> None:
        self._stop.set()
        await super().shutdown()


class BrokenShutdownService(FakeService):
    """Service whose shutdown() raises an exception."""

    async def shutdown(self) -> None:
        self._stop.set()
        raise RuntimeError("shutdown failed")


def _make_robot(*services: BaseService) -> Robot:
    """Create a Robot with specific services (bypasses real hardware)."""
    robot = Robot.__new__(Robot)
    robot.logger = Robot.__init__.__code__  # unused, just needs to exist
    # Re-do the bits of __init__ we need, with our fake services
    from robot.logger import get_logger

    robot.logger = get_logger("Robot")
    robot.event_bus = services[0].event_bus if services else EventBus()
    robot._run_task = None
    robot._shutdown_done = False
    robot.services = list(services)
    return robot


# -- Tests: SIGTERM --------------------------------------------------


class TestSigtermShutdown:
    """SIGTERM (systemd stop) triggers a clean shutdown."""

    async def test_sigterm_triggers_shutdown(self):
        """Sending SIGTERM to the process causes main() to exit cleanly."""
        mock_robot = AsyncMock()
        mock_robot.services = []
        mock_robot._shutdown_done = False

        async def fake_run():
            # Simulate a long-running service loop
            await asyncio.sleep(10)

        mock_robot.run = fake_run

        with patch("robot.__main__.Robot", return_value=mock_robot):
            task = asyncio.create_task(main())

            # Give main() time to register the signal handler
            await asyncio.sleep(0.05)

            # Send SIGTERM to ourselves (same as systemd stop)
            os.kill(os.getpid(), signal.SIGTERM)

            # main() should exit promptly — without the handler this hangs
            await asyncio.wait_for(task, timeout=2.0)

        mock_robot.shutdown.assert_awaited()


# -- Tests: Robot.shutdown() -----------------------------------------


class TestRobotShutdown:
    """Robot.shutdown() behaviour."""

    async def test_shutdown_calls_each_service(self):
        """Every service's shutdown() is called."""
        bus = EventBus()
        svc_a = FakeService(bus)
        svc_b = FakeService(bus)
        svc_a.running = True
        svc_b.running = True
        robot = _make_robot(svc_a, svc_b)

        await robot.shutdown()

        assert svc_a.running is False
        assert svc_b.running is False

    async def test_shutdown_is_idempotent(self):
        """Calling shutdown() twice only shuts down services once."""
        bus = EventBus()
        svc = FakeService(bus)
        svc.running = True
        robot = _make_robot(svc)

        await robot.shutdown()
        assert robot._shutdown_done is True

        # Replace shutdown with a mock to prove it isn't called again
        svc.shutdown = AsyncMock()
        await robot.shutdown()
        svc.shutdown.assert_not_awaited()

    async def test_shutdown_continues_after_service_error(self):
        """If one service's shutdown() raises, the others still run."""
        bus = EventBus()
        broken = BrokenShutdownService(bus)
        healthy = FakeService(bus)
        broken.running = True
        healthy.running = True
        robot = _make_robot(broken, healthy)

        # Should not raise, even though broken service explodes
        await robot.shutdown()

        assert healthy.running is False


# -- Tests: Robot.run() ----------------------------------------------


class TestRobotRun:
    """Robot.run() behaviour."""

    async def test_run_exits_when_no_services_enabled(self):
        """run() returns immediately if all services are disabled."""
        bus = EventBus()
        svc = FakeService(bus)
        svc.running = False
        robot = _make_robot(svc)

        # Should return without blocking
        await asyncio.wait_for(robot.run(), timeout=1.0)


# -- Tests: gui.quit event ------------------------------------------


class TestGuiQuitShutdown:
    """The gui.quit event (ESC / window close) triggers shutdown."""

    async def test_gui_quit_shuts_down_and_exits(self):
        """Publishing gui.quit stops all services and exits run()."""
        bus = EventBus()
        svc = FakeService(bus)
        svc.running = True
        robot = _make_robot(svc)

        # Start run() in a task
        run_task = asyncio.create_task(robot.run())

        # Give run() time to subscribe and start the gather
        await asyncio.sleep(0.05)

        # Simulate ESC press / window close
        await bus.publish("gui.quit", {})

        # run() should exit promptly
        await asyncio.wait_for(run_task, timeout=2.0)

        assert robot._shutdown_done is True
        assert svc.running is False
