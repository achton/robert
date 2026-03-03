"""
Main entry point.

This is the main robot runtime that coordinates all services through
the event bus. Run with: python -m robot
"""

import asyncio
import contextlib
import os
import signal
import sys
from typing import Any

from robot.audio_service import AudioService
from robot.base_service import BaseService
from robot.config import AudioConfig, GUIConfig, is_raspberry_pi
from robot.event_bus import EventBus
from robot.gui_service import GUIService
from robot.hardware import detect_display
from robot.logger import get_logger


class Robot:
    """
    Main robot runtime that coordinates all services.

    The Robot class is responsible for:
    1. Creating the event bus
    2. Instantiating all services
    3. Initializing services (which may fail gracefully)
    4. Running all enabled services concurrently
    5. Handling shutdown signals
    """

    def __init__(self) -> None:
        self.logger = get_logger("Robot")
        self.event_bus = EventBus()
        self._run_task: asyncio.Task[None] | None = None
        self._shutdown_done = False
        self.services: list[BaseService] = [
            GUIService(self.event_bus, GUIConfig()),
            AudioService(self.event_bus, AudioConfig()),
        ]

    async def initialize(self) -> None:
        """Initialize all services and log hardware detection results."""
        platform = "Raspberry Pi" if is_raspberry_pi() else "laptop/desktop"
        self.logger.info(f"Initializing on {platform}...")

        # Log what hardware is available
        display = detect_display()
        self.logger.info(f"Display available: {display}")

        # Initialize each service
        for service in self.services:
            service_name = type(service).__name__
            try:
                self.logger.info(f"Initializing {service_name}...")
                await service.initialize()

                if service.running:
                    self.logger.info(
                        f"{service_name} initialized successfully"
                    )
                else:
                    self.logger.warning(
                        f"{service_name} disabled" " (hardware not available)"
                    )

            except Exception as e:
                self.logger.error(f"Failed to initialize {service_name}: {e}")
                service.running = False

        # Check if at least one service is running
        enabled = [s for s in self.services if s.running]
        if not enabled:
            self.logger.info("No services registered. Nothing to run.")
            return

        self.logger.info(
            f"Initialization complete. {len(enabled)} service(s) enabled."
        )

    async def run(self) -> None:
        """Run all enabled services concurrently."""
        enabled = [s for s in self.services if s.running]

        if not enabled:
            return

        self.logger.info(f"Starting {len(enabled)} service(s)...")

        # Subscribe to quit event (published by GUIService on ESC / window close)
        self.event_bus.subscribe("gui.quit", self._handle_quit)

        # Wrap the gather in a task so _handle_quit can cancel it
        self._run_task = asyncio.current_task()

        try:
            await asyncio.gather(*[service.run() for service in enabled])
        except asyncio.CancelledError:
            self.logger.info("Services cancelled")
        except Exception as e:
            self.logger.error(f"Service error: {e}")
        finally:
            self.logger.info("Services stopped")

    async def shutdown(self) -> None:
        """Gracefully shutdown all services. Safe to call multiple times."""
        if self._shutdown_done:
            return
        self._shutdown_done = True

        self.logger.info("Shutting down...")

        for service in self.services:
            service_name = type(service).__name__
            try:
                await service.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down {service_name}: {e}")

        self.logger.info("Shutdown complete")

    async def _handle_quit(self, _data: Any) -> None:
        """Handle gui.quit — shut down all services and cancel the run loop."""
        self.logger.info("Quit event received, shutting down...")
        await self.shutdown()
        # Cancel the run() task so asyncio.gather() exits
        if self._run_task:
            self._run_task.cancel()


async def main() -> None:
    """Main entry point."""
    robot = Robot()

    # Handle SIGTERM (sent by systemd stop) via an asyncio.Event.
    # Without this, pygame/SDL installs its own SIGTERM handler that
    # enqueues a pygame event — but on the Pi the event queue is never
    # polled, so the signal is swallowed and systemd hangs.
    sigterm_received = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, sigterm_received.set)

    try:
        await robot.initialize()

        # Run services until they stop on their own OR SIGTERM arrives
        run_task = asyncio.create_task(robot.run())
        sigterm_task = asyncio.create_task(sigterm_received.wait())

        _done, pending = await asyncio.wait(
            [run_task, sigterm_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        await robot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

    # Force-exit to avoid hanging on pygame's atexit handler.
    # pygame registers pygame.quit() as an atexit callback, and
    # SDL_Quit() blocks indefinitely on the Pi with the dummy video
    # driver. Our shutdown() already cleans up everything we need.
    os._exit(0)
