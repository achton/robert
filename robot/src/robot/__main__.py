"""
Main entry point.

This is the main robot runtime that coordinates all services through
the event bus. Run with: python -m robot
"""

import asyncio
import sys
from typing import Any

from robot.base_service import BaseService
from robot.config import is_raspberry_pi
from robot.event_bus import EventBus
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
        self.services: list[BaseService] = []

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
            f"Initialization complete." f" {len(enabled)} service(s) enabled."
        )

    async def run(self) -> None:
        """Run all enabled services concurrently."""
        enabled = [s for s in self.services if s.running]

        if not enabled:
            return

        self.logger.info(f"Starting {len(enabled)} service(s)...")

        # Subscribe to quit event
        self.event_bus.subscribe("quit", self._handle_quit)

        try:
            await asyncio.gather(*[service.run() for service in enabled])
        except Exception as e:
            self.logger.error(f"Service error: {e}")
        finally:
            self.logger.info("Services stopped")

    async def shutdown(self) -> None:
        """Gracefully shutdown all services."""
        self.logger.info("Shutting down...")

        for service in self.services:
            service_name = type(service).__name__
            try:
                await service.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down {service_name}: {e}")

        self.logger.info("Shutdown complete")

    async def _handle_quit(self, _data: Any) -> None:
        """Handle quit event — signal all services to stop."""
        self.logger.info("Quit event received")
        for service in self.services:
            service.running = False


async def main() -> None:
    """Main entry point."""
    robot = Robot()

    try:
        await robot.initialize()
        await robot.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        await robot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
