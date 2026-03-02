"""
BaseService - Abstract base class for all services.

All services inherit from this class and follow a standard lifecycle:
1. __init__: Store event bus reference and configuration
2. initialize(): Set up hardware, connections, resources (can fail gracefully)
3. run(): Main async loop (runs until shutdown)
4. shutdown(): Clean up resources (optional)
"""

from abc import ABC, abstractmethod

from robot.event_bus import EventBus
from robot.logger import get_logger


class BaseService(ABC):
    """
    Abstract base for all long-running services.

    Services are the main building blocks. Each service handles one major
    subsystem (audio, GUI, vision, etc.) and communicates with other services
    exclusively through the EventBus.
    """

    def __init__(self, event_bus: EventBus) -> None:
        """
        Initialize the service.

        Args:
            event_bus: The central event bus for inter-service communication
        """
        self.event_bus = event_bus
        self.logger = get_logger(type(self).__name__)
        self.running = False

    async def initialize(self) -> None:
        """
        Initialize hardware, connections, or resources.

        This is called before run() and is where services should:
        - Detect and verify hardware availability
        - Set up connections or resources
        - Set self.running = False if initialization fails

        Default implementation does nothing and sets running = True.
        Override this if your service needs setup before run().
        """
        self.running = True

    @abstractmethod
    async def run(self) -> None:
        """
        Main service loop.

        This must be implemented by all service subclasses. The run() method
        should loop while self.running is True, doing the service's main work.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement run()"
        )

    async def shutdown(self) -> None:
        """
        Clean up resources.

        Override this if your service needs to clean up resources on shutdown
        (close files, disconnect from servers, release hardware, etc.).

        Default implementation just sets running = False.
        """
        self.running = False
        self.logger.info("Shutdown complete")
