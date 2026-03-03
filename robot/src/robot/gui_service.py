"""
GUIService — Manages the display with pygame.

Renders facial expressions on the 7" touchscreen (Pi) or a pygame window
(laptop). Pygame runs in a dedicated worker thread to avoid blocking the
async event loop.

Platform behaviour:
- Laptop (X11/Wayland): normal pygame window, SDL auto-detects driver.
- Pi (headless): SDL dummy driver renders offscreen, then blit to /dev/fb0
  as RGB565. See docs/display-rendering-research.md for details.
"""

import asyncio
import os
import random
import threading
from pathlib import Path
from typing import Any

from robot.base_service import BaseService
from robot.config import GUIConfig, is_raspberry_pi
from robot.event_bus import EventBus
from robot.hardware import detect_display

# Directory containing the face expression PNG files
ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
EXPRESSIONS_DIR = ASSETS_DIR / "expressions"

# Map expression names to image filenames (without extension).
# Matches the naming from robotv2.
# TODO: sad, scared, and surprised reuse other faces because we only
# have 4 PNGs. Add dedicated assets when available.
EXPRESSION_MAP: dict[str, str] = {
    "neutral": "face-smiling",
    "smiling": "face-smiling",
    "happy": "face-happy",
    "angry": "face-angry",
    "sad": "face-angry",
    "surprised": "face-happy",
    "scared": "face-angry",
    "closed": "face-closed",
    "init": "face-closed",
}


def _has_desktop_environment() -> bool:
    """Check if a desktop environment (X11 or Wayland) is available."""
    return "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ


class GUIService(BaseService):
    """
    GUI service for display management.

    Publishes:
        gui.quit — Window closed or ESC pressed, signals shutdown
        gui.touch — Mouse click / touch, payload: {x: int, y: int}

    Subscribes:
        gui.set_expression — Change facial expression, payload:
                             {expression: str}
    """

    def __init__(
        self, event_bus: EventBus, config: GUIConfig | None = None
    ) -> None:
        super().__init__(event_bus)
        self.config = config or GUIConfig()

        # Expression state — shared between the async event handler
        # (writer) and the pygame thread (reader). Always access
        # current_expression and _needs_redraw under _state_lock.
        self._state_lock = threading.Lock()
        self.current_expression = self.config.default_expression
        self._gui_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Set by the worker thread after pygame init
        self._screen: Any = None
        self._clock: Any = None

        # Framebuffer file handle (Pi only, kept open for performance)
        self._fb: Any = None

        # Whether we are rendering to the framebuffer (Pi headless mode)
        self._use_framebuffer = False

        # Cached loaded expression surfaces (populated in worker thread)
        self._expression_images: dict[str, Any] = {}

        # Blink timing
        self._last_blink_ms = 0
        self._blink_interval_ms = random.randint(3000, 10000)
        self._is_blinking = False
        self._blink_end_ms = 0

        # Dirty flag — only redraw when something changed.
        # Protected by _state_lock (see above).
        self._needs_redraw = True

    async def initialize(self) -> None:
        """Check for display hardware and configure SDL driver."""
        if not detect_display():
            self.logger.warning("No display detected. GUI service disabled.")
            self.running = False
            return

        # On Pi without a desktop: use SDL dummy driver + framebuffer
        if is_raspberry_pi() and not _has_desktop_environment():
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self._use_framebuffer = True
            self.logger.info("Pi headless mode: SDL dummy + /dev/fb0")
        else:
            self.logger.info("Desktop mode: SDL auto-detect")

        self.running = True

    async def run(self) -> None:
        """Start the pygame worker thread and wait until stopped."""
        if not self.running:
            return

        # Subscribe to expression change events
        self.event_bus.subscribe(
            "gui.set_expression", self._handle_set_expression
        )

        # Start the pygame worker thread, passing the async event loop
        # so the thread can schedule events back onto it
        main_loop = asyncio.get_running_loop()
        self._gui_thread = threading.Thread(
            target=self._pygame_worker,
            args=(main_loop,),
            daemon=True,
            name="GUIWorker",
        )
        self._gui_thread.start()

        # Keep the async run() alive until stopped
        while self.running and not self._stop_event.is_set():
            await asyncio.sleep(0.1)

    def _pygame_worker(self, main_loop: asyncio.AbstractEventLoop) -> None:
        """
        Pygame event loop — runs in a dedicated thread.

        Handles display init, event processing, rendering, and blitting
        to the framebuffer on Pi.
        """
        import pygame

        try:
            pygame.init()

            self._screen = pygame.display.set_mode(
                (self.config.width, self.config.height)
            )
            pygame.display.set_caption("Roberta")
            pygame.mouse.set_visible(False)
            self._clock = pygame.time.Clock()

            # Open framebuffer for Pi headless rendering.
            if self._use_framebuffer:
                try:
                    # Kept open for seek+write each frame; closed in finally.
                    self._fb = open("/dev/fb0", "wb")  # noqa: SIM115
                except OSError as err:
                    self.logger.error(f"Cannot open /dev/fb0: {err}")
                    self._stop_event.set()
                    return

            # Load all expression images
            self._load_expressions(pygame)

            self._last_blink_ms = pygame.time.get_ticks()
            self.logger.info("Pygame initialized")

            # --- Main render loop ---
            while not self._stop_event.is_set():
                # Process pygame events (only meaningful on desktop)
                if not self._use_framebuffer:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.logger.info("Window closed")
                            self._publish_from_thread(
                                main_loop, "gui.quit", {}
                            )
                            self._stop_event.set()
                            break

                        if (
                            event.type == pygame.KEYDOWN
                            and event.key == pygame.K_ESCAPE
                        ):
                            self.logger.info("ESC pressed")
                            self._publish_from_thread(
                                main_loop, "gui.quit", {}
                            )
                            self._stop_event.set()
                            break

                        if event.type == pygame.MOUSEBUTTONDOWN:
                            x, y = event.pos
                            self._publish_from_thread(
                                main_loop,
                                "gui.touch",
                                {"x": x, "y": y},
                            )

                # Handle auto-blink (may set _needs_redraw)
                self._update_blink(pygame)

                # Snapshot and reset the dirty flag. The lock
                # protects against the async event handler writing
                # current_expression + _needs_redraw concurrently.
                with self._state_lock:
                    needs_redraw = self._needs_redraw
                    self._needs_redraw = False

                if needs_redraw:
                    self._render(pygame)

                # Sleep until next check. During a blink we poll at
                # the configured FPS for a smooth transition; otherwise
                # we only need to wake a few times per second to check
                # for events and blink timing.
                if self._is_blinking:
                    self._clock.tick(self.config.fps)
                else:
                    self._clock.tick(10)

        except Exception as err:
            self.logger.error(f"Pygame thread error: {err}")
        finally:
            # Clear the framebuffer to black so the last frame doesn't
            # linger on the display after the app stops.
            if self._fb:
                try:
                    width = self.config.width
                    height = self.config.height
                    black = b"\x00\x00" * width * height  # RGB565 black
                    self._fb.seek(0)
                    self._fb.write(black)
                    self._fb.flush()
                except OSError as err:
                    self.logger.warning(f"Could not clear framebuffer: {err}")
                self._fb.close()
                self._fb = None
            pygame.quit()
            self.logger.info("Pygame thread stopped")

    def _load_expressions(self, pygame: Any) -> None:
        """Load all expression PNGs into a cache dict."""
        # Collect unique filenames from the expression map
        unique_files = set(EXPRESSION_MAP.values())

        for filename in unique_files:
            path = EXPRESSIONS_DIR / f"{filename}.png"
            if not path.exists():
                self.logger.warning(f"Expression image missing: {path}")
                continue

            image = pygame.image.load(str(path)).convert_alpha()
            self._expression_images[filename] = image
            self.logger.debug(f"Loaded expression: {filename}")

        self.logger.info(
            f"Loaded {len(self._expression_images)} expression images"
        )

    def _update_blink(self, pygame: Any) -> None:
        """Auto-blink: briefly show 'closed' eyes at random intervals."""
        now = pygame.time.get_ticks()

        # If currently blinking, check if blink duration has elapsed
        if self._is_blinking:
            if now >= self._blink_end_ms:
                self._is_blinking = False
                self._needs_redraw = True
            return

        # Only blink when showing a neutral expression
        with self._state_lock:
            expression = self.current_expression
        if expression != "neutral":
            self._last_blink_ms = now
            return

        # Check if it's time to blink
        if now - self._last_blink_ms >= self._blink_interval_ms:
            self._is_blinking = True
            self._blink_end_ms = now + 200  # 200ms blink
            self._last_blink_ms = now
            self._blink_interval_ms = random.randint(3000, 10000)
            self._needs_redraw = True

    def _render(self, pygame: Any) -> None:
        """Render the current expression to the screen/framebuffer."""
        # Clear screen to black
        self._screen.fill((0, 0, 0))

        # Determine which expression to display
        if self._is_blinking:
            display_expression = "closed"
        else:
            with self._state_lock:
                display_expression = self.current_expression

        # Look up the image filename for this expression
        filename = EXPRESSION_MAP.get(display_expression)
        if filename and filename in self._expression_images:
            surface = self._expression_images[filename]
            # Images are 800x440, placed at y=0 (leaves 40px at bottom)
            self._screen.blit(surface, (0, 0))

        # Output the frame
        if self._use_framebuffer:
            self._blit_to_framebuffer(pygame)
        else:
            pygame.display.flip()

    def _blit_to_framebuffer(self, pygame: Any) -> None:
        """Convert the pygame surface to RGB565 and write to /dev/fb0."""
        import numpy as np

        # surfarray gives (W, H, 3) — transpose to row-major (H, W, 3)
        arr = pygame.surfarray.array3d(self._screen).transpose(1, 0, 2)

        r = arr[:, :, 0].astype(np.uint16)
        g = arr[:, :, 1].astype(np.uint16)
        b = arr[:, :, 2].astype(np.uint16)

        rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
        data = rgb565.astype("<u2").tobytes()

        self._fb.seek(0)
        self._fb.write(data)
        self._fb.flush()

    def _publish_from_thread(
        self,
        loop: asyncio.AbstractEventLoop,
        event_type: str,
        data: Any,
    ) -> None:
        """Schedule an event bus publish from the pygame thread."""
        asyncio.run_coroutine_threadsafe(
            self.event_bus.publish(event_type, data), loop
        )

    async def _handle_set_expression(self, data: Any) -> None:
        """Handle gui.set_expression event."""
        expression = data.get("expression") if isinstance(data, dict) else None
        if not expression:
            self.logger.warning("gui.set_expression: no expression provided")
            return

        if expression not in EXPRESSION_MAP:
            self.logger.warning(f"Unknown expression: {expression}")
            return

        with self._state_lock:
            self.current_expression = expression
            self._needs_redraw = True
        self.logger.info(f"Expression set to: {expression}")

    async def shutdown(self) -> None:
        """Stop the pygame thread and clean up."""
        self._stop_event.set()

        # Unsubscribe from events
        self.event_bus.unsubscribe(
            "gui.set_expression", self._handle_set_expression
        )

        # Wait for the pygame thread to finish
        if self._gui_thread and self._gui_thread.is_alive():
            self._gui_thread.join(timeout=2.0)

        await super().shutdown()
