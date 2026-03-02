"""Tests for GUIService."""

from unittest.mock import patch

from robot.config import GUIConfig
from robot.event_bus import EventBus
from robot.gui_service import EXPRESSION_MAP, GUIService


class TestGUIServiceInit:
    """Test GUIService instantiation."""

    def test_can_instantiate(self):
        """GUIService can be created with default config."""
        bus = EventBus()
        service = GUIService(bus)
        assert service.running is False
        assert service.current_expression == "neutral"

    def test_custom_config(self):
        """GUIService respects a custom GUIConfig."""
        bus = EventBus()
        config = GUIConfig(fps=30, default_expression="happy")
        service = GUIService(bus, config)
        assert service.config.fps == 30
        assert service.current_expression == "happy"


class TestExpressionMap:
    """Test the expression name → filename mapping."""

    def test_neutral_maps_to_smiling(self):
        assert EXPRESSION_MAP["neutral"] == "face-smiling"

    def test_happy_maps_to_happy(self):
        assert EXPRESSION_MAP["happy"] == "face-happy"

    def test_angry_maps_to_angry(self):
        assert EXPRESSION_MAP["angry"] == "face-angry"

    def test_closed_maps_to_closed(self):
        assert EXPRESSION_MAP["closed"] == "face-closed"

    def test_all_entries_are_strings(self):
        for key, value in EXPRESSION_MAP.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


class TestGUIServiceInitialize:
    """Test GUIService.initialize() behaviour."""

    async def test_disables_when_no_display(self):
        """Service sets running=False when no display is detected."""
        bus = EventBus()
        service = GUIService(bus)

        with patch("robot.gui_service.detect_display", return_value=False):
            await service.initialize()

        assert service.running is False

    async def test_enables_on_desktop(self):
        """Service sets running=True when desktop display is available."""
        bus = EventBus()
        service = GUIService(bus)

        with (
            patch("robot.gui_service.detect_display", return_value=True),
            patch(
                "robot.gui_service.is_raspberry_pi",
                return_value=False,
            ),
            patch.dict("os.environ", {"DISPLAY": ":0"}, clear=False),
        ):
            await service.initialize()

        assert service.running is True

    async def test_pi_headless_sets_dummy_driver(self):
        """On Pi without desktop, SDL_VIDEODRIVER is set to 'dummy'."""
        bus = EventBus()
        service = GUIService(bus)

        with (
            patch("robot.gui_service.detect_display", return_value=True),
            patch(
                "robot.gui_service.is_raspberry_pi",
                return_value=True,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            await service.initialize()

        assert service.running is True
        assert service._use_framebuffer is True


class TestHandleSetExpression:
    """Test the expression change event handler."""

    async def test_sets_valid_expression(self):
        bus = EventBus()
        service = GUIService(bus)
        await service._handle_set_expression({"expression": "happy"})
        assert service.current_expression == "happy"

    async def test_ignores_unknown_expression(self):
        bus = EventBus()
        service = GUIService(bus)
        await service._handle_set_expression({"expression": "confused"})
        # Should stay at default
        assert service.current_expression == "neutral"

    async def test_ignores_missing_expression(self):
        bus = EventBus()
        service = GUIService(bus)
        await service._handle_set_expression({})
        assert service.current_expression == "neutral"

    async def test_marks_redraw_needed(self):
        """Setting an expression should flag a redraw."""
        bus = EventBus()
        service = GUIService(bus)
        service._needs_redraw = False
        await service._handle_set_expression({"expression": "angry"})
        assert service._needs_redraw is True

    async def test_no_redraw_on_invalid(self):
        """Invalid expression should not flag a redraw."""
        bus = EventBus()
        service = GUIService(bus)
        service._needs_redraw = False
        await service._handle_set_expression({"expression": "confused"})
        assert service._needs_redraw is False


class TestSetExpressionViaEventBus:
    """Test gui.set_expression through the full event bus flow."""

    async def test_event_changes_expression(self):
        """Publishing gui.set_expression on the bus updates the service."""
        bus = EventBus()
        service = GUIService(bus)

        # Subscribe the handler (normally done in run())
        bus.subscribe("gui.set_expression", service._handle_set_expression)

        await bus.publish("gui.set_expression", {"expression": "happy"})
        assert service.current_expression == "happy"

    async def test_event_ignores_unknown(self):
        """Publishing an unknown expression via the bus is a no-op."""
        bus = EventBus()
        service = GUIService(bus)
        bus.subscribe("gui.set_expression", service._handle_set_expression)

        await bus.publish("gui.set_expression", {"expression": "confused"})
        assert service.current_expression == "neutral"
