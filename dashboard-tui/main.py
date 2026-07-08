"""
Roberta dashboard — terminal UI.

Connects to the SSE stream exposed by Roberta's DashboardService and
renders live state: realtime mode, mic level, event rates, latest
transcripts, and recent errors.

Run::

    uv run --project dashboard-tui python dashboard-tui/main.py \\
        http://roberta.local:8765/events

It reconnects automatically if Roberta restarts.
"""

import asyncio
import json
import sys
import time
from collections import Counter, deque
from typing import Any

import httpx
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, RichLog, Static


# Events whose type we track but whose data we do not show verbatim.
_NOISY_EVENTS = frozenset({"audio.mic_level", "audio.mic_chunk"})


class StatusPanel(Static):
    """Top-left: connection and realtime state."""

    connection = reactive("connecting…")
    realtime = reactive("idle")
    expression = reactive("neutral")
    uptime_started: float | None = None

    def render(self) -> str:
        uptime = (
            _fmt_duration(time.time() - self.uptime_started)
            if self.uptime_started is not None
            else "—"
        )
        return (
            f"[b]Connection[/b]  {self.connection}\n"
            f"[b]Realtime[/b]    {self.realtime}\n"
            f"[b]Expression[/b]  {self.expression}\n"
            f"[b]Since[/b]       {uptime}"
        )


class MicMeter(Static):
    """Top-right: live mic level bar."""

    level = reactive(0.0)

    def render(self) -> str:
        width = 40
        filled = max(0, min(width, int(self.level * width)))
        bar = "█" * filled + "░" * (width - filled)
        return f"[b]Mic[/b]  {bar}  {self.level:4.2f}"


class EventRates(Static):
    """Middle-left: top event types by count."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._counts: Counter[str] = Counter()

    def bump(self, event_type: str) -> None:
        self._counts[event_type] += 1
        self.refresh()

    def render(self) -> str:
        top = self._counts.most_common(12)
        if not top:
            return "[b]Events[/b]\n(waiting…)"
        lines = [f"{count:>6}  {name}" for name, count in top]
        return "[b]Events (total)[/b]\n" + "\n".join(lines)


class Transcripts(Static):
    """Middle-right: last user + assistant turns."""

    user = reactive("—")
    model = reactive("—")

    def render(self) -> str:
        return (
            f"[b]You[/b]\n{self.user}\n\n"
            f"[b]Roberta[/b]\n{self.model}"
        )


class Dashboard(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #top { height: 6; }
    #mid { height: 16; }
    #log { height: 1fr; border: tall $panel; }
    StatusPanel, MicMeter, EventRates, Transcripts {
        border: tall $panel;
        padding: 0 1;
    }
    StatusPanel { width: 40; }
    MicMeter { width: 1fr; }
    EventRates { width: 40; }
    Transcripts { width: 1fr; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self._last_errors: deque[str] = deque(maxlen=50)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            StatusPanel(id="status"), MicMeter(id="mic"), id="top"
        )
        yield Horizontal(
            EventRates(id="rates"), Transcripts(id="tx"), id="mid"
        )
        yield RichLog(
            id="log",
            highlight=True,
            markup=True,
            wrap=True,
            max_lines=500,
        )
        yield Footer()

    async def on_mount(self) -> None:
        self.sub_title = self.url
        asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        """Stream events from the SSE endpoint; reconnect on failure."""
        status = self.query_one("#status", StatusPanel)
        log = self.query_one("#log", RichLog)
        backoff = 1.0

        while True:
            status.connection = "connecting…"
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("GET", self.url) as response:
                        response.raise_for_status()
                        status.connection = "[green]connected[/green]"
                        status.uptime_started = time.time()
                        backoff = 1.0

                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            try:
                                event = json.loads(line[6:])
                            except json.JSONDecodeError:
                                continue
                            self._apply(event)
            except (httpx.HTTPError, ConnectionError) as exc:
                status.connection = f"[red]disconnected[/red] ({exc})"
                log.write(f"[red]disconnected[/red]: {exc}")
            except Exception as exc:
                status.connection = f"[red]error[/red] ({exc})"
                log.write(f"[red]error[/red]: {exc}")

            status.uptime_started = None
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 15.0)

    def _apply(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        data = event.get("data")

        rates = self.query_one("#rates", EventRates)
        rates.bump(event_type)

        if event_type == "audio.mic_level" and isinstance(data, dict):
            level = data.get("level")
            if isinstance(level, int | float):
                self.query_one("#mic", MicMeter).level = float(level)

        elif event_type == "gui.set_expression" and isinstance(data, dict):
            expr = data.get("expression")
            if isinstance(expr, str):
                self.query_one("#status", StatusPanel).expression = expr

        elif event_type == "realtime.response_started":
            self.query_one("#status", StatusPanel).realtime = (
                "[yellow]speaking[/yellow]"
            )
        elif event_type == "realtime.response_completed":
            self.query_one("#status", StatusPanel).realtime = "idle"
        elif event_type == "realtime.interrupted":
            self.query_one("#status", StatusPanel).realtime = (
                "[magenta]interrupted[/magenta]"
            )
        elif event_type == "audio.start_recording":
            # Only nudge the state if we aren't currently speaking.
            status = self.query_one("#status", StatusPanel)
            if "speaking" not in status.realtime:
                status.realtime = "[cyan]listening[/cyan]"

        elif event_type == "realtime.user_transcript" and isinstance(
            data, dict
        ):
            text = data.get("text", "")
            if isinstance(text, str) and text:
                self.query_one("#tx", Transcripts).user = text
        elif event_type == "realtime.model_transcript" and isinstance(
            data, dict
        ):
            text = data.get("text", "")
            if isinstance(text, str) and text:
                self.query_one("#tx", Transcripts).model = text

        elif event_type == "realtime.error" and isinstance(data, dict):
            err = data.get("error", data)
            self.query_one("#log", RichLog).write(
                f"[red]realtime.error[/red]: {err}"
            )

        # Log non-noisy events for context.
        if event_type not in _NOISY_EVENTS:
            self.query_one("#log", RichLog).write(
                f"[dim]{_fmt_time(event.get('t'))}[/dim] "
                f"[b]{event_type}[/b] {_compact(data)}"
            )


def _fmt_time(ts: Any) -> str:
    if not isinstance(ts, int | float):
        return "--:--:--"
    lt = time.localtime(ts)
    return time.strftime("%H:%M:%S", lt)


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def _compact(data: Any) -> str:
    """Short, readable string for the log column."""
    if data is None or data == {}:
        return ""
    try:
        s = json.dumps(data, ensure_ascii=False)
    except TypeError:
        s = repr(data)
    return s if len(s) <= 120 else s[:117] + "…"


def main() -> None:
    url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "http://localhost:8765/events"
    )
    Dashboard(url).run()


if __name__ == "__main__":
    main()
