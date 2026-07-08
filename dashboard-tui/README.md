# Roberta Dashboard TUI

Terminal dashboard that streams live events from a running Roberta.

## How it works

Roberta's `DashboardService` (in `robot/src/robot/dashboard/`) opens an
HTTP endpoint on port `8765` and pushes every event-bus event as a
Server-Sent Events stream. This TUI subscribes to that stream and
renders a live view.

It is a **separate uv project** so Textual/httpx never land on the Pi.

## Run it

From the repo root:

```sh
# Local (Roberta running on this laptop)
uv run --project dashboard-tui python dashboard-tui/main.py

# Remote (Roberta running on the Pi)
uv run --project dashboard-tui python dashboard-tui/main.py \
    http://roberta.local:8765/events
```

Or via Taskfile:

```sh
task dashboard          # localhost
task dashboard -- pi    # roberta.local
```

Press `q` to quit. It auto-reconnects if Roberta restarts.

## What you see

- **Status** — connection state, current realtime mode
  (idle/listening/speaking/interrupted), expression, session uptime
- **Mic** — live level meter updated at ~10 Hz
- **Events** — running count per event type (catches runaway loops fast)
- **Transcripts** — last user utterance and last assistant reply
- **Log** — scrolling feed of every non-noisy event plus errors

## Disabling the server

Set `ROBOTA_DASHBOARD=0` in `robot/.env` to turn the SSE endpoint off
entirely. Or override host/port with `ROBOTA_DASHBOARD_HOST` and
`ROBOTA_DASHBOARD_PORT`.
