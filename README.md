# Roberta

An event-driven, multi-modal office robot running on a Raspberry Pi 4 with a
touchscreen, camera, microphone, and speaker. Roberta provides conversational
voice interaction, animated facial expressions, and face recognition.

## Hardware

| Component   | Model                          |
|-------------|--------------------------------|
| Computer    | Raspberry Pi 4 Model B (4 GB)  |
| Display     | Official 7" DSI touchscreen    |
| Camera      | Pi HQ Camera (IMX477)          |
| Microphone  | ReSpeaker 2-Mic HAT (WM8960)  |
| Speaker     | USB speaker                    |

## Quick start

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/achton/robert.git
cd robert/robot
uv sync
cd ..
task run
```

## Development

[go-task](https://taskfile.dev/) runs all project commands. Key tasks:

```
task check    # format + lint + typecheck + tests
task fix      # auto-fix formatting and lint issues
task run      # run locally (laptop or Pi)
```

## Connecting to the Pi

The Pi is reachable via mDNS as `roberta.local`. The `task pi:*` commands
expect an SSH host alias called `roberta`. Add it to your SSH config:

```
# ~/.ssh/config
Host roberta
    HostName roberta.local
    User pi
```

Then copy your SSH public key to the Pi and verify:

```bash
ssh-copy-id roberta   # enter the default Raspberry Pi OS password when prompted
ssh roberta           # should connect without a password
```

## Raspberry Pi deployment

```
task pi:bootstrap       # first-time setup (packages, audio, overlays, service)
task pi:deploy          # sync code, install deps, restart service
task pi:install-service # (re)install the systemd service
```

The app runs as a systemd user service that starts automatically on boot.
Useful commands after deployment:

```
task pi:status    # check if the service is running
task pi:logs      # follow live logs
task pi:restart   # restart the service
task pi:stop      # stop the service
task pi:run       # run interactively (bypasses systemd)
```

## Project status

See [docs/PLAN.md](docs/PLAN.md) for the implementation roadmap and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system design.
