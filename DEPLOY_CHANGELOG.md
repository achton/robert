# Deployment Changelog

Changes made to the Raspberry Pi (roberta.local).

## 2026-02-27

- Fresh OS install: Raspberry Pi OS Lite (Trixie/Debian 13, 64-bit)
- Hostname set to `roberta` (was `raspberrypi`)
- SSH key authorized for user `pi` (id_rsa.pub)
- WiFi configured for office and home networks
- OS packages updated (`apt-get upgrade`)
- Bootstrap script run: installed git, vim, libportaudio2, SDL2 libs, uv 0.10.7
- Generated `en_DK.UTF-8` locale
- Enabled I2C and I2S, built ReSpeaker 2-Mic v2.0 overlay
- Installed PipeWire 1.4.2, pipewire-pulse, wireplumber, pipewire-alsa
- Enabled user lingering (`loginctl enable-linger pi`)
- Configured WebRTC echo cancellation (AEC) via PipeWire module
  - Virtual source: `echo_cancel_source`
  - Uses `monitor.mode` (captures reference from default output)
  - Config: `~/.config/pipewire/pipewire.conf.d/echo-cancel.conf`
- All hardware verified: display, USB speaker, HQ camera (imx477), ReSpeaker HAT

## 2026-03-03

- Installed systemd user service (`~/.config/systemd/user/robot.service`)
- Service enabled: auto-starts on boot via `default.target`
- Added `pi` user to `systemd-journal` group (for journal access)
- Enabled SSH password authentication (`/etc/ssh/sshd_config.d/50-cloud-init.conf`)
- Reset `pi` user password to default Raspberry Pi OS password

## 2026-03-04

- Fixed PipeWire AEC config (`~/.config/pipewire/pipewire.conf.d/echo-cancel.conf`)
  - Changed `capture.props` from `node.name` to `node.target` (was naming the
    stream instead of targeting the physical mic)
  - Added `monitor.mode = true` — captures reference signal from default
    output's monitor port, no virtual sink needed
  - Added `node.passive = true` to capture props
  - Removed `sink.props` and `playback.props` (not needed with monitor mode)
- Set `echo_cancel_source` as default PipeWire audio source via `wpctl set-default`
  (persists in WirePlumber state database across restarts)
- Added `GEMINI_API_KEY` to `/home/pi/robot/.env` (mode 600)
- Updated `robot.service` with `EnvironmentFile=/home/pi/robot/.env` to load API keys

## 2026-07-03

- Raised USB speaker output volume from 40% to 75% via
  `wpctl set-volume @DEFAULT_AUDIO_SINK@ 0.75` (-17.4 dB to -7.85 dB)
  - Persisted by WirePlumber in `~/.local/state/wireplumber/default-routes`
- Deployed RealtimeService reconnect-with-backoff (`task pi:deploy`).
  A dropped or failed Gemini connection is now retried instead of stopping
  the service, fixing the stuck-on-logo failure after a cold boot with a
  stale clock (no RTC). Verified: service reconnects and greets on restart.
- Deployed client-side Silero VAD speech gate (`task pi:deploy`), adding the
  `onnxruntime` dependency (1.27.0, aarch64 wheel) and the ~2.3 MB
  `silero_vad.onnx` model. Only detected speech is forwarded to Gemini.
  Verified live: gate loads (ONNX), noise no longer produces phantom
  transcripts, real Danish speech still passes through cleanly.
- Deployed VAD fix "1b" (`task pi:deploy`). The mic-audio input path was
  wedging after a long gated-silence period: per-utterance `audio_stream_end`
  plus minutes with zero audio left Gemini ignoring realtime audio (text
  still worked). Fix: stopped sending `audio_stream_end`; the server VAD now
  ends turns from the trailing silence forwarded during the gate hangover
  (`min_silence_ms` 400 to 700, above the 500 ms server window). Added gate
  open/close debug logging. Verified: gate cycles and responds on restart.
- Reworked Roberta's persona. Her system prompt now lives in editable markdown
  at `robot/assets/prompt.md` (loaded at startup, with an inline fallback). The
  voice is dry, witty, direct and sarcastic - drawing on Reload's no-bullshit
  brand voice and on Monty Python, Blackadder and Marvin (Hitchhiker's Guide),
  with occasional deadpan Danish Gen Alpha slang (sus, bruh, delulu, cringe,
  flex, vibe, ...). "Reload" is pronounced in English. Verified live.
- Tuned VAD sensitivity after speech was being clipped: `threshold` 0.5 to 0.3
  (more sensitive), `speech_pad_ms` 200 to 300 (preserve word onsets).
- Switched the Gemini model to `gemini-2.5-flash-native-audio-latest` (rolling
  alias to the newest 2.5 native-audio) from the 12-2025 preview snapshot.
- Set the voice to `Aoede` (from `Kore`).
- Added an NTP-sync startup gate to `robot.service` (`ExecStartPre` waits up to
  90s for `timedatectl` to report the clock synced, else fails and lets
  `Restart=on-failure` retry). The app no longer starts with the Pi's stale
  cold-boot clock, which had caused the stuck-on-logo TLS error and a
  mid-session wedge when NTP later stepped the clock forward. Installed via
  `task pi:install-service`.
- Roberta now answers time and date in local Danish time. The OS local time
  (timezone Europe/Copenhagen, DST-aware) is read at each connection and
  injected into the session's system instruction, plus a prompt rule to use
  local time and never UTC. Verified: she reported 13:02 CEST, not 11:02 UTC.

## 2026-07-08

- Enabled `unattended-upgrades` for the OS: daily, Debian stable + security
  origins, auto-reboot disabled. (App Python deps are git-managed via uv.lock,
  so they are deliberately NOT auto-upgraded on the Pi.)
- One-time full OS upgrade: `apt full-upgrade` (~141 packages, incl. kernel
  6.12 to 6.18 and WiFi/BT firmware), `rpi-eeprom-update -a` (bootloader
  2025-11 to 2026-05), autoremove. Rebooted to apply.
- Upgraded app dependencies (`uv lock --upgrade`; google-genai held at 1.x
  pending the 2.x migration) and deployed to the Pi.
- Post-reboot verification: the NTP-sync gate held startup ~44s until the clock
  synced, then the bot connected, greeted, and answered a voice query in local
  Danish time. Kernel 6.18.34, EEPROM up to date.
