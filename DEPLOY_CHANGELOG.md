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
