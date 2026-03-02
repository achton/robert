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
  - Virtual sink: `echo_cancel_sink`
  - Config: `~/.config/pipewire/pipewire.conf.d/echo-cancel.conf`
- All hardware verified: display, USB speaker, HQ camera (imx477), ReSpeaker HAT
