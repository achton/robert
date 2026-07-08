#!/usr/bin/env bash
# bootstrap-pi.sh — Idempotent first-time setup for the robot on Raspberry Pi.
#
# Run this on a fresh Raspberry Pi OS Lite (Trixie) install to get the Pi
# ready for deployment. Safe to re-run at any time.
#
# Usage (from laptop):
#   scp scripts/bootstrap-pi.sh roberta:/tmp/ && ssh roberta '/tmp/bootstrap-pi.sh'
#
# Or via Taskfile:
#   task pi:bootstrap

set -euo pipefail

echo "=== Pi Bootstrap ==="
echo "Started: $(date)"

# --- System packages ---
echo ""
echo "--- Installing system packages ---"
sudo apt-get update -q
# Audio: libportaudio2 for sounddevice
# SDL2: libsdl2-* for pygame-ce
sudo apt-get install -q -y --no-install-recommends \
    git \
    vim \
    libportaudio2 \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0

# --- uv (Python package manager) ---
echo ""
echo "--- Installing uv ---"
if command -v uv &>/dev/null; then
    echo "uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "uv installed: $($HOME/.local/bin/uv --version)"
fi

# --- Locale fix (suppress debconf warnings) ---
echo ""
echo "--- Configuring locale ---"
if ! locale -a 2>/dev/null | grep -q "en_DK.utf8"; then
    sudo sed -i 's/^# *en_DK.UTF-8/en_DK.UTF-8/' /etc/locale.gen
    sudo locale-gen
fi

# --- Enable I2C and I2S for the ReSpeaker-style HAT ---
echo ""
echo "--- Enabling I2C and I2S ---"
sudo sed -i 's/^#dtparam=i2c_arm=on/dtparam=i2c_arm=on/' /boot/firmware/config.txt
sudo sed -i 's/^#dtparam=i2s=on/dtparam=i2s=on/' /boot/firmware/config.txt

# --- ReSpeaker 2-Mic HAT (WM8960) overlay ---
#
# Our board is a Keyestudio KS0314 clone of the ReSpeaker 2-Mic Pi HAT. It
# uses a Wolfson/Cirrus WM8960 codec (NOT the TLV320AIC3104 on the original
# Seeed v2.0 — see photos of "WM8960" silkscreen on the codec IC).
#
# The mainline Raspberry Pi OS ships a suitable overlay
# (`wm8960-soundcard.dtbo`) by default, so no custom overlay build needed.
# Attempting to use `respeaker-2mic-v2_0` here loads the tlv320aic3x driver
# and produces a storm of I2C EREMOTEIO errors because the register maps
# differ — see docs/pi-audio-issues.md for the full investigation.
echo ""
echo "--- Enabling WM8960 sound card overlay ---"
# Remove any previous (wrong) overlay line, then ensure the right one exists.
sudo sed -i '/^dtoverlay=respeaker-2mic-v2_0/d' /boot/firmware/config.txt
sudo sed -i '/^dtoverlay=respeaker-2mic-v1_0/d' /boot/firmware/config.txt
if ! grep -q "^dtoverlay=wm8960-soundcard" /boot/firmware/config.txt; then
    echo "dtoverlay=wm8960-soundcard" | sudo tee -a /boot/firmware/config.txt
fi

# --- PipeWire audio server ---
echo ""
echo "--- Installing PipeWire ---"
sudo apt-get install -q -y --no-install-recommends \
    pipewire \
    pipewire-audio \
    pipewire-alsa \
    pipewire-pulse \
    wireplumber

# Enable user lingering so PipeWire runs without a login session.
sudo loginctl enable-linger pi

# --- WM8960 mic mixer defaults ---
#
# The stock `wm8960-soundcard` overlay comes up with the onboard mic path
# effectively muted. These amixer settings route the onboard mics through
# the boost mixer and ADC at a sensible gain (~+31 dB total) — loud enough
# for normal speech at ~1 m without saturating, and matched to the GUI
# `mic_level > 0.12` indicator threshold.
#
# Note on naming: `Input Mixer Boost Switch` is the ADC routing enable,
# NOT just an extra +20 dB preamp. Turning it OFF disconnects the input
# stage entirely, which is why we keep it `on`.
#
# Tuned empirically; adjust with `alsamixer -c 1` if needed and persist
# with `sudo alsactl store`.
echo ""
echo "--- Configuring WM8960 mic mixer ---"
WM8960_CARD=$(aplay -l 2>/dev/null | awk -F'[ :]' '/wm8960/ {print $2; exit}')
if [ -n "$WM8960_CARD" ]; then
    amixer -c "$WM8960_CARD" cset name="Left Input Boost Mixer LINPUT1 Volume" 2 >/dev/null
    amixer -c "$WM8960_CARD" cset name="Right Input Boost Mixer RINPUT1 Volume" 2 >/dev/null
    amixer -c "$WM8960_CARD" cset name="Left Input Mixer Boost Switch" on >/dev/null
    amixer -c "$WM8960_CARD" cset name="Right Input Mixer Boost Switch" on >/dev/null
    amixer -c "$WM8960_CARD" cset name="Left Boost Mixer LINPUT1 Switch" on >/dev/null
    amixer -c "$WM8960_CARD" cset name="Right Boost Mixer RINPUT1 Switch" on >/dev/null
    amixer -c "$WM8960_CARD" cset name="Capture Volume" 40 >/dev/null
    amixer -c "$WM8960_CARD" cset name="ADC PCM Capture Volume" 195 >/dev/null
    # Persist so alsa-restore.service brings this back on every boot.
    sudo alsactl store
    echo "WM8960 mixer configured and stored."
else
    echo "WM8960 card not detected yet — reboot and re-run bootstrap to apply mixer."
fi

# --- Echo cancellation config (disabled by default) ---
#
# PipeWire's WebRTC echo-cancel module breaks PortAudio output callbacks
# entirely on the Pi 4 (see docs/pi-audio-issues.md, Problem 2). Until that
# is root-caused, we drop the config at `.disabled` so PipeWire does NOT
# auto-load it. Rename to `.conf` to re-enable for experimentation.
echo ""
echo "--- Writing echo cancellation config (NOT enabled by default) ---"
mkdir -p "$HOME/.config/pipewire/pipewire.conf.d"
cat > "$HOME/.config/pipewire/pipewire.conf.d/echo-cancel.conf.disabled" << 'AECCONF'
# Echo cancellation using WebRTC AEC.
# Creates a virtual source (echo-cancelled mic) that subtracts speaker
# audio from the microphone input.
#
# monitor.mode captures the reference signal from the default output's
# monitor ports, so AEC works regardless of which sink apps play to.
#
# The capture node.target is hardware-specific. If you swap the mic,
# update it to match. Find node names with:
#   pw-cli list-objects | grep node.name
context.modules = [
    {
        name = libpipewire-module-echo-cancel
        args = {
            audio.rate     = 16000
            audio.channels = 1
            library.name   = "aec/libspa-aec-webrtc"
            monitor.mode   = true
            capture.props = {
                node.target  = "alsa_input.platform-soc_sound.stereo-fallback"
                node.passive = true
            }
            source.props = {
                node.name        = "echo_cancel_source"
                node.description = "Echo-Cancelled Microphone"
            }
        }
    }
]
AECCONF
echo "Echo cancellation config available (disabled — rename to .conf to enable)."

# --- App directory ---
echo ""
echo "--- Preparing app directory ---"
mkdir -p "$HOME/robot"

# --- Systemd user service directory ---
echo ""
echo "--- Preparing systemd user directory ---"
mkdir -p "$HOME/.config/systemd/user"

echo ""
echo "=== Bootstrap complete: $(date) ==="
echo "Next: run 'task pi:deploy' from your laptop to sync code."
