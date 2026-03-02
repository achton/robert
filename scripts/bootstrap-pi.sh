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

# --- Enable I2C and I2S for ReSpeaker HAT ---
echo ""
echo "--- Enabling I2C and I2S ---"
sudo sed -i 's/^#dtparam=i2c_arm=on/dtparam=i2c_arm=on/' /boot/firmware/config.txt
sudo sed -i 's/^#dtparam=i2s=on/dtparam=i2s=on/' /boot/firmware/config.txt

# --- ReSpeaker 2-Mic HAT v2.0 overlay ---
echo ""
echo "--- Installing ReSpeaker 2-Mic v2.0 overlay ---"
if [ ! -f /boot/firmware/overlays/respeaker-2mic-v2_0.dtbo ]; then
    cd /tmp
    rm -rf seeed-linux-dtoverlays
    git clone --depth 1 https://github.com/Seeed-Studio/seeed-linux-dtoverlays.git
    cd seeed-linux-dtoverlays
    make overlays/rpi/respeaker-2mic-v2_0-overlay.dtbo
    sudo cp overlays/rpi/respeaker-2mic-v2_0-overlay.dtbo /boot/firmware/overlays/respeaker-2mic-v2_0.dtbo
    rm -rf /tmp/seeed-linux-dtoverlays
    echo "Overlay installed."
else
    echo "Overlay already installed."
fi
if ! grep -q "dtoverlay=respeaker-2mic-v2_0" /boot/firmware/config.txt; then
    echo "dtoverlay=respeaker-2mic-v2_0" | sudo tee -a /boot/firmware/config.txt
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

# --- Echo cancellation config ---
echo ""
echo "--- Configuring echo cancellation ---"
mkdir -p "$HOME/.config/pipewire/pipewire.conf.d"
cat > "$HOME/.config/pipewire/pipewire.conf.d/echo-cancel.conf" << 'AECCONF'
# Echo cancellation using WebRTC AEC.
# Creates a virtual source (echo-cancelled mic) and a virtual sink
# (monitored speaker output) that PipeWire uses to subtract speaker
# audio from the microphone input.
context.modules = [
    {
        name = libpipewire-module-echo-cancel
        args = {
            audio.rate     = 16000
            audio.channels = 1
            source.props = {
                node.name        = "echo_cancel_source"
                node.description = "Echo-Cancelled Microphone"
            }
            sink.props = {
                node.name        = "echo_cancel_sink"
                node.description = "Echo-Cancel Speaker"
            }
            library.name   = "aec/libspa-aec-webrtc"
            capture.props = {
                node.name = "alsa_input.platform-soc_sound.stereo-fallback"
            }
            playback.props = {
                node.name = "alsa_output.usb-Jieli_Technology_UACDemoV1.0_4150344E37343209-00.analog-stereo"
            }
        }
    }
]
AECCONF
echo "Echo cancellation configured."

# --- App directory ---
echo ""
echo "--- Preparing app directory ---"
mkdir -p "$HOME/robot"

echo ""
echo "=== Bootstrap complete: $(date) ==="
echo "Next: run 'task pi:deploy' from your laptop to sync code."
