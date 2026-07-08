# Pi Audio Issues — Investigation Notes

Date: 2026-03-04, resolved 2026-04-13.

## Resolution (2026-04-13): wrong codec driver

**Short version:** our HAT is a **Keyestudio KS0314** clone, not a genuine
Seeed ReSpeaker 2-Mic v2.0. It uses a **Wolfson/Cirrus WM8960** codec (clearly
printed on the chip in the photos taken of the board), not the
**TLV320AIC3104** that the v2.0 overlay targets. The original bootstrap
installed `dtoverlay=respeaker-2mic-v2_0`, which binds the `tlv320aic3x`
kernel driver at I²C address 0x18 and then hammers register addresses that
do not exist on a WM8960. Every transfer NAKs with `-EREMOTEIO` (−121) or
`-EIO` (−5), which is exactly what we saw in `dmesg`, and ALSA returns 0
bytes / `Input/output error` to userspace. The mic hardware itself is fine.

The fix was a one-line overlay swap plus a mixer tweak:

```text
/boot/firmware/config.txt:
-  dtoverlay=respeaker-2mic-v2_0
+  dtoverlay=wm8960-soundcard
```

`wm8960-soundcard.dtbo` ships with mainline Raspberry Pi OS, so no custom
overlay build is needed. After reboot, `dmesg` shows a clean `wm8960 1-001a:`
probe (note the different I²C address — 0x1a vs 0x18), no errors, and
`arecord`/`pw-record` capture real audio.

Two HAT variants exist and are easy to confuse:

| HAT                              | Codec            | I²C addr | Correct overlay            |
|----------------------------------|------------------|----------|----------------------------|
| Seeed ReSpeaker 2-Mic **v1.0**   | WM8960           | 0x1a     | `wm8960-soundcard` *or* `respeaker-2mic-v1_0` |
| Seeed ReSpeaker 2-Mic **v2.0**   | TLV320AIC3104    | 0x18     | `respeaker-2mic-v2_0`      |
| Keyestudio KS0314 (our board)    | WM8960           | 0x1a     | `wm8960-soundcard`         |
| Generic eBay 2-Mic Pi HAT        | almost always WM8960 | 0x1a | `wm8960-soundcard`         |

### Mic mixer config

The stock `wm8960-soundcard` overlay comes up with the onboard mic path
effectively muted (the `Left/Right Input Mixer Boost Switch` controls
ADC routing — turning them off disconnects the mic entirely, not just
−20 dB of preamp as the name suggests). These amixer settings (now in
`scripts/bootstrap-pi.sh` and persisted with `alsactl store`) give a
sensible analog gain:

```shell
amixer -c 1 cset name="Left Input Boost Mixer LINPUT1 Volume" 2   # +20 dB
amixer -c 1 cset name="Right Input Boost Mixer RINPUT1 Volume" 2
amixer -c 1 cset name="Left Input Mixer Boost Switch" on          # ADC route
amixer -c 1 cset name="Right Input Mixer Boost Switch" on
amixer -c 1 cset name="Left Boost Mixer LINPUT1 Switch" on        # route LINPUT1
amixer -c 1 cset name="Right Boost Mixer RINPUT1 Switch" on
amixer -c 1 cset name="Capture Volume" 40                         # ~+11 dB PGA
amixer -c 1 cset name="ADC PCM Capture Volume" 195                # 0 dB ADC
sudo alsactl store
```

Total ≈ +31 dB analog gain. The HAT mics still pick up a fair bit of
electrical noise from the Pi (USB, camera ribbon, display), so the GUI
mic indicator is gated at `mic_level > 0.12` to avoid flickering on
incidental room activity.

### Status of the three original problems, after the fix

| Problem | Status | Notes |
|---------|--------|-------|
| Problem 1 — FIFO injection blocked | Fixed earlier | `wait_drain` + partial-buffer discard in `audio_service.py` |
| Problem 2 — PipeWire AEC breaks PortAudio output | Still present | AEC config now shipped as `.disabled` by bootstrap; retest once we revisit AEC |
| Problem 3 — "Mic hardware failure" | **Resolved** — never a hardware failure; wrong driver |

### References

- Keyestudio KS0314 wiki: <https://docs.keyestudio.com/projects/KS0314/en/latest/docs/KS0314.html>
- Seeed ReSpeaker 2-Mic HAT wiki: <https://wiki.seeedstudio.com/ReSpeaker_2_Mics_Pi_HAT_Raspberry/>
- Seeed DT overlays (for `respeaker-2mic-v1_0` if we want Seeed's mixer defaults baked in): <https://github.com/Seeed-Studio/seeed-linux-dtoverlays>
- Same symptom reported for a KS0314: <https://github.com/balenalabs/balena-sound/issues/166>

---

## Original investigation notes

The sections below are preserved as-is for historical context and because
Problem 2 (PipeWire AEC) is still open.

## Environment

- Raspberry Pi 4 Model B, Raspberry Pi OS Lite (Debian 13 Trixie, 64-bit)
- PipeWire 1.4.2 with pipewire-pulse, WirePlumber
- sounddevice (PortAudio) for audio I/O in the robot app
- Hardware: ReSpeaker 2-Mic HAT v2.0 (TLV320AIC3104), USB speaker (UACDemoV1.0)

## Problem 1: FIFO text injection gets no response

**Symptom:** Writing to `/tmp/roberta.fifo` successfully injects text into the
Gemini session (`send_client_content` succeeds, "Injected: ..." logged), but
the model never responds — no audio, no transcript, nothing.

**Root cause:** The response loop in RealtimeService was blocked. After the
greeting response completes (`turn_complete`), the handler calls
`audio.wait_drain` which polls until the speaker queue and playback buffer are
empty. But the output callback was never firing (see Problem 2), so the queue
never drained. The response loop was stuck forever at `wait_drain` and never
called `session.receive()` again.

The FIFO injection runs in a separate asyncio task and calls
`send_client_content` directly — this succeeds because asyncio can still
schedule that task. But the model's response is never read because the
response loop is blocked.

**Evidence from logs:**
```
15:35:59 Recording enabled       (mic started at connect time)
15:36:01 Recording disabled       (mic paused when model speaks greeting)
15:36:09 Model: Hej! Jeg er...   (greeting transcript — turn_complete)
                                   [wait_drain starts here — never completes]
                                   [no "Recording enabled" ever appears]
15:37:43 Injected: Hvad er...    (FIFO text sent, but no response follows)
```

## Problem 2: PortAudio output callbacks never fire

**Symptom:** `sd.OutputStream` opens without error and reports `active: True`,
but the callback function is never called. Tested with timeouts up to 35
seconds — zero callbacks. This affects both callback mode and blocking write
mode (`stream.write()`).

**Root cause:** The PipeWire echo-cancel module breaks PortAudio's ALSA
backend entirely. When `echo-cancel.conf` is loaded, PortAudio output
callbacks never fire. Disabling the echo-cancel config file immediately fixes
output — 75 callbacks in 3 seconds, audio plays through USB speaker.

**Echo-cancel config** (was at `~/.config/pipewire/pipewire.conf.d/echo-cancel.conf`):
```
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
```

**Current status:** Renamed to `echo-cancel.conf.disabled`. Output works.

**Possibly related upstream issues:**
- [PortAudio #656](https://github.com/PortAudio/portaudio/issues/656) — 25+
  second callback delay on RPi4 (different symptom, same ecosystem)
- [sounddevice #609](https://github.com/spatialaudio/python-sounddevice/issues/609) —
  no PipeWire devices listed
- PortAudio has no native PipeWire host API; it relies on pipewire-alsa

## Problem 3: Microphone (seeed2mic) hardware failure

**Symptom:** No audio can be captured from the ReSpeaker 2-Mic HAT through
any method — sounddevice, SDL2 AudioDevice, arecord (direct ALSA), pw-record.
All return 0 bytes. This persists even with PipeWire completely stopped.

**Root cause:** The TLV320AIC3104 codec on the HAT has I2C communication
failures. The kernel logs show:

```
tlv320aic3x 1-0018: ASoC: error at soc_component_read_no_lock for register: [0x00000013] -16
bcm2835-i2s fe203000.i2s: I2S SYNC error!
tlv320aic3x 1-0018: Unable to sync registers 0x2-0x3. -5
tlv320aic3x 1-0018: ASoC: error at soc_component_read_no_lock for register: [0x00000005] -121
```

Error codes: -16 = EBUSY, -121 = EREMOTEIO (I2C failure), -5 = EIO.

The ALSA card and mixer controls appear normally (the driver loaded from the
device tree overlay), but the actual codec chip cannot communicate over I2C.
The `amixer` controls show plausible values because some are cached or have
defaults.

**Possible causes to investigate:**
- Loose HAT connection (reseat the GPIO header)
- Power issue (the HAT draws from the Pi's 3.3V rail)
- Damaged I2C bus or codec chip
- Kernel driver regression in Debian 13 / kernel 6.x
- Device tree overlay misconfiguration

**Current status:** Unresolved. Microphone input does not work on the Pi.

## Problem 4: PipeWire output routing (secondary)

**Observed during investigation:** Even without the echo-cancel module, some
audio paths through PipeWire don't work:

- `aplay -D default` (PipeWire ALSA) — no sound from USB speaker
- `pw-play --raw` — hangs
- `pygame.mixer` (SDL2 PulseAudio backend) — no sound from USB speaker
- `aplay -D plughw:4` (direct ALSA) — works, but only when PipeWire doesn't
  hold the device

However, sounddevice output through PipeWire DOES work once echo-cancel is
disabled. The other paths may have been tested while PipeWire was in a bad
state (zombie streams from hung test scripts). This should be re-tested in a
clean state.

**Key finding:** The USB speaker only supports 48000 Hz, stereo, S16_LE.
PipeWire handles the resampling from 24000 Hz mono transparently when the
echo-cancel module is not loaded.

## What works now

| Component | Status | Notes |
|-----------|--------|-------|
| Audio output (USB speaker) | Working | Via sounddevice → PipeWire (echo-cancel disabled) |
| FIFO text injection | Working | wait_drain completes, response loop processes replies |
| Greeting prompt | Working | Model speaks, audio plays, lifecycle completes |
| Audio input (seeed2mic) | Broken | Hardware I2C failure, needs physical investigation |
| Echo cancellation | Disabled | Breaks PortAudio entirely, also requires working mic |

## Files changed

- `robot/src/robot/audio_service.py` — partial-buffer fix in output callback
  (discard trailing bytes when queue is empty so wait_drain can complete)
- `~/.config/pipewire/pipewire.conf.d/echo-cancel.conf` → renamed to
  `.disabled` on the Pi

## Next steps

1. **Investigate seeed2mic hardware** — reseat HAT, check I2C with
   `i2cdetect`, try on a different Pi, test with a fresh OS image
2. **Re-enable echo cancellation** — once mic works, investigate why the
   echo-cancel module breaks PortAudio. May need a different PipeWire config
   or a newer PipeWire version. Alternative: implement echo cancellation in
   the app instead of PipeWire
3. **Consider alternative mic** — USB microphone would bypass the I2C issue
   entirely and might work better with PipeWire
