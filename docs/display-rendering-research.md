# Display Rendering on Raspberry Pi OS Lite

Research notes from getting pygame-ce to render on the 7" DSI touchscreen
display on Raspberry Pi 4 running Pi OS Lite (Debian Trixie, 64-bit) without
a desktop environment (no X11, no Wayland).

## Hardware

- Raspberry Pi 4 Model B Rev 1.4
- Official 7" DSI touchscreen (800x480, connected via ribbon cable)
- Pi OS Lite — Debian 13 (Trixie), 64-bit, kernel 6.x
- pygame-ce 2.5.6, SDL 2.32.10

## DRM Device Layout on Pi 4

The Pi 4 GPU exposes **two DRM cards**:

| Device           | Driver | Purpose                        |
|------------------|--------|--------------------------------|
| `/dev/dri/card0` | v3d    | 3D accelerator, no connectors  |
| `/dev/dri/card1` | vc4    | Display controller (DSI, HDMI) |

The DSI display appears as connector `DSI-1` on `card1`. You can verify with:

```bash
# Show connected displays
for conn in /sys/class/drm/card*-*/; do
    echo "$(basename $conn): $(cat $conn/status)"
done

# Expected output:
# card1-DSI-1: connected
# card1-HDMI-A-1: disconnected
# card1-HDMI-A-2: disconnected
```

## Framebuffer

The kernel exposes a legacy framebuffer device backed by the DRM driver:

```
/dev/fb0 — vc4drmfb — 800x480 — 16bpp (RGB565) — stride 1600
```

Check with:

```bash
cat /sys/class/graphics/fb0/virtual_size   # 800,480
cat /sys/class/graphics/fb0/bits_per_pixel  # 16
cat /sys/class/graphics/fb0/stride          # 1600
cat /sys/class/graphics/fb0/name            # vc4drmfb
```

Writing directly to `/dev/fb0` always works and immediately updates the
display — no permissions issues, no DRM master needed.

## SDL2 Video Drivers Tested

### KMSDRM — does NOT work reliably

SDL2's KMSDRM driver talks directly to the DRM/KMS subsystem. It creates
GPU-backed GBM buffers, does hardware-accelerated OpenGL ES rendering, and
uses atomic modesetting to page-flip frames onto the display.

**Requires `SDL_KMSDRM_DEVICE_INDEX=1`** on Pi 4 (DSI is on card1, not card0).

**The problem: DRM master access.** KMSDRM needs exclusive "DRM master"
control of the display hardware. On Pi OS Lite this is blocked by:

- **seatd** — if running, it holds DRM master. Even `sudo` can't get it.
  Must be stopped/disabled.
- **getty@tty1** — the login console on the display. Must be stopped.
- **systemd-logind** — manages VT/seat assignments. SSH sessions don't get
  a seat, so processes started via SSH cannot acquire DRM master.

Even with seatd and getty disabled, **KMSDRM from SSH silently fails** — it
initialises, reports success, but renders to an invisible DRM plane. The
console framebuffer remains on top. A systemd service with `TTYPath=/dev/tty1`
was also tried but did not produce visible output.

`kmscube` (a raw DRM test tool) shows the same behaviour: it initialises EGL
fine but fails with "Permission denied" when seatd is running, and produces
no visible output when run from SSH even after seatd is disabled.

**Verdict:** KMSDRM is the "correct" modern approach but is impractical on
Pi OS Lite without a display manager or compositor to handle DRM master.

### fbcon — not available

The legacy SDL1-era framebuffer console driver. Removed from SDL2 in recent
versions. `pygame.error: fbcon not available`.

### directfb — not available

Another legacy driver. Not compiled into the system SDL2 on Trixie.
`pygame.error: directfb not available`.

### rpi (dispmanx) — removed

The old Raspberry Pi-specific SDL driver using the Broadcom dispmanx API.
Removed from SDL2 >= 2.0.20 when Pi OS moved to KMS.

### dummy + /dev/fb0 blit — WORKS

Use SDL's `dummy` video driver for off-screen rendering, then copy the
frame to `/dev/fb0` as RGB565. This is the approach that reliably works.

**Environment:**

```python
os.environ["SDL_VIDEODRIVER"] = "dummy"
```

**Conversion (numpy, ~0.7 ms on Pi 4):**

```python
import numpy as np
import pygame

def surface_to_rgb565(surface: pygame.Surface) -> bytes:
    """Convert a pygame Surface to RGB565 bytes for /dev/fb0."""
    # surfarray gives (W, H, 3) — transpose to row-major (H, W, 3)
    arr = pygame.surfarray.array3d(surface).transpose(1, 0, 2)

    r = arr[:, :, 0].astype(np.uint16)
    g = arr[:, :, 1].astype(np.uint16)
    b = arr[:, :, 2].astype(np.uint16)

    rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    return rgb565.astype("<u2").tobytes()
```

**Writing to the framebuffer:**

```python
fb_data = surface_to_rgb565(screen)
with open("/dev/fb0", "wb") as fb:
    fb.write(fb_data)
```

For continuous rendering, keep the file handle open and seek to 0 before
each write:

```python
fb = open("/dev/fb0", "wb")
# In render loop:
fb.seek(0)
fb.write(surface_to_rgb565(screen))
fb.flush()
```

**Performance:** 0.7 ms per frame write on Pi 4 — easily supports 60 fps
(16.6 ms budget). The numpy vectorised conversion is the key; a pure Python
pixel loop would be far too slow.

**Limitations:**

- No GPU acceleration (software rendering only)
- No vsync / double-buffering (tearing possible, unlikely at 800x480)
- No touchscreen input via SDL (the `dummy` driver ignores input devices)
- Must handle input separately (e.g. evdev for touchscreen)

For Roberta's use case (2D faces, text, waveforms at 800x480), these
limitations are acceptable.

## Possible future improvement: mmap

The pygame GitHub issue #3168 mentions using `mmap` to map the framebuffer
into memory instead of `write()`. This could be faster for partial updates
(dirty-rect rendering) since you'd only update changed regions:

```python
import mmap

fb_fd = os.open("/dev/fb0", os.O_RDWR)
fb_map = mmap.mmap(fb_fd, 800 * 480 * 2)  # 16bpp
# Write directly into mapped memory
fb_map.seek(0)
fb_map.write(rgb565_data)
```

Not tested yet. The current write() approach is fast enough.

## Required system configuration

For the dummy+fb0 approach to work, seatd and getty should be disabled so
they don't interfere with the framebuffer:

```bash
sudo systemctl disable --now seatd
sudo systemctl disable --now getty@tty1
```

The `pi` user must be in the `video` group to write to `/dev/fb0`:

```bash
id pi | grep -o 'video'  # should print "video"
```

## References

- [pygame/pygame#3168](https://github.com/pygame/pygame/issues/3168) —
  "Can't get display to work on Pi OS Lite"
- [pygame-community/pygame-ce#1597](https://github.com/pygame-community/pygame-ce/issues/1597) —
  "Cannot initialize display on Raspberry Pi OS Lite"
- [Don't Press That — Bookworm DRM](https://dontpressthat.wordpress.com/2025/09/20/bookworm-drm/) —
  SDL_KMSDRM_DEVICE_INDEX solution for Bookworm
- [RPi Forums — Pygame/KMSDRM driver problem](https://forums.raspberrypi.com/viewtopic.php?t=367567) —
  KMSDRM issues on kernel 6.6
- [SDL Wiki — SDL_HINT_KMSDRM_REQUIRE_DRM_MASTER](https://wiki.libsdl.org/SDL3/SDL_HINT_KMSDRM_REQUIRE_DRM_MASTER) —
  DRM master hint documentation
