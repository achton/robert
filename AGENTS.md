# Roberta — Office Robot

An event-driven, multi-modal assistant running on a Raspberry Pi 4 with a 7"
touchscreen, camera, microphone, and speaker. Roberta provides conversational
voice interaction, animated facial expressions, and face recognition.

Code lives in `robot/`, research and design docs in `docs/`. Legacy prototypes
and old experiments are archived in `.OLD/` (not tracked by git).

## Guiding Principle: Simplicity and Readability

The project team is not deeply experienced in Python. All code must be as
simple, clear, and easy to understand as possible.

- **Clarity over cleverness.** Straightforward, idiomatic code. No complex
  one-liners or obscure language features.
- **Descriptive names.** For variables, functions, classes — be verbose.
- **Comment the *why*.** Assume someone with minimal context is reading it.
- **Small functions, small modules.** Prefer focused pieces over monoliths.
- **Fail fast.** Do not write complex fallback logic for theoretical edge
  cases. A clear failure is better than hidden complexity.

This principle outweighs premature optimization. A simple, working, and
understandable system is the primary goal.

## Hardware

| Component   | Model                                     |
|-------------|-------------------------------------------|
| Computer    | Raspberry Pi 4 Model B (4 GB)             |
| Display     | Official 7" DSI touchscreen (800x480)     |
| Camera      | Pi High Quality Camera (IMX477)           |
| Microphone  | ReSpeaker 2-Mic HAT v2.0 (TLV320AIC3104)  |
| Speaker     | USB speaker (Noname)                      |
| Chassis     | Wild Thumper 6WD (future project)         |

## Target Environments

The code must run in both environments. Services detect available hardware at
startup and gracefully disable features that are not present (e.g. no camera
on the laptop, no framebuffer on the desktop).

**Raspberry Pi (production):**

- Raspberry Pi OS Lite, Debian 13 (Trixie), 64-bit
- Python 3.13+, managed with `uv`
- No desktop environment — headless with direct framebuffer rendering
- All hardware listed above is attached

**Laptop (development):**

- x86_64 Linux (Ubuntu/similar)
- Python 3.13+, managed with `uv`
- X11 or Wayland desktop — pygame renders in a normal window
- No Pi-specific hardware; camera and mic use laptop defaults if available

## Working With This Repo

- **All changes are coordinated with the user.** Do not make sweeping changes
  without confirmation.
- **One feature at a time.** Each commit should be a small, focused addition.
- **Run checks before committing.** Tooling will be defined as we go.
- Refer to `docs/PLAN.md` for the implementation roadmap.
