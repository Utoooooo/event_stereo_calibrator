# Event Camera Stereo Calibration

Tooling for stereo calibration of event cameras.

## Checkerboard Generator (current)

A GUI that displays a **flashing checkerboard** on screen. Event cameras only
report brightness *changes*, so the target alternates between the checkerboard
and an "off" phase to make its edges observable. The setup targets **2 event
cameras + 1 normal frame camera**.

### Features

- Configure rows, columns (number of squares), and square size in pixels.
- Explicit phase **durations** instead of a single rate: checkerboard shown for
  `board_ms` (default **100 ms**), off phase for `off_ms` (default **20 ms**).
- **Inverted checkerboard** toggle: the off phase shows the colour-inverted
  board instead of black, giving event cameras stronger, symmetric edges.
- Live preview and OpenCV inner-corner readout `(cols-1) x (rows-1)`.
- Pick which monitor to show the fullscreen pattern on (handy for stereo rigs).
- Export the static pattern as a PNG.

### Inverted board vs. the normal camera

A colour-inverted checkerboard has its inner corners at the **same physical
positions**, and OpenCV's corner detector is polarity-agnostic, so it does
**not** affect frame-camera calibration accuracy. The only caveat is timing:
expose the normal camera fully **within the long, stable checkerboard phase**
(the 100 ms window) so it never integrates across a transition (which would
blur to a low-contrast gray). Keep its exposure shorter than `board_ms`.

### Install

```bash
python -m pip install -r requirements.txt
```

### Run

```bash
python run.py
```

In the fullscreen display:

- `Esc` / `Q` — exit
- `Space` — pause / resume

### Notes on timing

Phase durations cannot be resolved finer than one monitor refresh interval
(~16.7 ms at 60 Hz). The GUI warns when the shortest phase is below a single
frame of the selected monitor; use a high-refresh display for short off-phase
times.

## Event-Camera Capture (current)

The same GUI (`python run.py`) now also sets up event-camera capture for the
two OpenMV GENX320 cameras:

- **Left / Right camera port** dropdowns (refreshable) for the serial ports.
- **Accumulation time** (default **200 ms**): the capture window per shot.
- **Control GUI display** selector so the control panel can live on a second
  monitor (checkerboard on one screen, control panel on another).
- **Connect && Open Control Panel** connects both cameras and opens the panel.

### Control panel

- **Live monitor**: real-time stitched `Left | Right` event view plus per-camera
  connection status and event rate.
- **Capture**: accumulates events within the accumulation window on each camera
  and saves one PNG per camera into the `captured/` folder, named
  `capture_<timestamp>_<idx>_L.png` / `_R.png`.
- **Exit (disconnect)**: stops streaming and releases both serial ports.

### Architecture

- `event_streaming_cam.py` runs *on* the camera (MicroPython); it streams GENX320
  events over the `events` protocol channel and produces **no logging/output**.
  The capture GUI exec's this script onto each camera.
- `checkerboard_gui/camera.py` is the PC-side capture worker: one thread per
  camera, with separate live and capture accumulators and **no file logging**
  (unlike `event_streaming_pc.py`, which is the separate logging workflow used
  via `event_log_stereo.py`).
- `checkerboard_gui/control.py` is the control-panel window.
