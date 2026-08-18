"""Event-camera streaming worker for the calibration capture GUI.

Each :class:`EventCameraWorker` owns one OpenMV GENX320 event camera on a serial
port.  It runs on its own thread: it connects, exec's the camera-side streaming
script, and continuously polls the ``events`` channel.  Incoming events feed two
independent accumulators:

* a **live** buffer for the real-time monitor (reset every time the GUI grabs a
  frame), and
* a **capture** buffer that collects events inside a fixed accumulation window
  when the user presses *Capture*.

Unlike ``event_streaming_pc.py`` this module performs **no file logging** -- it
is purely for on-demand capture and live monitoring.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

try:  # Hardware/serial deps are optional at import time (e.g. for tests).
    from openmv.camera import Camera
except Exception:  # pragma: no cover - environment without openmv
    Camera = None  # type: ignore[assignment]


EVENT_DTYPE = np.dtype("<u2")
EVENT_WORDS = 6
EVENT_BYTES = EVENT_WORDS * 2
SENSOR_W = 320
SENSOR_H = 320

# Event count at (or above) which a pixel is rendered fully white (255) in a
# captured frame.  Counts below ramp linearly; counts at/above clip to 255.
CAPTURE_SATURATION = 20


def list_serial_ports() -> list[tuple[str, str]]:
    """Return available serial ports as ``(device, description)`` tuples."""
    try:
        from serial.tools import list_ports
    except Exception:  # pragma: no cover
        return []
    return [(p.device, p.description or p.device) for p in list_ports.comports()]


def accumulate_events(accum: np.ndarray, events: np.ndarray) -> None:
    """Add event hits (column 4 = x, column 5 = y) into ``accum`` in place."""
    if events.ndim != 2 or events.shape[1] != EVENT_WORDS:
        return
    x = events[:, 4].astype(np.int32)
    y = events[:, 5].astype(np.int32)
    valid = (x >= 0) & (x < SENSOR_W) & (y >= 0) & (y < SENSOR_H)
    if not np.any(valid):
        return
    np.add.at(accum, (y[valid], x[valid]), 1)


def frame_to_uint8(accum: np.ndarray, saturation: int | None = None) -> np.ndarray:
    """Convert an event-count buffer to an 8-bit grayscale image.

    If ``saturation`` is given (and > 0), a fixed scale is used: a pixel whose
    event count reaches ``saturation`` maps to 255, counts below ramp linearly,
    and counts above clip to 255.  This keeps brightness consistent across
    captures and cameras and is robust to single hot pixels.

    If ``saturation`` is None, the buffer is normalized to its own maximum
    (the brightest pixel maps to 255).
    """
    if saturation is not None and saturation > 0:
        scale = 255.0 / saturation
        return np.clip(accum * scale, 0, 255).astype(np.uint8)

    max_count = int(accum.max())
    if max_count <= 0:
        return np.zeros(accum.shape, dtype=np.uint8)
    scale = 255.0 / max_count
    return np.clip(accum * scale, 0, 255).astype(np.uint8)


@dataclass
class CameraParams:
    baudrate: int = 921600
    crc: bool = False
    seq: bool = True
    ack: bool = False
    events: bool = True
    timeout: float = 1.0
    max_retry: int = 3
    max_payload: int = 4096
    drop_rate: float = 0.0


class EventCameraWorker(threading.Thread):
    """Streams one event camera on a background thread (no file logging)."""

    def __init__(
        self,
        port: str,
        name: str,
        script: str,
        params: CameraParams | None = None,
        poll_ms: int = 4,
        connect_timeout_s: float = 12.0,
        start_settle_ms: int = 500,
        ema_alpha: float = 0.2,
    ) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.name = name
        self._script = script
        self._params = params or CameraParams()
        self._poll_sleep = max(0.001, poll_ms / 1000.0)
        self._connect_timeout_s = connect_timeout_s
        self._settle_s = max(0.0, start_settle_ms / 1000.0)
        self._alpha = ema_alpha

        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._live_accum = np.zeros((SENSOR_H, SENSOR_W), dtype=np.uint16)
        self._capture_accum = np.zeros((SENSOR_H, SENSOR_W), dtype=np.uint16)
        self._capture_active = False
        self._capture_end = 0.0
        self._capture_result: np.ndarray | None = None

        # Status (simple scalars; safe to read without the lock).
        self.connected = False
        self.error: str | None = None
        self.event_rate = 0.0
        self.total_events = 0

    # -- public API (called from GUI thread) -------------------------------
    def request_capture(self, duration_s: float) -> None:
        with self._lock:
            self._capture_accum.fill(0)
            self._capture_result = None
            self._capture_end = time.perf_counter() + max(0.0, duration_s)
            self._capture_active = True

    def poll_capture_result(self) -> np.ndarray | None:
        """Return and clear the finished capture buffer, or None if not ready."""
        with self._lock:
            result = self._capture_result
            self._capture_result = None
            return result

    def capture_pending(self) -> bool:
        with self._lock:
            return self._capture_active

    def get_live_frame(self) -> np.ndarray:
        """Return the live 8-bit frame and reset the live accumulator."""
        with self._lock:
            frame = frame_to_uint8(self._live_accum)
            self._live_accum.fill(0)
        return frame

    def stop(self) -> None:
        self._stop.set()
        if self.is_alive():
            self.join(timeout=3.0)

    # -- thread body -------------------------------------------------------
    def run(self) -> None:
        if Camera is None:
            self.error = "openmv package is not available"
            return
        camera = None
        try:
            camera = self._build_camera()
            self._connect_with_timeout(camera)
            self.connected = True
            self._start_stream(camera)
            self._loop(camera)
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.error = str(exc)
        finally:
            self.connected = False
            if camera is not None:
                self._safe_disconnect(camera)

    def _build_camera(self):
        p = self._params
        return Camera(
            self.port,
            baudrate=p.baudrate,
            crc=p.crc,
            seq=p.seq,
            ack=p.ack,
            events=p.events,
            timeout=p.timeout,
            max_retry=p.max_retry,
            max_payload=p.max_payload,
            drop_rate=p.drop_rate,
        )

    def _connect_with_timeout(self, camera) -> None:
        import queue

        result: queue.Queue = queue.Queue(maxsize=1)

        def _target():
            try:
                camera.connect()
                result.put((True, None))
            except Exception as exc:
                result.put((False, exc))

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        worker.join(timeout=self._connect_timeout_s)
        if worker.is_alive():
            raise TimeoutError(
                f"{self.name} connect timed out after {self._connect_timeout_s:.1f}s "
                f"on {self.port}. Check the cable/port or another app holding it."
            )
        ok, err = result.get_nowait()
        if not ok:
            raise err

    def _start_stream(self, camera) -> None:
        camera.stop()
        time.sleep(self._settle_s)
        camera.exec(self._script)

    def _read_events(self, camera):
        status = camera.read_status()
        if not camera.has_channel("events") or not status.get("events"):
            time.sleep(self._poll_sleep)
            return None
        size = camera.channel_size("events")
        if size <= 0:
            time.sleep(self._poll_sleep)
            return None
        data = camera.channel_read("events", size)
        if not data or (len(data) % EVENT_BYTES) != 0:
            return None
        events = np.frombuffer(data, dtype=EVENT_DTYPE).reshape(-1, EVENT_WORDS)
        if events.shape[0] <= 0:
            return None
        return data, events

    def _loop(self, camera) -> None:
        last_time = time.perf_counter()
        while not self._stop.is_set():
            result = self._read_events(camera)
            now = time.perf_counter()

            if result is not None:
                data, events = result
                n = events.shape[0]
                self.total_events += n
                with self._lock:
                    accumulate_events(self._live_accum, events)
                    if self._capture_active:
                        accumulate_events(self._capture_accum, events)

                dt = now - last_time
                last_time = now
                if dt > 0.0:
                    instant = n / dt
                    self.event_rate = (
                        instant
                        if self.event_rate == 0.0
                        else self.event_rate * (1.0 - self._alpha) + instant * self._alpha
                    )

            # Finalize a capture once its window has elapsed.
            with self._lock:
                if self._capture_active and now >= self._capture_end:
                    self._capture_result = self._capture_accum.copy()
                    self._capture_active = False

    @staticmethod
    def _safe_disconnect(camera) -> None:
        for action in ("stop", "disconnect"):
            try:
                getattr(camera, action)()
            except Exception:
                pass
        try:
            serial_obj = getattr(camera, "_serial", None)
            if serial_obj is not None and getattr(serial_obj, "is_open", False):
                serial_obj.close()
        except Exception:
            pass
