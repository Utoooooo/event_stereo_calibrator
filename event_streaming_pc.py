#!/usr/bin/env python3
#
# This work is licensed under the MIT license.
# Copyright (c) 2013-2025 OpenMV LLC. All rights reserved.
# https://github.com/openmv/openmv/blob/master/LICENSE
#
# Stereo PC-side event streaming:
# 1) start/stop both camera streams from PC,
# 2) log left/right events to event_log_<timestamp>_L/R.bin/json,
# 3) monitor stitched (L|R) video at fixed FPS without blocking logging.

import sys
import argparse
import time
import logging
import json
import threading
import queue
from datetime import datetime

import cv2
import numpy as np
from openmv.camera import Camera


COLOR_LEFT = "\033[36m"
COLOR_RIGHT = "\033[35m"
COLOR_RESET = "\033[0m"

EVENT_DTYPE = np.dtype("<u2")
EVENT_WORDS = 6
EVENT_BYTES = EVENT_WORDS * 2
SENSOR_W = 320
SENSOR_H = 320

LEFT = "L"
RIGHT = "R"
CAMERA_KEYS = (LEFT, RIGHT)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


class StereoEventRenderer:
    """Stitched monitor view (left | right) rendered on a dedicated thread."""

    def __init__(self, fps=30):
        self.frame_period = 1.0 / fps
        self.queue = queue.Queue(maxsize=32)
        self.stop_event = threading.Event()
        self.window_name = "Stereo Event Monitor (L | R) - q closes monitor"
        self.frame_accum = {
            LEFT: np.zeros((SENSOR_H, SENSOR_W), dtype=np.uint16),
            RIGHT: np.zeros((SENSOR_H, SENSOR_W), dtype=np.uint16),
        }
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.monitor_closed = False

    @staticmethod
    def _to_frame(accum):
        max_count = int(accum.max())
        if max_count <= 0:
            return np.zeros((SENSOR_H, SENSOR_W), dtype=np.uint8)
        scale = 255.0 / max_count
        return np.clip(accum * scale, 0, 255).astype(np.uint8)

    def start(self):
        self._thread.start()

    def stop(self):
        self.stop_event.set()
        self._thread.join(timeout=2.0)
        cv2.destroyAllWindows()

    def submit(self, camera_key, events):
        if self.monitor_closed or events.size == 0:
            return
        batch = events.copy()
        item = (camera_key, batch)
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                pass

    def _accumulate(self, camera_key, events):
        if events.ndim != 2 or events.shape[1] != EVENT_WORDS:
            return
        x = events[:, 4].astype(np.int32)
        y = events[:, 5].astype(np.int32)
        valid = (x >= 0) & (x < SENSOR_W) & (y >= 0) & (y < SENSOR_H)
        if not np.any(valid):
            return
        x = x[valid]
        y = y[valid]
        np.add.at(self.frame_accum[camera_key], (y, x), 1)

    def _render_stitched(self):
        left_img = self._to_frame(self.frame_accum[LEFT])
        right_img = self._to_frame(self.frame_accum[RIGHT])
        stitched = np.hstack((left_img, right_img))
        cv2.putText(stitched, "L", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv2.LINE_AA)
        cv2.putText(stitched, "R", (SENSOR_W + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv2.LINE_AA)
        cv2.imshow(self.window_name, stitched)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.monitor_closed = True
            cv2.destroyWindow(self.window_name)
        self.frame_accum[LEFT].fill(0)
        self.frame_accum[RIGHT].fill(0)

    def _run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, np.zeros((SENSOR_H, SENSOR_W * 2), dtype=np.uint8))
        cv2.waitKey(1)
        next_render_time = time.perf_counter() + self.frame_period

        while not self.stop_event.is_set() or not self.queue.empty():
            timeout = max(0.0, min(0.02, next_render_time - time.perf_counter()))
            try:
                camera_key, events = self.queue.get(timeout=timeout)
                self._accumulate(camera_key, events)
            except queue.Empty:
                pass

            now = time.perf_counter()
            if now >= next_render_time:
                if not self.monitor_closed:
                    self._render_stitched()
                next_render_time += self.frame_period
                while next_render_time < now:
                    next_render_time += self.frame_period


class StereoEventLogger:
    def __init__(self, fmt):
        self.fmt = fmt
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.paths = {}
        self._files = {}
        self._json_first = {LEFT: True, RIGHT: True}
        self.total_events = {LEFT: 0, RIGHT: 0}
        self.total_bytes = {LEFT: 0, RIGHT: 0}

    def open(self):
        if self.fmt == "none":
            return
        mode = "wb" if self.fmt == "bin" else "w"
        for key in CAMERA_KEYS:
            path = f"event_log_{self.timestamp}_{key}.{self.fmt}"
            self.paths[key] = path
            self._files[key] = open(path, mode, encoding=None if self.fmt == "bin" else "utf-8")
            if self.fmt == "json":
                self._files[key].write("[\n")
            logging.info("Logging %s camera to %s", key, path)

    def write(self, camera_key, data, events):
        file_obj = self._files.get(camera_key)
        if file_obj is None:
            return
        if self.fmt == "bin":
            file_obj.write(data)
            self.total_bytes[camera_key] += len(data)
            self.total_events[camera_key] += events.shape[0]
            return

        first = self._json_first[camera_key]
        for event in events:
            obj = {
                "type": int(event[0]),
                "timestamp_s": int(event[1]),
                "timestamp_ms": int(event[2]),
                "timestamp_us": int(event[3]),
                "x": int(event[4]),
                "y": int(event[5]),
            }
            if not first:
                file_obj.write(",\n")
            file_obj.write(json.dumps(obj))
            first = False
            self.total_events[camera_key] += 1
        self._json_first[camera_key] = first

    def close(self):
        for key, file_obj in self._files.items():
            if self.fmt == "json":
                file_obj.write("\n]\n")
            file_obj.flush()
            file_obj.close()
        self._files.clear()


def build_camera(port, args):
    return Camera(
        port,
        baudrate=args.baudrate,
        crc=args.crc,
        seq=args.seq,
        ack=args.ack,
        events=args.events,
        timeout=args.timeout,
        max_retry=args.max_retry,
        max_payload=args.max_payload,
        drop_rate=args.drop_rate,
    )


def connect_camera_with_timeout(camera, name, timeout_s):
    result_queue = queue.Queue(maxsize=1)

    def _target():
        try:
            camera.connect()
            result_queue.put((True, None))
        except Exception as exc:  # pragma: no cover - defensive wrapper
            result_queue.put((False, exc))

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)
    if worker.is_alive():
        raise TimeoutError(
            f"{name} camera connect timed out after {timeout_s:.1f}s. "
            f"Check USB cable/port and whether another app is holding the COM port."
        )
    ok, err = result_queue.get_nowait()
    if not ok:
        raise err


def safe_disconnect_camera(camera):
    """Best-effort serial cleanup to guarantee COM port release."""
    try:
        camera.stop()
    except Exception:
        pass
    try:
        camera.disconnect()
    except Exception:
        pass
    # Fallback: force-close pyserial handle if still open.
    try:
        serial_obj = getattr(camera, "_serial", None)
        if serial_obj is not None and getattr(serial_obj, "is_open", False):
            serial_obj.close()
    except Exception:
        pass


def start_camera_stream(camera, script, name, settle_time):
    camera.stop()
    time.sleep(settle_time)
    camera.exec(script)
    logging.info("%s camera streaming started.", name)


def read_camera_events(camera, poll_sleep):
    status = camera.read_status()
    if not camera.has_channel("events") or not status.get("events"):
        time.sleep(poll_sleep)
        return None
    size = camera.channel_size("events")
    if size <= 0:
        time.sleep(poll_sleep)
        return None
    data = camera.channel_read("events", size)
    if not data:
        return None
    if (len(data) % EVENT_BYTES) != 0:
        logging.warning("Misaligned packet: %d bytes (not multiple of %d)", len(data), EVENT_BYTES)
        return None
    events = np.frombuffer(data, dtype=EVENT_DTYPE).reshape(-1, EVENT_WORDS)
    if events.shape[0] <= 0:
        return None
    return data, events, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-left", required=True, help="Left camera serial port")
    parser.add_argument("--port-right", required=True, help="Right camera serial port")
    parser.add_argument("--script", required=True, help="Camera-side script used by default for both cameras")
    parser.add_argument("--script-left", default=None, help="Optional left script override")
    parser.add_argument("--script-right", default=None, help="Optional right script override")
    parser.add_argument("--poll", type=int, default=4, help="Poll sleep in ms (default: 4)")
    parser.add_argument("--timeout", type=float, default=1.0, help="Protocol timeout in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--baudrate", type=int, default=921600, help="Serial baudrate (default: 921600)")
    parser.add_argument(
        "--crc", type=str2bool, nargs="?", const=True, default=False, help="Enable CRC validation"
    )
    parser.add_argument(
        "--seq", type=str2bool, nargs="?", const=True, default=True, help="Enable sequence validation"
    )
    parser.add_argument(
        "--ack", type=str2bool, nargs="?", const=True, default=False, help="Enable packet acknowledgment"
    )
    parser.add_argument(
        "--events", type=str2bool, nargs="?", const=True, default=True, help="Enable event notifications"
    )
    parser.add_argument("--max-retry", type=int, default=3, help="Maximum number of retries")
    parser.add_argument("--max-payload", type=int, default=4096, help="Maximum payload size")
    parser.add_argument("--drop-rate", type=float, default=0.0, help="Packet drop simulation rate")
    parser.add_argument("--quiet", action="store_true", help="Suppress camera stdout")
    parser.add_argument("--alpha", type=float, default=0.2, help="EMA smoothing factor for stats")
    parser.add_argument(
        "--record-format", choices=["none", "bin", "json"], default="bin", help="Logging format"
    )
    parser.add_argument("--render", type=str2bool, nargs="?", const=True, default=True, help="Enable monitor")
    parser.add_argument("--fps", type=float, default=30.0, help="Monitor FPS")
    parser.add_argument("--start-settle-ms", type=int, default=500, help="Delay after stop before exec")
    parser.add_argument(
        "--connect-timeout-s",
        type=float,
        default=12.0,
        help="Max seconds to wait per camera connection before failing fast",
    )
    args = parser.parse_args()

    if not (0.0 < args.alpha <= 1.0):
        raise ValueError("--alpha must be in (0, 1].")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0.")
    if args.connect_timeout_s <= 0:
        raise ValueError("--connect-timeout-s must be > 0.")

    log_level = logging.DEBUG if args.debug else (logging.INFO if not args.quiet else logging.ERROR)
    logging.basicConfig(format="%(relativeCreated)010.3f - %(message)s", level=log_level)

    script_left_path = args.script_left or args.script
    script_right_path = args.script_right or args.script
    with open(script_left_path, "r", encoding="utf-8") as f:
        script_left = f.read()
    with open(script_right_path, "r", encoding="utf-8") as f:
        script_right = f.read()
    logging.info("Loaded left script from %s", script_left_path)
    logging.info("Loaded right script from %s", script_right_path)

    renderer = StereoEventRenderer(args.fps) if args.render else None
    logger_obj = StereoEventLogger(args.record_format)
    poll_sleep = max(0.001, args.poll / 1000.0)
    settle_time = max(0.0, args.start_settle_ms / 1000.0)

    cameras = {}
    ema_rate = {LEFT: 0.0, RIGHT: 0.0}
    ema_bw = {LEFT: 0.0, RIGHT: 0.0}
    last_time = {LEFT: time.perf_counter(), RIGHT: time.perf_counter()}
    total_events = {LEFT: 0, RIGHT: 0}
    start_time = time.perf_counter()
    next_stats_time = start_time + 1.0

    try:
        cameras[LEFT] = build_camera(args.port_left, args)
        cameras[RIGHT] = build_camera(args.port_right, args)
        connect_camera_with_timeout(cameras[LEFT], "LEFT", args.connect_timeout_s)
        logging.info("Connected LEFT camera on %s", args.port_left)
        connect_camera_with_timeout(cameras[RIGHT], "RIGHT", args.connect_timeout_s)
        logging.info("Connected RIGHT camera on %s", args.port_right)

        start_camera_stream(cameras[LEFT], script_left, "LEFT", settle_time)
        start_camera_stream(cameras[RIGHT], script_right, "RIGHT", settle_time)
        logger_obj.open()
        if renderer is not None:
            renderer.start()

        while True:
            for key in CAMERA_KEYS:
                camera = cameras[key]
                camera_name = "LEFT" if key == LEFT else "RIGHT"
                result = read_camera_events(camera, poll_sleep)
                if result is None:
                    continue
                data, events, status = result

                if not args.quiet and status and status.get("stdout"):
                    text = camera.read_stdout()
                    if text:
                        color = COLOR_LEFT if key == LEFT else COLOR_RIGHT
                        print(f"{color}[{camera_name}] {text}{COLOR_RESET}", end="")

                logger_obj.write(key, data, events)
                if renderer is not None:
                    renderer.submit(key, events)

                now = time.perf_counter()
                dt = now - last_time[key]
                last_time[key] = now
                if dt > 0.0:
                    instant_rate = events.shape[0] / dt
                    instant_bw = len(data) / 1048576.0 / dt
                    if ema_rate[key] == 0.0:
                        ema_rate[key] = instant_rate
                        ema_bw[key] = instant_bw
                    else:
                        ema_rate[key] = ema_rate[key] * (1.0 - args.alpha) + instant_rate * args.alpha
                        ema_bw[key] = ema_bw[key] * (1.0 - args.alpha) + instant_bw * args.alpha

                total_events[key] += events.shape[0]

            now = time.perf_counter()
            if now >= next_stats_time:
                elapsed = now - start_time
                logging.info(
                    "L: rate=%8.0f ev/s bw=%5.2f MB/s total=%9d | R: rate=%8.0f ev/s bw=%5.2f MB/s total=%9d | uptime=%7.1fs",
                    ema_rate[LEFT],
                    ema_bw[LEFT],
                    total_events[LEFT],
                    ema_rate[RIGHT],
                    ema_bw[RIGHT],
                    total_events[RIGHT],
                    elapsed,
                )
                next_stats_time = now + 1.0

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except Exception as e:
        logging.error("Error: %s", e)
        if args.debug:
            import traceback

            logging.error("%s", traceback.format_exc())
        sys.exit(1)
    finally:
        for key in CAMERA_KEYS:
            camera = cameras.get(key)
            if camera is None:
                continue
            safe_disconnect_camera(camera)
            logging.info("%s serial port disconnected.", key)
        if renderer is not None:
            renderer.stop()
        logger_obj.close()
        for key in CAMERA_KEYS:
            if key in logger_obj.paths:
                logging.info("Saved %s log: %s", key, logger_obj.paths[key])


if __name__ == "__main__":
    main()
