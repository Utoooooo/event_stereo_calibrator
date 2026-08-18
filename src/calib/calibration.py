"""Per-camera intrinsic calibration and stereo calibration via OpenCV."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .board import BoardSpec

_STEREO_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)


@dataclass
class CameraCalibration:
    name: str
    image_size: tuple[int, int]  # (width, height)
    K: np.ndarray
    dist: np.ndarray
    rms: float
    rvecs: list[np.ndarray] = None  # per-view board rotation (board -> camera)
    tvecs: list[np.ndarray] = None  # per-view board translation (mm)


@dataclass
class StereoCalibration:
    camera_a: CameraCalibration
    camera_b: CameraCalibration
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    rms: float
    num_pairs: int
    board: BoardSpec


def calibrate_camera(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    name: str,
) -> CameraCalibration:
    """Estimate intrinsics + distortion for a single camera."""
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return CameraCalibration(
        name, image_size, K, dist, float(rms), list(rvecs), list(tvecs)
    )


def stereo_calibrate(
    board: BoardSpec,
    image_points_a: list[np.ndarray],
    image_points_b: list[np.ndarray],
    cam_a: CameraCalibration,
    cam_b: CameraCalibration,
) -> StereoCalibration:
    """Stereo-calibrate two cameras, keeping each camera's intrinsics fixed.

    ``R`` and ``T`` map points from camera A's frame to camera B's frame.
    """
    n = len(image_points_a)
    object_points = [board.object_points() for _ in range(n)]

    rms, _Ka, _da, _Kb, _db, R, T, E, F = cv2.stereoCalibrate(
        object_points,
        image_points_a,
        image_points_b,
        cam_a.K,
        cam_a.dist,
        cam_b.K,
        cam_b.dist,
        cam_a.image_size,
        criteria=_STEREO_CRITERIA,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    return StereoCalibration(
        camera_a=cam_a,
        camera_b=cam_b,
        R=R,
        T=T,
        E=E,
        F=F,
        rms=float(rms),
        num_pairs=n,
        board=board,
    )


def run_full_calibration(
    board: BoardSpec,
    image_points_a: list[np.ndarray],
    image_points_b: list[np.ndarray],
    size_a: tuple[int, int],
    size_b: tuple[int, int],
    name_a: str = "Camera A",
    name_b: str = "Camera B",
) -> StereoCalibration:
    """Calibrate both cameras intrinsically, then stereo-calibrate."""
    n = len(image_points_a)
    object_points = [board.object_points() for _ in range(n)]
    cam_a = calibrate_camera(object_points, image_points_a, size_a, name_a)
    cam_b = calibrate_camera(object_points, image_points_b, size_b, name_b)
    return stereo_calibrate(board, image_points_a, image_points_b, cam_a, cam_b)


def to_payload(stereo: StereoCalibration) -> dict:
    """Build a JSON-serialisable dict describing the calibration result."""

    def cam(c: CameraCalibration) -> dict:
        return {
            "name": c.name,
            "image_size": list(c.image_size),
            "K": c.K,
            "dist": c.dist,
            "reprojection_rms_px": c.rms,
        }

    return {
        "board": {
            "cols": stereo.board.cols,
            "rows": stereo.board.rows,
            "square_mm": stereo.board.square_mm,
        },
        "num_pairs": stereo.num_pairs,
        "camera_a": cam(stereo.camera_a),
        "camera_b": cam(stereo.camera_b),
        "stereo": {
            "R": stereo.R,
            "T": stereo.T,
            "E": stereo.E,
            "F": stereo.F,
            "rms_px": stereo.rms,
            "baseline_mm": float(np.linalg.norm(stereo.T)),
            "note": "R, T map points from camera A frame to camera B frame.",
        },
    }
