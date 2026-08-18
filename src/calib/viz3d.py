"""3D visualization of the stereo calibration (MATLAB-style scene).

Draws a *camera-centric* scene in camera A's coordinate frame: both camera
frustums at their relative pose, and the checkerboard target plane for every
calibration view, positioned using the per-view extrinsics from camera A.
"""

from __future__ import annotations

import cv2
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .calibration import StereoCalibration


def _frustum_lines(
    R_wc: np.ndarray, c: np.ndarray, K, image_size, depth: float
) -> list[np.ndarray]:
    """Edges of a camera frustum given world<-camera rotation and center.

    ``R_wc`` maps camera-frame points to world (scene) frame; ``c`` is the
    camera centre in the scene frame. The camera looks along +Z.
    """
    w, h = image_size
    fx = K[0, 0]
    fy = K[1, 1]
    # Half extents of the image plane at the chosen depth.
    hx = depth * (w / 2.0) / fx
    hy = depth * (h / 2.0) / fy
    corners_cam = np.array(
        [
            [hx, hy, depth],
            [hx, -hy, depth],
            [-hx, -hy, depth],
            [-hx, hy, depth],
        ]
    )
    corners_world = (R_wc @ corners_cam.T).T + c
    edges = []
    for corner in corners_world:
        edges.append(np.array([c, corner]))  # apex -> corner
    for i in range(4):
        edges.append(np.array([corners_world[i], corners_world[(i + 1) % 4]]))
    return edges


def _board_polygon(rvec: np.ndarray, tvec: np.ndarray, board) -> np.ndarray:
    """Return the 4 outer corners of the board plane in camera A's frame (mm)."""
    R, _ = cv2.Rodrigues(rvec)
    sx = (board.cols - 1) * board.square_mm
    sy = (board.rows - 1) * board.square_mm
    corners_board = np.array(
        [[0, 0, 0], [sx, 0, 0], [sx, sy, 0], [0, sy, 0]], dtype=np.float64
    )
    return (R @ corners_board.T).T + np.asarray(tvec).reshape(1, 3)


def plot_scene(ax, stereo: StereoCalibration) -> None:
    """Render the calibration scene onto a Matplotlib 3D axis."""
    ax.clear()
    cam_a = stereo.camera_a
    cam_b = stereo.camera_b

    # Scene scale from board distances -> reasonable frustum size.
    if cam_a.tvecs:
        dists = [float(np.linalg.norm(t)) for t in cam_a.tvecs]
        scene_scale = float(np.median(dists))
    else:
        scene_scale = float(np.linalg.norm(stereo.T)) * 4.0 or 100.0
    frustum_depth = 0.18 * scene_scale

    # Camera A at origin (looking along +Z).
    R_a = np.eye(3)
    c_a = np.zeros(3)
    # Camera B: R, T map A-frame -> B-frame, so B centre in A frame is -R^T T.
    R_b = stereo.R.T  # camera B -> A (world) rotation
    c_b = (-stereo.R.T @ stereo.T).reshape(3)

    _draw_camera(ax, R_a, c_a, cam_a.K, cam_a.image_size, frustum_depth,
                 color="#1f77b4", label=cam_a.name)
    _draw_camera(ax, R_b, c_b, cam_b.K, cam_b.image_size, frustum_depth,
                 color="#d62728", label=cam_b.name)

    # Target boards (one per view) in camera A frame.
    n = len(cam_a.rvecs or [])
    cmap = _viridis(n)
    polys = []
    for i in range(n):
        quad = _board_polygon(cam_a.rvecs[i], cam_a.tvecs[i], stereo.board)
        polys.append(quad)
    if polys:
        collection = Poly3DCollection(
            polys, alpha=0.35, facecolors=cmap, edgecolors="k", linewidths=0.4
        )
        ax.add_collection3d(collection)

    _set_equal_aspect(ax, stereo, c_b)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(
        f"Stereo scene  -  {n} target poses,  baseline "
        f"{float(np.linalg.norm(stereo.T)):.1f} mm"
    )
    ax.legend(loc="upper right", fontsize=8)
    # Camera-style view: look down the +Z axis from behind the cameras.
    ax.view_init(elev=-65, azim=-90)


def _draw_camera(ax, R_wc, c, K, image_size, depth, color, label) -> None:
    edges = _frustum_lines(R_wc, c, K, image_size, depth)
    ax.add_collection3d(Line3DCollection(edges, colors=color, linewidths=1.5))
    ax.scatter([c[0]], [c[1]], [c[2]], color=color, s=30, label=label)


def _viridis(n: int):
    import matplotlib.cm as cm

    if n <= 0:
        return []
    return [cm.viridis(i / max(1, n - 1)) for i in range(n)]


def _set_equal_aspect(ax, stereo: StereoCalibration, c_b) -> None:
    pts = [np.zeros(3), np.asarray(c_b).reshape(3)]
    cam_a = stereo.camera_a
    for i in range(len(cam_a.rvecs or [])):
        pts.append(np.asarray(cam_a.tvecs[i]).reshape(3))
    pts = np.array(pts)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float((maxs - mins).max())
    span = span if span > 1e-6 else 100.0
    r = span * 0.6
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
