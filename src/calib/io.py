"""Image-pair discovery and result serialization (Qt-free)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ImagePair:
    name: str
    path_a: Path
    path_b: Path


def _list_images(folder: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images[path.stem] = path
    return images


def find_pairs(folder_a: str | Path, folder_b: str | Path) -> tuple[list[ImagePair], str]:
    """Match images between two folders.

    Pairs are matched by shared filename stem (e.g. ``001.png`` <-> ``001.png``).
    If no stems match, falls back to pairing by sorted order.

    Returns (pairs, method) where method is "stem" or "order".
    """
    folder_a = Path(folder_a)
    folder_b = Path(folder_b)
    if not folder_a.is_dir() or not folder_b.is_dir():
        return [], "none"

    images_a = _list_images(folder_a)
    images_b = _list_images(folder_b)

    common = sorted(set(images_a) & set(images_b))
    if common:
        return [ImagePair(s, images_a[s], images_b[s]) for s in common], "stem"

    # Fallback: pair by sorted order up to the shorter list.
    list_a = [images_a[s] for s in sorted(images_a)]
    list_b = [images_b[s] for s in sorted(images_b)]
    pairs = [
        ImagePair(f"{pa.stem}|{pb.stem}", pa, pb)
        for pa, pb in zip(list_a, list_b)
    ]
    return pairs, ("order" if pairs else "none")


def _matrix(m: np.ndarray | None) -> list | None:
    return None if m is None else np.asarray(m).tolist()


def save_results(out_dir: str | Path, payload: dict) -> tuple[Path, Path]:
    """Save calibration results as JSON and NPZ. Returns (json_path, npz_path)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"calibration_{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_matrix)

    # Flatten arrays for the NPZ.
    npz_path = out_dir / f"calibration_{stamp}.npz"
    arrays = {}
    for cam_key in ("camera_a", "camera_b"):
        cam = payload.get(cam_key, {})
        for field in ("K", "dist", "image_size"):
            if cam.get(field) is not None:
                arrays[f"{cam_key}_{field}"] = np.asarray(cam[field])
    for field in ("R", "T", "E", "F"):
        if payload.get("stereo", {}).get(field) is not None:
            arrays[field] = np.asarray(payload["stereo"][field])
    np.savez(npz_path, **arrays)

    return json_path, npz_path
