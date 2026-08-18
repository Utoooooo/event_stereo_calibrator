"""Persist GUI options to a small JSON cache between sessions.

Settings are best-effort: loading a missing/corrupt file yields an empty dict,
and callers are expected to fall back to defaults for any value that is missing
or no longer valid (e.g. a monitor index out of range or a serial port that is
no longer connected).
"""

from __future__ import annotations

import json
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parent.parent / ".gui_settings.json"


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass
