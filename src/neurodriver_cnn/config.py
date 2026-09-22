"""Load project configuration and resolve the project root.

Keeps paths portable across Windows/Linux/Colab: never hardcode personal
paths, always resolve relative to this file via ``pathlib``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Expected it under {get_project_root() / 'configs'}."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_config(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or get_project_root()
    return _load_json(root / "configs" / "dataset_config.json")


def load_training_config(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or get_project_root()
    return _load_json(root / "configs" / "training_config.json")


CLASS_NAMES = ["CLEAR", "VEHICLE", "PEDESTRIAN", "MIXED"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
