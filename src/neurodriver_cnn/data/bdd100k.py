"""Reusable parsing of official BDD100K object-detection annotations.

This project does NOT train an object detector. Bounding boxes are only
used to derive frame-level classification ground truth (see
``neurodriver_cnn.labeling``). Field names here are based on the official
BDD100K detection JSON format as documented at
https://doc.bdd100k.com/format.html — verify against
``reports/schema_analysis.md`` (produced by ``scripts/01_inspect_bdd100k.py``)
before trusting these assumptions on a specific release of the dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class BBox:
    """A single object bounding box with derived geometry.

    Coordinates are in absolute pixels, ``(x1, y1)`` top-left and
    ``(x2, y2)`` bottom-right, as published by BDD100K's ``box2d`` field.
    """

    category: str
    x1: float
    y1: float
    x2: float
    y2: float
    image_width: int
    image_height: int

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def normalized_center(self) -> tuple[float, float]:
        cx, cy = self.center
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_width/image_height must be positive to normalize center")
        return cx / self.image_width, cy / self.image_height

    @property
    def bbox_area_ratio(self) -> float:
        image_area = self.image_width * self.image_height
        if image_area <= 0:
            return 0.0
        return self.area / image_area


@dataclass(frozen=True)
class FrameAnnotation:
    """All parsed boxes plus available metadata for one BDD100K frame."""

    image_id: str
    split: str
    width: int
    height: int
    boxes: list[BBox] = field(default_factory=list)
    weather: str | None = None
    scene: str | None = None
    timeofday: str | None = None
    group_id: str | None = None  # real sequence/video id when BDD100K exposes one


def load_bdd100k_labels(labels_path: Path) -> list[dict[str, Any]]:
    """Load a raw BDD100K detection labels JSON file.

    Raises a clear, human-readable error if the file is missing rather than
    letting a bare ``FileNotFoundError`` traceback surface.
    """
    if not labels_path.exists():
        raise FileNotFoundError(
            f"BDD100K annotation file not found: {labels_path}\n"
            "See docs/bdd100k_setup.md for the official download and expected placement."
        )
    with labels_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_box2d(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = obj.get("box2d")
    if not box:
        return None
    try:
        return float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None


def parse_frame_record(record: dict[str, Any], split: str, default_size: tuple[int, int] = (1280, 720)) -> FrameAnnotation:
    """Parse a single BDD100K record (one entry of the labels JSON list).

    ``default_size`` is used only if the record does not carry explicit
    image dimensions, since standard BDD100K images are 1280x720 — this is
    a documented assumption, not an invented one, and should be confirmed
    by ``scripts/01_inspect_bdd100k.py`` against real files.
    """
    image_id = record.get("name", "")
    attributes = record.get("attributes", {}) or {}
    width, height = default_size

    boxes: list[BBox] = []
    for obj in record.get("labels", []) or []:
        coords = _extract_box2d(obj)
        if coords is None:
            continue
        category = obj.get("category", "")
        x1, y1, x2, y2 = coords
        boxes.append(
            BBox(
                category=category,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                image_width=width,
                image_height=height,
            )
        )

    return FrameAnnotation(
        image_id=image_id,
        split=split,
        width=width,
        height=height,
        boxes=boxes,
        weather=attributes.get("weather"),
        scene=attributes.get("scene"),
        timeofday=attributes.get("timeofday"),
        group_id=record.get("videoName") or record.get("video_name"),
    )


def iter_frame_annotations(
    labels_path: Path, split: str, default_size: tuple[int, int] = (1280, 720)
) -> Iterator[FrameAnnotation]:
    """Yield one ``FrameAnnotation`` per record in a BDD100K labels file."""
    for record in load_bdd100k_labels(labels_path):
        yield parse_frame_record(record, split=split, default_size=default_size)


def category_counts(annotations: list[FrameAnnotation]) -> dict[str, int]:
    """Count boxes per raw category across a list of frame annotations."""
    counts: dict[str, int] = {}
    for ann in annotations:
        for box in ann.boxes:
            counts[box.category] = counts.get(box.category, 0) + 1
    return counts
