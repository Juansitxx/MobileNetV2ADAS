"""Parsing for the real available BDD100K source: "BDD100K Images 10K"
(DatasetNinja / Supervisely export format), distributed as a ``.tar``.

This supersedes the earlier assumption of the official BDD100K 100k
``box2d`` label JSON (see docs/decisions_log.md, 2026-09-22 entry): real
inspection of the acquired dataset showed Supervisely-style per-image
annotation files with **polygon** geometry, not ``box2d``. This project
still does not train an object detector — polygons are converted to
axis-aligned bounding boxes only to derive frame-level classification
ground truth (see ``neurodriver_cnn.labeling``).

Expected on-disk layout after extraction (see ``extract_supervisely_tar``):

.. code-block:: text

    <split>/ann/<image_id>.jpg.json   # one JSON object per frame
    <split>/img/<image_id>.jpg

Annotation JSON shape (Supervisely convention — reconfirm against real
files with ``scripts/01_inspect_bdd100k.py`` / ``reports/schema_analysis.md``
before trusting field names beyond what's documented here):

.. code-block:: json

    {
      "size": {"width": 1280, "height": 720},
      "tags": [{"name": "weather", "value": "clear"}, ...],
      "objects": [
        {"classTitle": "car", "geometryType": "polygon",
         "points": {"exterior": [[x, y], ...], "interior": []}}
      ]
    }

The DatasetNinja "test" split has zero annotated objects in every record
(confirmed by real inspection) and carries no usable ground truth — it is
never loaded by ``iter_frames_from_split_dir`` callers in this project.
There is no real video/sequence identifier in this dataset (frames are
independent, not tracking sequences); ``group_id`` is therefore always
``None`` here, never invented.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

SUPPORTED_GEOMETRY_TYPES = {"polygon", "rectangle"}


@dataclass(frozen=True)
class BBox:
    """A single object bounding box with derived geometry.

    For this dataset, coordinates are derived from a Supervisely polygon
    (or rectangle) via axis-aligned min/max, not published directly as a
    box — see ``polygon_to_bbox``.
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
    """All parsed boxes plus available metadata for one frame."""

    image_id: str
    split: str  # raw DatasetNinja split name: "train" or "val" ("test" is never loaded)
    width: int
    height: int
    boxes: list[BBox] = field(default_factory=list)
    weather: str | None = None
    scene: str | None = None
    timeofday: str | None = None
    group_id: str | None = None  # always None: this dataset has no real sequence/video id


def extract_supervisely_tar(tar_path: Path, extract_dir: Path) -> Path:
    """Extract the Supervisely ``.tar`` once (idempotent) into ``extract_dir``.

    Raises a clear error if the tar is missing rather than a bare
    ``FileNotFoundError`` traceback. Uses the ``filter="data"`` safe-extract
    mode on Python versions that support it (3.12+), falling back cleanly
    on older versions.
    """
    if not tar_path.exists():
        raise FileNotFoundError(
            f"Supervisely dataset archive not found: {tar_path}\n"
            "See docs/bdd100k_setup.md for where to obtain and place it."
        )

    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted"
    if marker.exists():
        return extract_dir

    with tarfile.open(tar_path) as tar:
        try:
            tar.extractall(extract_dir, filter="data")
        except TypeError:
            tar.extractall(extract_dir)  # Python < 3.12 has no `filter` kwarg

    marker.write_text("ok", encoding="utf-8")
    return extract_dir


def find_split_dir(extract_dir: Path, split_name: str) -> Path | None:
    """Locate the directory containing ``<split_name>/ann`` and ``<split_name>/img``.

    Searches recursively so an extra wrapping folder inside the tar (e.g.
    ``bdd100k-10k/train/ann/...``) does not need to be guessed in advance.
    """
    matches = [d.parent for d in extract_dir.rglob("ann") if d.parent.name.lower() == split_name.lower()]
    return matches[0] if matches else None


def _extract_tag_value(tags: list[dict[str, Any]], tag_name: str) -> str | None:
    """Best-effort lookup of a Supervisely image tag by name.

    Returns ``None`` (never a guess) if the tag is absent or its shape
    doesn't match the expected ``{"name": ..., "value": ...}`` convention —
    confirm real tag shapes via ``reports/schema_analysis.md`` before
    relying on this for anything beyond descriptive audit fields.
    """
    for tag in tags:
        name = tag.get("name") or tag.get("tagId")
        if isinstance(name, str) and name.lower() == tag_name.lower():
            return tag.get("value")
    return None


def polygon_to_bbox(
    points_exterior: list[list[float]], category: str, image_width: int, image_height: int
) -> BBox | None:
    """Convert a Supervisely polygon/rectangle ``points.exterior`` list to a BBox.

    ``x1=min(x)``, ``y1=min(y)``, ``x2=max(x)``, ``y2=max(y)``. Returns
    ``None`` for degenerate input (fewer than 2 points, or zero area) rather
    than fabricating a box.
    """
    if not points_exterior or len(points_exterior) < 2:
        return None
    xs = [p[0] for p in points_exterior]
    ys = [p[1] for p in points_exterior]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(
        category=category, x1=x1, y1=y1, x2=x2, y2=y2, image_width=image_width, image_height=image_height
    )


def parse_supervisely_record(ann_json: dict[str, Any], image_id: str, split: str) -> FrameAnnotation:
    """Parse one Supervisely per-image annotation JSON object."""
    size = ann_json.get("size", {}) or {}
    width = int(size.get("width", 0) or 0)
    height = int(size.get("height", 0) or 0)

    boxes: list[BBox] = []
    for obj in ann_json.get("objects", []) or []:
        if obj.get("geometryType") not in SUPPORTED_GEOMETRY_TYPES:
            continue
        exterior = ((obj.get("points") or {}).get("exterior")) or []
        bbox = polygon_to_bbox(exterior, obj.get("classTitle", ""), width, height)
        if bbox is not None:
            boxes.append(bbox)

    tags = ann_json.get("tags", []) or []
    return FrameAnnotation(
        image_id=image_id,
        split=split,
        width=width,
        height=height,
        boxes=boxes,
        weather=_extract_tag_value(tags, "weather"),
        scene=_extract_tag_value(tags, "scene"),
        timeofday=_extract_tag_value(tags, "timeofday"),
        group_id=None,
    )


def iter_frames_from_split_dir(split_dir: Path, split: str) -> Iterator[FrameAnnotation]:
    """Yield one ``FrameAnnotation`` per ``<split_dir>/ann/*.json`` file."""
    ann_dir = split_dir / "ann"
    if not ann_dir.exists():
        return
    for ann_path in sorted(ann_dir.glob("*.json")):
        image_id = ann_path.name[: -len(".json")]  # "0001.jpg.json" -> "0001.jpg"
        with ann_path.open("r", encoding="utf-8") as f:
            ann_json = json.load(f)
        yield parse_supervisely_record(ann_json, image_id=image_id, split=split)


def category_counts(annotations: list[FrameAnnotation]) -> dict[str, int]:
    """Count boxes per raw ``classTitle`` category across frame annotations."""
    counts: dict[str, int] = {}
    for ann in annotations:
        for box in ann.boxes:
            counts[box.category] = counts.get(box.category, 0) + 1
    return counts
