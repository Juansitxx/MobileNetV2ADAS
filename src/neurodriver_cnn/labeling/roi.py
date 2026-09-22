"""Pure, testable ADAS ROI geometry helpers.

The ROI is a normalized rectangle (fractions of image width/height) meant
to approximate the forward driving corridor relevant to an ADAS. All
functions here are pure (no I/O) so they are cheap to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Final experiment thresholds (fixed 2026-09-22, see docs/decisions_log.md):
# car/truck/bus need a larger real-world footprint to count as ADAS-relevant,
# while motorcycle/pedestrian use a much smaller threshold so small-but-real
# detections are kept — motorcycle in particular, since real counts showed
# 0.0005 retains 145 TRAIN / 27 VAL motorcycle frames with minimal effect on
# overall vehicle balance, which matters for the future Colombian domain
# (higher motorcycle density than BDD100K/US driving).
DEFAULT_MIN_BBOX_AREA_RATIO_BY_CATEGORY = {
    "car": 0.01,
    "truck": 0.01,
    "bus": 0.01,
    "motorcycle": 0.0005,
    "pedestrian": 0.0005,
}
# Fallback for categories with no explicit threshold above (rider, bicycle):
# not specified by the professor/experiment decision — defaulted to the
# same small threshold as motorcycle/pedestrian since these are similarly
# small objects. Revisit if evidence suggests otherwise.
DEFAULT_MIN_BBOX_AREA_RATIO_FALLBACK = 0.0005


@dataclass(frozen=True)
class ROIConfig:
    x_min: float = 0.20
    x_max: float = 0.80
    y_min: float = 0.35
    y_max: float = 1.00
    bbox_intersection_threshold: float = 0.35
    min_bbox_area_ratio_by_category: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MIN_BBOX_AREA_RATIO_BY_CATEGORY)
    )
    default_min_bbox_area_ratio: float = DEFAULT_MIN_BBOX_AREA_RATIO_FALLBACK

    def min_area_ratio_for(self, category: str) -> float:
        return self.min_bbox_area_ratio_by_category.get(category, self.default_min_bbox_area_ratio)

    @classmethod
    def from_dict(cls, d: dict) -> "ROIConfig":
        return cls(
            x_min=d.get("x_min", 0.20),
            x_max=d.get("x_max", 0.80),
            y_min=d.get("y_min", 0.35),
            y_max=d.get("y_max", 1.00),
            bbox_intersection_threshold=d.get("bbox_intersection_threshold", 0.35),
            min_bbox_area_ratio_by_category=d.get(
                "min_bbox_area_ratio_by_category", dict(DEFAULT_MIN_BBOX_AREA_RATIO_BY_CATEGORY)
            ),
            default_min_bbox_area_ratio=d.get(
                "default_min_bbox_area_ratio", DEFAULT_MIN_BBOX_AREA_RATIO_FALLBACK
            ),
        )


def roi_pixel_box(roi: ROIConfig, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    """Convert a normalized ROI to absolute pixel coordinates for one image."""
    return (
        roi.x_min * image_width,
        roi.y_min * image_height,
        roi.x_max * image_width,
        roi.y_max * image_height,
    )


def _intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def is_center_inside_roi(
    center_x: float, center_y: float, roi_box: tuple[float, float, float, float]
) -> bool:
    rx1, ry1, rx2, ry2 = roi_box
    return rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2


def intersection_ratio(
    bbox: tuple[float, float, float, float], roi_box: tuple[float, float, float, float]
) -> float:
    """``intersection_area / bbox_area``; 0.0 if the bbox has zero area."""
    x1, y1, x2, y2 = bbox
    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if bbox_area <= 0:
        return 0.0
    return _intersection_area(bbox, roi_box) / bbox_area


def is_bbox_roi_relevant(
    bbox: tuple[float, float, float, float],
    roi_box: tuple[float, float, float, float],
    threshold: float,
) -> bool:
    """A bbox is ROI-relevant if its center is inside the ROI, OR its
    intersection-over-bbox-area ratio meets the configured threshold.
    """
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if is_center_inside_roi(center_x, center_y, roi_box):
        return True
    return intersection_ratio(bbox, roi_box) >= threshold
