"""Frame-level classification: BDD100K boxes -> {CLEAR, VEHICLE, PEDESTRIAN, MIXED}.

This is the core labeling logic shared by Full Frame and ADAS ROI strategies.
Category names are matched case-sensitively against BDD100K's published
category vocabulary (confirm against reports/schema_analysis.md).
"""

from __future__ import annotations

from typing import Iterable

from neurodriver_cnn.data.bdd100k import BBox
from neurodriver_cnn.labeling.roi import ROIConfig, is_bbox_roi_relevant, roi_pixel_box

VEHICLE_CATEGORIES = {"car", "truck", "bus", "motorcycle"}
PEDESTRIAN_CATEGORIES = {"pedestrian"}
AUXILIARY_CATEGORIES = {"rider", "bicycle"}
MOTORCYCLE_CATEGORY = "motorcycle"


def derive_label(has_vehicle: bool, has_pedestrian: bool) -> str:
    """Derive the 4-state ADAS label from the two binary training targets.

    This is the single source of truth for the (has_vehicle, has_pedestrian)
    -> {CLEAR, VEHICLE, PEDESTRIAN, MIXED} mapping, used both when building
    manifest ground truth here and when deriving predictions from the
    models' two sigmoid outputs at evaluation time
    (``neurodriver_cnn.evaluation.metrics``).
    """
    if has_vehicle and has_pedestrian:
        return "MIXED"
    if has_vehicle:
        return "VEHICLE"
    if has_pedestrian:
        return "PEDESTRIAN"
    return "CLEAR"


def _summarize(relevant_boxes: list[BBox]) -> dict:
    num_vehicles = sum(1 for b in relevant_boxes if b.category in VEHICLE_CATEGORIES)
    num_pedestrians = sum(1 for b in relevant_boxes if b.category in PEDESTRIAN_CATEGORIES)
    num_motorcycles = sum(1 for b in relevant_boxes if b.category == MOTORCYCLE_CATEGORY)
    num_riders = sum(1 for b in relevant_boxes if b.category == "rider")
    num_bicycles = sum(1 for b in relevant_boxes if b.category == "bicycle")

    has_vehicle = num_vehicles > 0
    has_pedestrian = num_pedestrians > 0
    has_motorcycle = num_motorcycles > 0

    return {
        "label": derive_label(has_vehicle, has_pedestrian),
        "has_vehicle": has_vehicle,
        "has_pedestrian": has_pedestrian,
        "has_motorcycle": has_motorcycle,
        "num_vehicles": num_vehicles,
        "num_pedestrians": num_pedestrians,
        "num_motorcycles": num_motorcycles,
        "num_riders": num_riders,
        "num_bicycles": num_bicycles,
    }


def classify_fullframe(boxes: Iterable[BBox]) -> dict:
    """Full-Frame strategy: every mapped object in the image counts."""
    return _summarize(list(boxes))


def classify_roi(
    boxes: Iterable[BBox],
    image_width: int,
    image_height: int,
    roi_config: ROIConfig | None = None,
) -> dict:
    """ADAS ROI strategy: only objects relevant to the configured ROI count.

    Objects smaller than a per-category minimum bbox-area-ratio
    (``roi_config.min_area_ratio_for(category)``) are excluded before the
    ROI-relevance test: car/truck/bus need a larger footprint (0.01) than
    motorcycle/pedestrian (0.0005), which are kept smaller specifically to
    preserve motorcycle representation for the future Colombian domain (see
    docs/decisions_log.md, 2026-09-22 — this is the final, fixed labeling
    strategy for the experiment).
    """
    roi_config = roi_config or ROIConfig()
    roi_box = roi_pixel_box(roi_config, image_width, image_height)

    sized = [b for b in boxes if b.bbox_area_ratio >= roi_config.min_area_ratio_for(b.category)]
    relevant = [
        b
        for b in sized
        if is_bbox_roi_relevant(
            (b.x1, b.y1, b.x2, b.y2), roi_box, roi_config.bbox_intersection_threshold
        )
    ]
    result = _summarize(relevant)
    result["roi_config"] = {
        "x_min": roi_config.x_min,
        "x_max": roi_config.x_max,
        "y_min": roi_config.y_min,
        "y_max": roi_config.y_max,
        "bbox_intersection_threshold": roi_config.bbox_intersection_threshold,
        "min_bbox_area_ratio_by_category": dict(roi_config.min_bbox_area_ratio_by_category),
    }
    return result
