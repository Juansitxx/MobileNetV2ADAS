from neurodriver_cnn.data.bdd100k import BBox
from neurodriver_cnn.labeling.frame_labels import classify_fullframe, classify_roi
from neurodriver_cnn.labeling.roi import ROIConfig

W, H = 1000, 1000


def make_box(category: str, x1: float, y1: float, x2: float, y2: float) -> BBox:
    return BBox(category=category, x1=x1, y1=y1, x2=x2, y2=y2, image_width=W, image_height=H)


def test_no_objects_is_clear():
    result = classify_fullframe([])
    assert result["label"] == "CLEAR"
    assert result["has_vehicle"] is False
    assert result["has_pedestrian"] is False


def test_car_truck_bus_motorcycle_are_vehicle():
    for category in ["car", "truck", "bus", "motorcycle"]:
        result = classify_fullframe([make_box(category, 400, 400, 600, 600)])
        assert result["label"] == "VEHICLE", category
        assert result["has_vehicle"] is True


def test_motorcycle_sets_has_motorcycle_flag_and_count():
    result = classify_fullframe([make_box("motorcycle", 400, 400, 600, 600)])
    assert result["label"] == "VEHICLE"
    assert result["has_motorcycle"] is True
    assert result["num_motorcycles"] == 1


def test_pedestrian_alone_is_pedestrian():
    result = classify_fullframe([make_box("pedestrian", 400, 400, 600, 600)])
    assert result["label"] == "PEDESTRIAN"
    assert result["has_pedestrian"] is True
    assert result["has_vehicle"] is False


def test_vehicle_and_pedestrian_is_mixed():
    boxes = [
        make_box("car", 400, 400, 600, 600),
        make_box("pedestrian", 100, 100, 150, 200),
    ]
    result = classify_fullframe(boxes)
    assert result["label"] == "MIXED"


def test_motorcycle_and_pedestrian_is_mixed_with_motorcycle_flag():
    boxes = [
        make_box("motorcycle", 400, 400, 600, 600),
        make_box("pedestrian", 100, 100, 150, 200),
    ]
    result = classify_fullframe(boxes)
    assert result["label"] == "MIXED"
    assert result["has_motorcycle"] is True


def test_rider_and_bicycle_are_not_silently_remapped():
    # rider/bicycle are auxiliary categories: alone, they must NOT count as
    # vehicle or pedestrian and must not force a non-CLEAR label.
    boxes = [make_box("rider", 400, 400, 500, 500), make_box("bicycle", 300, 300, 350, 350)]
    result = classify_fullframe(boxes)
    assert result["label"] == "CLEAR"
    assert result["has_vehicle"] is False
    assert result["has_pedestrian"] is False
    assert result["num_riders"] == 1
    assert result["num_bicycles"] == 1


def test_roi_excludes_object_outside_roi():
    roi = ROIConfig()
    # Box near the top-left corner, outside the default ROI (x:[0.2,0.8] y:[0.35,1.0]).
    boxes = [make_box("car", 0, 0, 50, 50)]
    fullframe = classify_fullframe(boxes)
    roi_result = classify_roi(boxes, W, H, roi)
    assert fullframe["label"] == "VEHICLE"
    assert roi_result["label"] == "CLEAR"


def test_roi_includes_object_inside_roi():
    roi = ROIConfig()
    boxes = [make_box("pedestrian", 500, 500, 600, 600)]
    roi_result = classify_roi(boxes, W, H, roi)
    assert roi_result["label"] == "PEDESTRIAN"


def test_roi_config_is_echoed_in_result():
    roi = ROIConfig()
    result = classify_roi([], W, H, roi)
    assert result["roi_config"]["bbox_intersection_threshold"] == roi.bbox_intersection_threshold


def test_roi_excludes_object_below_min_bbox_area_ratio():
    roi = ROIConfig(min_bbox_area_ratio_by_category={"pedestrian": 0.01})
    # Tiny box well inside the ROI: area = 5x5 = 25 -> ratio 0.000025, below threshold.
    tiny_box = [make_box("pedestrian", 500, 500, 505, 505)]
    result = classify_roi(tiny_box, W, H, roi)
    assert result["label"] == "CLEAR"
    assert result["num_pedestrians"] == 0


def test_roi_keeps_object_at_or_above_min_bbox_area_ratio():
    roi = ROIConfig(min_bbox_area_ratio_by_category={"pedestrian": 0.01})
    # 150x150 box = 22500 area -> ratio 0.0225, above threshold, inside ROI.
    big_box = [make_box("pedestrian", 450, 450, 600, 600)]
    result = classify_roi(big_box, W, H, roi)
    assert result["label"] == "PEDESTRIAN"


def test_roi_default_thresholds_treat_motorcycle_more_leniently_than_car():
    # Fixed 2026-09-22 experiment thresholds: car/truck/bus = 0.01,
    # motorcycle/pedestrian = 0.0005 (see docs/decisions_log.md).
    roi = ROIConfig()  # defaults
    # Same small box (30x30 = 900 -> ratio 0.0009 on a 1000x1000 image):
    # above the motorcycle threshold (0.0005), below the car threshold (0.01).
    small_car = [make_box("car", 500, 500, 530, 530)]
    small_motorcycle = [make_box("motorcycle", 500, 500, 530, 530)]

    car_result = classify_roi(small_car, W, H, roi)
    moto_result = classify_roi(small_motorcycle, W, H, roi)

    assert car_result["label"] == "CLEAR"
    assert moto_result["label"] == "VEHICLE"
    assert moto_result["has_motorcycle"] is True


def test_roi_min_area_ratio_falls_back_for_unlisted_categories():
    roi = ROIConfig()
    assert roi.min_area_ratio_for("rider") == roi.default_min_bbox_area_ratio
    assert roi.min_area_ratio_for("bicycle") == roi.default_min_bbox_area_ratio
    assert roi.min_area_ratio_for("car") == 0.01
    assert roi.min_area_ratio_for("motorcycle") == 0.0005
