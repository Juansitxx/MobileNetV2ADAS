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
