from neurodriver_cnn.data.bdd100k import parse_supervisely_record, polygon_to_bbox


def test_polygon_to_bbox_computes_min_max():
    points = [[100, 200], [150, 180], [140, 260], [90, 240]]
    bbox = polygon_to_bbox(points, category="car", image_width=1000, image_height=1000)
    assert bbox is not None
    assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == (90, 180, 150, 260)


def test_polygon_to_bbox_rejects_degenerate_input():
    assert polygon_to_bbox([], "car", 1000, 1000) is None
    assert polygon_to_bbox([[10, 10]], "car", 1000, 1000) is None
    assert polygon_to_bbox([[10, 10], [10, 10]], "car", 1000, 1000) is None  # zero area


def test_rectangle_geometry_also_converts_via_min_max():
    # Supervisely rectangles store exterior as two opposite corners.
    points = [[300, 400], [500, 600]]
    bbox = polygon_to_bbox(points, category="pedestrian", image_width=1000, image_height=1000)
    assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == (300, 400, 500, 600)


def test_parse_supervisely_record_full_shape():
    ann_json = {
        "size": {"width": 1280, "height": 720},
        "tags": [{"name": "weather", "value": "clear"}, {"name": "timeofday", "value": "daytime"}],
        "objects": [
            {
                "classTitle": "car",
                "geometryType": "polygon",
                "points": {"exterior": [[100, 100], [200, 100], [200, 200], [100, 200]], "interior": []},
            },
            {
                "classTitle": "pedestrian",
                "geometryType": "rectangle",
                "points": {"exterior": [[500, 500], [520, 560]], "interior": []},
            },
        ],
    }
    frame = parse_supervisely_record(ann_json, image_id="0001.jpg", split="train")

    assert frame.image_id == "0001.jpg"
    assert frame.split == "train"
    assert frame.width == 1280
    assert frame.height == 720
    assert frame.weather == "clear"
    assert frame.timeofday == "daytime"
    assert frame.scene is None
    assert frame.group_id is None  # never invented: this dataset has no real sequence id
    assert len(frame.boxes) == 2
    categories = {b.category for b in frame.boxes}
    assert categories == {"car", "pedestrian"}


def test_parse_supervisely_record_skips_unsupported_geometry_and_missing_size():
    ann_json = {
        "size": {},
        "tags": [],
        "objects": [
            {"classTitle": "lane", "geometryType": "line", "points": {"exterior": [[0, 0], [1, 1]]}},
        ],
    }
    frame = parse_supervisely_record(ann_json, image_id="0002.jpg", split="val")
    assert frame.width == 0
    assert frame.height == 0
    assert frame.boxes == []


def test_parse_supervisely_record_no_objects_is_empty():
    ann_json = {"size": {"width": 100, "height": 100}, "tags": [], "objects": []}
    frame = parse_supervisely_record(ann_json, image_id="0003.jpg", split="train")
    assert frame.boxes == []
