from neurodriver_cnn.labeling.roi import (
    ROIConfig,
    is_bbox_roi_relevant,
    is_center_inside_roi,
    intersection_ratio,
    roi_pixel_box,
)

ROI = ROIConfig()  # x:[0.20,0.80] y:[0.35,1.00] threshold 0.35, on a 1000x1000 image
ROI_BOX = roi_pixel_box(ROI, 1000, 1000)  # (200, 350, 800, 1000)


def test_roi_pixel_box_scales_by_image_size():
    assert ROI_BOX == (200.0, 350.0, 800.0, 1000.0)


def test_center_inside_roi():
    assert is_center_inside_roi(500, 500, ROI_BOX) is True


def test_center_outside_roi():
    assert is_center_inside_roi(50, 50, ROI_BOX) is False


def test_bbox_fully_inside_roi_is_relevant():
    bbox = (300, 400, 500, 600)
    assert is_bbox_roi_relevant(bbox, ROI_BOX, ROI.bbox_intersection_threshold) is True


def test_bbox_fully_outside_roi_is_not_relevant():
    bbox = (0, 0, 100, 100)
    assert is_bbox_roi_relevant(bbox, ROI_BOX, ROI.bbox_intersection_threshold) is False


def test_bbox_partial_intersection_above_threshold_is_relevant():
    # bbox straddles the right ROI edge (x=800); its own center (815) falls
    # outside the ROI, but most of the box area overlaps the ROI.
    bbox = (710, 400, 920, 600)  # area 210x200=42000, overlap 90x200=18000 -> ratio ~0.429
    assert is_center_inside_roi((710 + 920) / 2, (400 + 600) / 2, ROI_BOX) is False
    assert intersection_ratio(bbox, ROI_BOX) >= ROI.bbox_intersection_threshold
    assert is_bbox_roi_relevant(bbox, ROI_BOX, ROI.bbox_intersection_threshold) is True


def test_bbox_partial_intersection_below_threshold_is_not_relevant():
    # Small sliver overlap, center outside ROI, ratio below threshold.
    bbox = (150, 0, 210, 1000)  # area 60x1000=60000, overlap 10x650=6500 -> ratio ~0.108
    assert is_center_inside_roi((150 + 210) / 2, 500, ROI_BOX) is False
    assert intersection_ratio(bbox, ROI_BOX) < ROI.bbox_intersection_threshold
    assert is_bbox_roi_relevant(bbox, ROI_BOX, ROI.bbox_intersection_threshold) is False


def test_zero_area_bbox_has_zero_intersection_ratio():
    bbox = (500, 500, 500, 500)
    assert intersection_ratio(bbox, ROI_BOX) == 0.0
