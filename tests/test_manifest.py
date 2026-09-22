import pandas as pd

from neurodriver_cnn.data.manifest import (
    check_split_overlap,
    group_aware_split,
    validate_manifest_invariants,
)


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_path": "a.jpg",
                "label": "CLEAR",
                "split": "TRAIN",
                "source_dataset": "BDD100K",
                "has_vehicle": False,
                "has_pedestrian": False,
                "has_motorcycle": False,
            },
            {
                "image_path": "b.jpg",
                "label": "VEHICLE",
                "split": "TRAIN",
                "source_dataset": "BDD100K",
                "has_vehicle": True,
                "has_pedestrian": False,
                "has_motorcycle": True,
            },
            {
                "image_path": "c.jpg",
                "label": "MIXED",
                "split": "VALIDATION",
                "source_dataset": "BDD100K",
                "has_vehicle": True,
                "has_pedestrian": True,
                "has_motorcycle": False,
            },
        ]
    )


def test_valid_manifest_has_no_violations():
    assert validate_manifest_invariants(_valid_df()) == []


def test_missing_required_column_is_reported():
    df = _valid_df().drop(columns=["has_motorcycle"])
    violations = validate_manifest_invariants(df)
    assert any("Missing required columns" in v for v in violations)


def test_label_inconsistent_with_flags_is_reported():
    df = _valid_df()
    df.loc[0, "label"] = "VEHICLE"  # row 0 has has_vehicle=False -> should be CLEAR
    violations = validate_manifest_invariants(df)
    assert any("inconsistent with has_vehicle" in v for v in violations)


def test_motorcycle_without_vehicle_is_reported():
    df = _valid_df()
    df.loc[0, "has_motorcycle"] = True  # row 0 has has_vehicle=False
    violations = validate_manifest_invariants(df)
    assert any("has_motorcycle=True but has_vehicle=False" in v for v in violations)


def test_duplicate_image_path_is_reported():
    df = _valid_df()
    df.loc[2, "image_path"] = "a.jpg"
    violations = validate_manifest_invariants(df)
    assert any("duplicate image_path" in v for v in violations)


def test_group_aware_split_keeps_group_together():
    df = pd.DataFrame(
        {
            "image_path": [f"img_{i}.jpg" for i in range(20)],
            "group_id": [f"video_{i % 5}" for i in range(20)],
        }
    )
    split = group_aware_split(df, group_col="group_id", val_ratio=0.2, seed=42)
    df = df.assign(split=split)
    per_group_splits = df.groupby("group_id")["split"].nunique()
    assert (per_group_splits == 1).all()


def test_group_aware_split_falls_back_without_group_col():
    df = pd.DataFrame({"image_path": [f"img_{i}.jpg" for i in range(10)]})
    split = group_aware_split(df, group_col="missing_col", val_ratio=0.3, seed=42)
    assert set(split.unique()) <= {"TRAIN", "VALIDATION"}
    assert len(split) == 10


def test_check_split_overlap_detects_leaked_path():
    df = pd.DataFrame(
        {
            "image_path": ["a.jpg", "a.jpg", "b.jpg"],
            "split": ["TRAIN", "TEST", "TRAIN"],
        }
    )
    issues = check_split_overlap(df)
    assert any("image_path" in i for i in issues)


def test_check_split_overlap_detects_leaked_group():
    df = pd.DataFrame(
        {
            "image_path": ["a.jpg", "b.jpg", "c.jpg"],
            "split": ["TRAIN", "TEST", "TRAIN"],
            "group_id": ["v1", "v1", "v2"],
        }
    )
    issues = check_split_overlap(df, group_col="group_id")
    assert any("group_id" in i for i in issues)


def test_check_split_overlap_clean_data_has_no_issues():
    df = pd.DataFrame(
        {
            "image_path": ["a.jpg", "b.jpg", "c.jpg"],
            "split": ["TRAIN", "TEST", "TRAIN"],
            "group_id": ["v1", "v2", "v1"],
        }
    )
    assert check_split_overlap(df, group_col="group_id") == []
