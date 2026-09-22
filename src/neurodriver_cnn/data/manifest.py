"""Common Manifest schema, invariant validation, and group-aware splitting.

The Common Manifest is the contract training code depends on instead of
BDD100K-specific folders (see docs/neurodriver_dataset_contract.md). This
module works on plain ``pandas.DataFrame`` objects so it has no TensorFlow
dependency and can be unit tested cheaply.
"""

from __future__ import annotations

import pandas as pd

from neurodriver_cnn.labeling.frame_labels import derive_label

REQUIRED_COMMON_FIELDS = [
    "image_path",
    "label",
    "split",
    "source_dataset",
    "has_vehicle",
    "has_pedestrian",
    "has_motorcycle",
]

VALID_LABELS = {"CLEAR", "VEHICLE", "PEDESTRIAN", "MIXED"}
VALID_SPLITS = {"TRAIN", "VALIDATION", "TEST"}


def validate_manifest_invariants(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable violation messages (empty if valid).

    Checks:
    - required columns present;
    - label values are one of the four allowed classes;
    - label is consistent with has_vehicle/has_pedestrian flags;
    - has_motorcycle can only be True when has_vehicle is True;
    - no duplicate (image_path) entries;
    - split values are one of TRAIN/VALIDATION/TEST.
    """
    violations: list[str] = []

    missing_cols = [c for c in REQUIRED_COMMON_FIELDS if c not in df.columns]
    if missing_cols:
        violations.append(f"Missing required columns: {missing_cols}")
        return violations  # remaining checks need these columns

    bad_labels = set(df["label"].unique()) - VALID_LABELS
    if bad_labels:
        violations.append(f"Invalid label values found: {sorted(bad_labels)}")

    bad_splits = set(df["split"].unique()) - VALID_SPLITS
    if bad_splits:
        violations.append(f"Invalid split values found: {sorted(bad_splits)}")

    expected_label = df.apply(
        lambda r: derive_label(bool(r["has_vehicle"]), bool(r["has_pedestrian"])), axis=1
    )
    mismatched = df[df["label"] != expected_label]
    if len(mismatched) > 0:
        violations.append(
            f"{len(mismatched)} row(s) have label inconsistent with has_vehicle/has_pedestrian flags"
        )

    bad_motorcycle = df[(df["has_motorcycle"] == True) & (df["has_vehicle"] == False)]  # noqa: E712
    if len(bad_motorcycle) > 0:
        violations.append(
            f"{len(bad_motorcycle)} row(s) have has_motorcycle=True but has_vehicle=False"
        )

    dup_paths = df[df.duplicated(subset=["image_path"], keep=False)]
    if len(dup_paths) > 0:
        violations.append(f"{dup_paths['image_path'].nunique()} duplicate image_path value(s) found")

    return violations


def group_aware_split(
    df: pd.DataFrame,
    group_col: str,
    val_ratio: float,
    seed: int = 42,
) -> pd.Series:
    """Assign TRAIN/VALIDATION labels keeping all rows sharing a group_col
    value on the same side of the split (no leakage across a video/run).

    Falls back to per-row random assignment when ``group_col`` is missing
    or entirely null (documented behavior, never invents group IDs).
    """
    if group_col not in df.columns or df[group_col].isna().all():
        rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=seed)
        n_val = int(round(len(df) * val_ratio))
        val_idx = set(rng.index[:n_val])
        return pd.Series(
            ["VALIDATION" if i in val_idx else "TRAIN" for i in range(len(df))], index=df.index
        )

    groups = df[group_col].dropna().unique()
    groups_series = pd.Series(groups)
    shuffled = groups_series.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val_groups = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 0 else 0
    val_groups = set(shuffled.iloc[:n_val_groups])

    return df[group_col].apply(lambda g: "VALIDATION" if g in val_groups else "TRAIN")


def check_split_overlap(df: pd.DataFrame, group_col: str | None = None) -> list[str]:
    """Detect image_path (and optionally group_col) values appearing in more
    than one split — a direct data-leakage signal.
    """
    issues: list[str] = []

    path_splits = df.groupby("image_path")["split"].nunique()
    leaked_paths = path_splits[path_splits > 1]
    if len(leaked_paths) > 0:
        issues.append(f"{len(leaked_paths)} image_path value(s) appear in more than one split")

    if group_col and group_col in df.columns:
        non_null = df[df[group_col].notna()]
        if len(non_null) > 0:
            group_splits = non_null.groupby(group_col)["split"].nunique()
            leaked_groups = group_splits[group_splits > 1]
            if len(leaked_groups) > 0:
                issues.append(
                    f"{len(leaked_groups)} {group_col} value(s) appear in more than one split"
                )

    return issues
