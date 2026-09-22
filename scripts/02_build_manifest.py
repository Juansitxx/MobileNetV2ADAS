"""Build data/processed/manifests/bdd100k_manifest.csv from real BDD100K data.

Requires BDD100K images + annotations to be present (see
docs/bdd100k_setup.md). Never invents rows, group IDs, or dimensions beyond
the documented BDD100K default (see neurodriver_cnn.data.bdd100k).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neurodriver_cnn.config import load_dataset_config  # noqa: E402
from neurodriver_cnn.data.bdd100k import iter_frame_annotations  # noqa: E402
from neurodriver_cnn.data.manifest import group_aware_split, validate_manifest_invariants  # noqa: E402
from neurodriver_cnn.labeling.frame_labels import classify_fullframe, classify_roi  # noqa: E402
from neurodriver_cnn.labeling.roi import ROIConfig  # noqa: E402

LABEL_FILES = {
    "train": PROJECT_ROOT / "data" / "raw" / "bdd100k" / "labels" / "bdd100k_labels_images_train.json",
    "val": PROJECT_ROOT / "data" / "raw" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json",
}
IMAGE_DIRS = {
    "train": PROJECT_ROOT / "data" / "raw" / "bdd100k" / "images" / "100k" / "train",
    "val": PROJECT_ROOT / "data" / "raw" / "bdd100k" / "images" / "100k" / "val",
}
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "manifests" / "bdd100k_manifest.csv"


def build_rows(roi_config: ROIConfig) -> list[dict]:
    rows: list[dict] = []
    for split, label_path in LABEL_FILES.items():
        if not label_path.exists():
            continue
        image_dir = IMAGE_DIRS[split]
        for ann in iter_frame_annotations(label_path, split=split):
            image_path = image_dir / ann.image_id
            image_available = image_path.exists()

            fullframe = classify_fullframe(ann.boxes)
            roi = classify_roi(ann.boxes, ann.width, ann.height, roi_config)

            rows.append(
                {
                    "image_id": ann.image_id,
                    "image_path": str(image_path),
                    "source_dataset": "BDD100K",
                    "bdd_split": split,
                    "group_id": ann.group_id,
                    "width": ann.width,
                    "height": ann.height,
                    "weather": ann.weather,
                    "scene": ann.scene,
                    "timeofday": ann.timeofday,
                    "object_count": len(ann.boxes),
                    "label_fullframe": fullframe["label"],
                    "has_vehicle_fullframe": fullframe["has_vehicle"],
                    "has_pedestrian_fullframe": fullframe["has_pedestrian"],
                    "has_motorcycle_fullframe": fullframe["has_motorcycle"],
                    "num_vehicles_fullframe": fullframe["num_vehicles"],
                    "num_pedestrians_fullframe": fullframe["num_pedestrians"],
                    "num_motorcycles_fullframe": fullframe["num_motorcycles"],
                    "label_roi": roi["label"],
                    "has_vehicle_roi": roi["has_vehicle"],
                    "has_pedestrian_roi": roi["has_pedestrian"],
                    "has_motorcycle_roi": roi["has_motorcycle"],
                    "num_vehicles_roi": roi["num_vehicles"],
                    "num_pedestrians_roi": roi["num_pedestrians"],
                    "num_motorcycles_roi": roi["num_motorcycles"],
                    "image_available": image_available,
                    "annotation_available": True,
                    "valid_sample": image_available,
                    # Common Manifest contract fields (see
                    # docs/neurodriver_dataset_contract.md). Default label
                    # strategy is Full Frame until docs/decisions_log.md
                    # records a final choice.
                    "label": fullframe["label"],
                    "has_vehicle": fullframe["has_vehicle"],
                    "has_pedestrian": fullframe["has_pedestrian"],
                    "has_motorcycle": fullframe["has_motorcycle"],
                    "split": "TRAIN" if split == "train" else "TEST",
                }
            )
    return rows


def main() -> None:
    dataset_config = load_dataset_config()
    roi_config = ROIConfig.from_dict(dataset_config["roi"])

    if not any(p.exists() for p in LABEL_FILES.values()):
        print("[MISSING] No BDD100K annotation files found. See docs/bdd100k_setup.md.")
        print("[INFO] Nothing to build yet. Run this script again once data is placed.")
        return

    rows = build_rows(roi_config)
    if not rows:
        print("[WARNING] Annotation files were found but produced zero rows.")
        return

    df = pd.DataFrame(rows)

    # Preferred BDD scheme: official TRAIN -> internal TRAIN + VALIDATION,
    # official VAL -> academic TEST. Split is group-aware (by group_id) so a
    # video/run never crosses TRAIN/VALIDATION.
    train_mask = df["split"] == "TRAIN"
    val_ratio = 1.0 - dataset_config["splits"]["train_val_split_ratio"]
    df.loc[train_mask, "split"] = group_aware_split(
        df.loc[train_mask], group_col="group_id", val_ratio=val_ratio, seed=dataset_config["seed"]
    )

    violations = validate_manifest_invariants(df)
    if violations:
        print("[WARNING] Manifest invariant violations detected:")
        for v in violations:
            print(f"  - {v}")
    else:
        print("[OK] Manifest passed invariant validation.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
