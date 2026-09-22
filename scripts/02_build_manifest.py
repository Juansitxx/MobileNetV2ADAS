"""Build data/processed/manifests/bdd100k_manifest.csv from the real
BDD100K Images 10K (DatasetNinja, Supervisely format) archive.

One command builds the manifest end to end: extracts the ``.tar`` (once,
idempotent) and parses every ``train``/``val`` annotation. The DatasetNinja
``test`` split is intentionally never read here — real inspection showed it
has zero annotated objects in every record (see docs/decisions_log.md,
2026-09-22) and no usable ground truth. Never invents rows, group IDs, or
dimensions beyond what's in the real annotation JSON.

Usage:
    python scripts/02_build_manifest.py [--tar PATH] [--extract-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neurodriver_cnn.config import load_dataset_config  # noqa: E402
from neurodriver_cnn.data.bdd100k import (  # noqa: E402
    extract_supervisely_tar,
    find_split_dir,
    iter_frames_from_split_dir,
)
from neurodriver_cnn.data.manifest import group_aware_split, validate_manifest_invariants  # noqa: E402
from neurodriver_cnn.labeling.frame_labels import classify_fullframe, classify_roi  # noqa: E402
from neurodriver_cnn.labeling.roi import ROIConfig  # noqa: E402

DEFAULT_TAR_PATH = PROJECT_ROOT / "data" / "raw" / "bdd100k" / "bdd100k_10k_supervisely.tar"
DEFAULT_EXTRACT_DIR = PROJECT_ROOT / "data" / "raw" / "bdd100k" / "extracted"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "manifests" / "bdd100k_manifest.csv"

# DatasetNinja raw split -> internal split rule (see docs/decisions_log.md, 2026-09-22):
# raw "train" -> internal TRAIN + VALIDATION (group-aware split below)
# raw "val"   -> academic TEST
# raw "test"  -> excluded entirely (no usable ground truth)
RAW_SPLITS_TO_LOAD = ["train", "val"]


def build_rows(extract_dir: Path, roi_config: ROIConfig) -> list[dict]:
    rows: list[dict] = []
    for raw_split in RAW_SPLITS_TO_LOAD:
        split_dir = find_split_dir(extract_dir, raw_split)
        if split_dir is None:
            print(f"[WARNING] Could not locate a '{raw_split}' split directory under {extract_dir}")
            continue

        img_dir = split_dir / "img"
        for ann in iter_frames_from_split_dir(split_dir, split=raw_split):
            image_path = img_dir / ann.image_id
            image_available = image_path.exists()

            fullframe = classify_fullframe(ann.boxes)
            roi = classify_roi(ann.boxes, ann.width, ann.height, roi_config)

            rows.append(
                {
                    "image_id": ann.image_id,
                    "image_path": str(image_path),
                    "source_dataset": "BDD100K_Images_10K_DatasetNinja",
                    "raw_split": raw_split,
                    "group_id": ann.group_id,  # always None: no real sequence id in this dataset
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
                    # records a final choice. Models train on has_vehicle/
                    # has_pedestrian directly (multi-label), not on `label`.
                    "label": fullframe["label"],
                    "has_vehicle": fullframe["has_vehicle"],
                    "has_pedestrian": fullframe["has_pedestrian"],
                    "has_motorcycle": fullframe["has_motorcycle"],
                    "split": "TRAIN" if raw_split == "train" else "TEST",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", type=Path, default=DEFAULT_TAR_PATH)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    args = parser.parse_args()

    if not args.tar.exists():
        print(f"[MISSING] {args.tar}. See docs/bdd100k_setup.md.")
        print("[INFO] Nothing to build yet. Run this script again once the archive is placed.")
        return

    dataset_config = load_dataset_config()
    roi_config = ROIConfig.from_dict(dataset_config["roi"])

    print(f"[INFO] Extracting (idempotent) {args.tar} -> {args.extract_dir}")
    extract_dir = extract_supervisely_tar(args.tar, args.extract_dir)

    rows = build_rows(extract_dir, roi_config)
    if not rows:
        print("[WARNING] Extraction succeeded but produced zero rows. Check the archive layout.")
        return

    df = pd.DataFrame(rows)

    # Preferred scheme: raw "train" -> internal TRAIN + VALIDATION, raw "val"
    # -> academic TEST. Split is group-aware, but this dataset has no real
    # group_id (independent frames), so it falls back to a seeded random
    # per-row split (see neurodriver_cnn.data.manifest.group_aware_split).
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
    print(f"[INFO] Split counts: {df['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
