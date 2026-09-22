"""Validate the Common Manifest(s): paths, labels, duplicates, split/group overlap.

Optional flags:
    --verify-images   open each image to confirm it decodes (slow).
    --check-hashes     detect exact-duplicate files via content hash (slow).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neurodriver_cnn.data.manifest import check_split_overlap, validate_manifest_invariants  # noqa: E402

MANIFEST_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "manifests" / "bdd100k_manifest.csv",
    PROJECT_ROOT / "data" / "processed" / "manifests" / "experiment_manifest.csv",
]


def validate_paths_exist(df: pd.DataFrame) -> list[str]:
    missing = df[~df["image_path"].apply(lambda p: Path(p).exists())]
    if len(missing) == 0:
        return []
    return [f"{len(missing)} image_path value(s) do not exist on disk"]


def validate_duplicate_ids(df: pd.DataFrame) -> list[str]:
    issues = []
    if "image_id" in df.columns:
        dup = df[df.duplicated(subset=["image_id"], keep=False)]
        if len(dup) > 0:
            issues.append(f"{dup['image_id'].nunique()} duplicate image_id value(s)")
    return issues


def verify_images_decode(df: pd.DataFrame) -> list[str]:
    from PIL import Image, UnidentifiedImageError

    bad = []
    for path in df["image_path"]:
        try:
            with Image.open(path) as img:
                img.verify()
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            bad.append(path)
    if not bad:
        return []
    return [f"{len(bad)} image(s) failed to decode"]


def check_hash_duplicates(df: pd.DataFrame) -> list[str]:
    hashes: dict[str, str] = {}
    duplicate_groups = 0
    for path in df["image_path"]:
        p = Path(path)
        if not p.exists():
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in hashes:
            duplicate_groups += 1
        else:
            hashes[digest] = path
    if duplicate_groups == 0:
        return []
    return [f"{duplicate_groups} exact-duplicate file(s) found by content hash"]


def validate_manifest_file(path: Path, verify_images: bool, check_hashes: bool) -> bool:
    if not path.exists():
        print(f"[MISSING] {path.name}: not found (skipped).")
        return True  # not a failure of validation itself; just nothing to check yet

    df = pd.read_csv(path)
    print(f"\n=== {path.name} ({len(df)} rows) ===")

    all_issues: list[str] = []
    all_issues += validate_manifest_invariants(df)
    all_issues += validate_paths_exist(df)
    all_issues += validate_duplicate_ids(df)
    if "split" in df.columns:
        all_issues += check_split_overlap(df, group_col="group_id" if "group_id" in df.columns else None)
    if verify_images:
        all_issues += verify_images_decode(df)
    if check_hashes:
        all_issues += check_hash_duplicates(df)

    if not all_issues:
        print("[OK] No issues found.")
        return True

    for issue in all_issues:
        print(f"[WARNING] {issue}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--check-hashes", action="store_true")
    args = parser.parse_args()

    all_ok = True
    for path in MANIFEST_PATHS:
        ok = validate_manifest_file(path, args.verify_images, args.check_hashes)
        all_ok = all_ok and ok

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
