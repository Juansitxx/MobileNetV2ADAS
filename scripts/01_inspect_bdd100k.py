"""Inspect the REAL BDD100K annotation schema before trusting parser assumptions.

Writes reports/schema_analysis.md. If annotation files are missing, writes a
report stating exactly that (never invents schema details) and exits 0 so
the rest of the pipeline can still be scaffolded.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LABEL_FILES = {
    "train": PROJECT_ROOT / "data" / "raw" / "bdd100k" / "labels" / "bdd100k_labels_images_train.json",
    "val": PROJECT_ROOT / "data" / "raw" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json",
}
REPORT_PATH = PROJECT_ROOT / "reports" / "schema_analysis.md"


def inspect_split(label_path: Path, split: str) -> dict | None:
    if not label_path.exists():
        print(f"[MISSING] {split}: {label_path}")
        return None

    with label_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"[OK] {split}: {len(records)} records loaded from {label_path.name}")

    top_level_keys: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    weather_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    timeofday_counts: Counter[str] = Counter()
    bbox_key_samples: set[str] = set()
    has_video_name = False
    sample_record = records[0] if records else {}

    for record in records:
        top_level_keys.update(record.keys())
        attrs = record.get("attributes", {}) or {}
        if "weather" in attrs:
            weather_counts[attrs["weather"]] += 1
        if "scene" in attrs:
            scene_counts[attrs["scene"]] += 1
        if "timeofday" in attrs:
            timeofday_counts[attrs["timeofday"]] += 1
        if record.get("videoName") or record.get("video_name"):
            has_video_name = True
        for obj in record.get("labels", []) or []:
            category_counts[obj.get("category", "<missing>")] += 1
            if "box2d" in obj:
                bbox_key_samples.update(obj["box2d"].keys())

    return {
        "split": split,
        "record_count": len(records),
        "top_level_keys": dict(top_level_keys),
        "category_counts": dict(category_counts.most_common()),
        "weather_counts": dict(weather_counts.most_common()),
        "scene_counts": dict(scene_counts.most_common()),
        "timeofday_counts": dict(timeofday_counts.most_common()),
        "bbox_keys": sorted(bbox_key_samples),
        "has_video_name": has_video_name,
        "sample_record_keys": sorted(sample_record.keys()) if sample_record else [],
    }


def render_report(results: dict[str, dict | None]) -> str:
    lines = ["# BDD100K schema analysis", ""]
    any_found = any(r is not None for r in results.values())

    if not any_found:
        lines += [
            "**No BDD100K annotation files were found.**",
            "",
            "See `docs/bdd100k_setup.md` for the official download and expected",
            "placement under `data/raw/bdd100k/labels/`. This report will be",
            "regenerated automatically once real files are placed and",
            "`scripts/01_inspect_bdd100k.py` is re-run.",
        ]
        return "\n".join(lines)

    for split, result in results.items():
        if result is None:
            lines += [f"## {split}", "", "**Missing.**", ""]
            continue
        lines += [
            f"## {split}",
            "",
            f"- Record count: {result['record_count']}",
            f"- Top-level keys: {sorted(result['top_level_keys'].keys())}",
            f"- Sample record keys: {result['sample_record_keys']}",
            f"- Has real video/group identifier (`videoName`): {result['has_video_name']}",
            f"- box2d keys observed: {result['bbox_keys']}",
            "",
            "### Category frequencies",
            "",
            "```json",
            json.dumps(result["category_counts"], indent=2),
            "```",
            "",
            "### Weather / Scene / Time-of-day",
            "",
            "```json",
            json.dumps(
                {
                    "weather": result["weather_counts"],
                    "scene": result["scene_counts"],
                    "timeofday": result["timeofday_counts"],
                },
                indent=2,
            ),
            "```",
            "",
        ]

    return "\n".join(lines)


def main() -> None:
    results = {split: inspect_split(path, split) for split, path in LABEL_FILES.items()}
    report = render_report(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"[OK] Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
