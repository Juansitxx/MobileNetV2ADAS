"""Inspect the REAL Supervisely annotation schema before trusting parser assumptions.

Reads directly from the ``.tar`` (no extraction needed for inspection).
Writes reports/schema_analysis.md. If the archive is missing, writes a
report stating exactly that (never invents schema details) and exits 0 so
the rest of the pipeline can still be scaffolded.
"""

from __future__ import annotations

import json
import tarfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_TAR_PATH = PROJECT_ROOT / "data" / "raw" / "bdd100k" / "bdd100k_10k_supervisely.tar"
REPORT_PATH = PROJECT_ROOT / "reports" / "schema_analysis.md"
SPLITS = ["train", "val", "test"]


def _split_of(member_name: str) -> str | None:
    parts = Path(member_name).parts
    for split in SPLITS:
        if split in (p.lower() for p in parts):
            return split
    return None


def inspect_tar(tar_path: Path) -> dict[str, dict]:
    results: dict[str, dict] = {
        split: {
            "record_count": 0,
            "records_with_objects": 0,
            "geometry_type_counts": Counter(),
            "category_counts": Counter(),
            "tag_name_counts": Counter(),
            "sample_top_level_keys": None,
        }
        for split in SPLITS
    }

    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not member.isfile() or "/ann/" not in f"/{member.name}" or not member.name.endswith(".json"):
                continue
            split = _split_of(member.name)
            if split is None:
                continue

            f = tar.extractfile(member)
            if f is None:
                continue
            record = json.loads(f.read().decode("utf-8"))

            bucket = results[split]
            bucket["record_count"] += 1
            if bucket["sample_top_level_keys"] is None:
                bucket["sample_top_level_keys"] = sorted(record.keys())

            objects = record.get("objects", []) or []
            if objects:
                bucket["records_with_objects"] += 1
            for obj in objects:
                bucket["geometry_type_counts"][obj.get("geometryType", "<missing>")] += 1
                bucket["category_counts"][obj.get("classTitle", "<missing>")] += 1
            for tag in record.get("tags", []) or []:
                bucket["tag_name_counts"][tag.get("name", "<missing>")] += 1

    return results


def render_report(results: dict[str, dict] | None) -> str:
    lines = ["# BDD100K (Supervisely) schema analysis", ""]

    if results is None:
        lines += [
            "**No Supervisely dataset archive was found.**",
            "",
            "See `docs/bdd100k_setup.md` for the official download and expected",
            f"placement. This report will be regenerated automatically once the",
            "`.tar` is placed and `scripts/01_inspect_bdd100k.py` is re-run.",
        ]
        return "\n".join(lines)

    for split, r in results.items():
        lines += [
            f"## {split}",
            "",
            f"- Record count: {r['record_count']}",
            f"- Records with >=1 object: {r['records_with_objects']}",
            f"- Sample top-level keys: {r['sample_top_level_keys']}",
            "",
            "### Geometry types",
            "",
            "```json",
            json.dumps(dict(r["geometry_type_counts"].most_common()), indent=2),
            "```",
            "",
            "### Category (classTitle) frequencies",
            "",
            "```json",
            json.dumps(dict(r["category_counts"].most_common()), indent=2),
            "```",
            "",
            "### Tag names observed",
            "",
            "```json",
            json.dumps(dict(r["tag_name_counts"].most_common()), indent=2),
            "```",
            "",
        ]

    test_objects = results.get("test", {}).get("records_with_objects", None)
    lines += [
        "## Note on the 'test' split",
        "",
        (
            f"`test` has {test_objects} record(s) with >=1 object out of "
            f"{results['test']['record_count']} total — confirms it carries no usable "
            "ground truth (see docs/decisions_log.md, 2026-09-22). It is excluded from "
            "manifest building and all supervised evaluation."
            if test_objects is not None
            else "Not inspected."
        ),
    ]

    return "\n".join(lines)


def main() -> None:
    if not RAW_TAR_PATH.exists():
        print(f"[MISSING] {RAW_TAR_PATH}")
        report = render_report(None)
    else:
        print(f"[OK] Inspecting {RAW_TAR_PATH} ...")
        results = inspect_tar(RAW_TAR_PATH)
        for split, r in results.items():
            print(f"[OK] {split}: {r['record_count']} records, {r['records_with_objects']} with objects")
        report = render_report(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"[OK] Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
