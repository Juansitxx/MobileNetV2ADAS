"""Audit the Common Manifest: distributions, Full Frame vs ROI, motorcycle slice.

Writes reports/dataset_audit.md and reports/dataset_audit.json, plus figures
under reports/figures/ when data exists. Reports "Pending" rather than
fabricating numbers when the manifest is absent or empty.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifests" / "bdd100k_manifest.csv"
REPORT_MD = PROJECT_ROOT / "reports" / "dataset_audit.md"
REPORT_JSON = PROJECT_ROOT / "reports" / "dataset_audit.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def _pending_report() -> None:
    text = (
        "# Dataset audit\n\n"
        "**Pending.** No manifest found at "
        f"`{MANIFEST_PATH.relative_to(PROJECT_ROOT)}`.\n\n"
        "Run `python scripts/02_build_manifest.py` after placing BDD100K data "
        "(see docs/bdd100k_setup.md), then re-run this script.\n"
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(text, encoding="utf-8")
    REPORT_JSON.write_text(json.dumps({"status": "pending"}, indent=2), encoding="utf-8")
    print("[MISSING] No manifest found. Wrote pending audit report.")


def _label_transition_matrix(df: pd.DataFrame) -> dict:
    return pd.crosstab(df["label_fullframe"], df["label_roi"]).to_dict()


def _plot_class_distribution(df: pd.DataFrame, column: str, out_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = df[column].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title(out_name.replace("_", " ").replace(".png", ""))
    ax.set_ylabel("count")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / out_name)
    plt.close(fig)


def _plot_fullframe_vs_roi(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparison = pd.DataFrame(
        {
            "Full Frame": df["label_fullframe"].value_counts(),
            "ADAS ROI": df["label_roi"].value_counts(),
        }
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    comparison.plot(kind="bar", ax=ax)
    ax.set_title("Full Frame vs ADAS ROI class distribution")
    ax.set_ylabel("count")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fullframe_vs_roi.png")
    plt.close(fig)


def _plot_motorcycle_distribution(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = df["has_motorcycle_fullframe"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Frames with >=1 motorcycle (Full Frame)")
    ax.set_ylabel("count")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "motorcycle_distribution.png")
    plt.close(fig)


def _plot_class_examples(df: pd.DataFrame, n_per_class: int = 2) -> None:
    """A small grid of real example images per Full Frame class.

    Per-object bounding boxes are not stored in the manifest (only
    aggregated counts/flags), so this shows the raw frame rather than
    drawing boxes — sufficient for Notebook 00's qualitative discussion.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    available = df[df["image_available"] == True]  # noqa: E712
    classes = ["CLEAR", "VEHICLE", "PEDESTRIAN", "MIXED"]
    samples = {c: available[available["label_fullframe"] == c].head(n_per_class) for c in classes}

    if all(len(s) == 0 for s in samples.values()):
        print("[INFO] Skipping class_examples.png: no accessible images.")
        return

    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(3 * n_per_class, 3 * len(classes)))
    for row, cls in enumerate(classes):
        rows = samples[cls]
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(rows):
                img_path = rows.iloc[col]["image_path"]
                try:
                    ax.imshow(Image.open(img_path))
                except (FileNotFoundError, OSError):
                    continue
            if col == 0:
                ax.set_title(cls, loc="left")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "class_examples.png")
    plt.close(fig)


def _plot_roi_examples(df: pd.DataFrame, roi_config: dict, n_examples: int = 4) -> None:
    """A few real frames with the ADAS ROI rectangle overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    available = df[df["image_available"] == True]  # noqa: E712
    samples = available.sample(n=min(n_examples, len(available)), random_state=42)
    if len(samples) == 0:
        print("[INFO] Skipping roi_examples.png: no accessible images.")
        return

    fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4))
    if len(samples) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, samples.iterrows()):
        ax.axis("off")
        try:
            img = Image.open(row["image_path"])
        except (FileNotFoundError, OSError):
            continue
        ax.imshow(img)
        w, h = row["width"], row["height"]
        rx1, ry1 = roi_config["x_min"] * w, roi_config["y_min"] * h
        rw, rh = (roi_config["x_max"] - roi_config["x_min"]) * w, (roi_config["y_max"] - roi_config["y_min"]) * h
        ax.add_patch(patches.Rectangle((rx1, ry1), rw, rh, linewidth=2, edgecolor="red", facecolor="none"))
        ax.set_title(f"FF={row['label_fullframe']} ROI={row['label_roi']}", fontsize=9)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "roi_examples.png")
    plt.close(fig)


def main() -> None:
    if not MANIFEST_PATH.exists():
        _pending_report()
        return

    df = pd.read_csv(MANIFEST_PATH)
    if len(df) == 0:
        _pending_report()
        return

    valid_df = df[df["valid_sample"] == True]  # noqa: E712
    missing_images = int((~df["image_available"]).sum())

    audit = {
        "status": "complete",
        "total_samples": int(len(df)),
        "valid_samples": int(len(valid_df)),
        "missing_or_invalid_images": missing_images,
        "resolutions": {f"{w}x{h}": int(n) for (w, h), n in df.groupby(["width", "height"]).size().items()},
        "fullframe_class_distribution": df["label_fullframe"].value_counts().to_dict(),
        "roi_class_distribution": df["label_roi"].value_counts().to_dict(),
        "label_transition_fullframe_to_roi": _label_transition_matrix(df),
        "motorcycle_frame_count_fullframe": int(df["has_motorcycle_fullframe"].sum()),
        "motorcycle_frame_count_roi": int(df["has_motorcycle_roi"].sum()),
        "weather_distribution": df["weather"].value_counts(dropna=True).to_dict()
        if "weather" in df
        else {},
        "scene_distribution": df["scene"].value_counts(dropna=True).to_dict()
        if "scene" in df
        else {},
        "timeofday_distribution": df["timeofday"].value_counts(dropna=True).to_dict()
        if "timeofday" in df
        else {},
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")

    md_lines = [
        "# Dataset audit",
        "",
        f"- Total samples: {audit['total_samples']}",
        f"- Valid samples: {audit['valid_samples']}",
        f"- Missing/invalid images: {audit['missing_or_invalid_images']}",
        "",
        "## Full Frame class distribution",
        "",
        "```json",
        json.dumps(audit["fullframe_class_distribution"], indent=2),
        "```",
        "",
        "## ADAS ROI class distribution",
        "",
        "```json",
        json.dumps(audit["roi_class_distribution"], indent=2),
        "```",
        "",
        "## Motorcycle representation",
        "",
        f"- Frames with >=1 motorcycle (Full Frame): {audit['motorcycle_frame_count_fullframe']}",
        f"- Frames with >=1 motorcycle (ROI): {audit['motorcycle_frame_count_roi']}",
        "",
        "## Limitations",
        "",
        "- BDD100K is the SOURCE domain (US driving); it is not Colombian data.",
        "  Class/weather/scene distributions above describe BDD100K only and",
        "  should not be assumed to transfer to Colombian NeuroDriver footage.",
        "- Full Frame vs ROI strategy has not been finalized; see",
        "  `docs/decisions_log.md`.",
    ]
    REPORT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    from neurodriver_cnn.config import load_dataset_config

    _plot_class_distribution(df, "label_fullframe", "class_distribution_fullframe.png")
    _plot_class_distribution(df, "label_roi", "class_distribution_roi.png")
    _plot_fullframe_vs_roi(df)
    _plot_motorcycle_distribution(df)
    _plot_class_examples(df)
    _plot_roi_examples(df, load_dataset_config()["roi"])

    print(f"[OK] Wrote {REPORT_MD} and {REPORT_JSON}")
    print(f"[OK] Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
