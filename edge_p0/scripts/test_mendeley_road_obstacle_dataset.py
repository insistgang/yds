from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


P0_CLASSES = {
    0: "road_obstacle",
    1: "stairs",
    2: "ramp",
    3: "blind_road_occupied",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitStats:
    images: int = 0
    labels: int = 0
    labeled_images: int = 0
    boxes: int = 0


@dataclass
class DatasetReport:
    root: str
    output_dir: str
    splits: dict[str, SplitStats] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    missing_label_files: list[str] = field(default_factory=list)
    label_without_image: list[str] = field(default_factory=list)
    invalid_label_lines: list[str] = field(default_factory=list)
    non_road_obstacle_labels: list[str] = field(default_factory=list)
    sampled_images: list[str] = field(default_factory=list)
    contact_sheet: str = ""

    @property
    def check_ok(self) -> bool:
        return (
            not self.label_without_image
            and not self.invalid_label_lines
            and not self.non_road_obstacle_labels
            and self.class_counts.get("road_obstacle", 0) > 0
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the converted Mendeley road_obstacle YOLO dataset and "
            "generate annotated preview images. This script does not train a model."
        )
    )
    parser.add_argument("--root", default="datasets/public_p0_yolo", help="Converted YOLO dataset root.")
    parser.add_argument(
        "--output-dir",
        default="runs/public_dataset_checks/mendeley_road_obstacle",
        help="Where annotated previews and reports are written.",
    )
    parser.add_argument("--sample-count", type=int, default=16, help="Number of labeled images to preview.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits to validate.")
    return parser.parse_args()


def list_images(images_dir: Path) -> dict[str, Path]:
    if not images_dir.exists():
        return {}
    return {
        path.stem: path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def parse_label_line(line: str, label_path: Path, line_no: int) -> tuple[int, float, float, float, float] | str:
    parts = line.split()
    if len(parts) != 5:
        return f"{label_path}:{line_no}: expected 5 fields, got {len(parts)}"
    try:
        class_id = int(parts[0])
        x_center, y_center, width, height = (float(value) for value in parts[1:])
    except ValueError as exc:
        return f"{label_path}:{line_no}: parse error: {exc}"
    if class_id not in P0_CLASSES:
        return f"{label_path}:{line_no}: class id {class_id} is not a LinkAble P0 class"
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return f"{label_path}:{line_no}: bbox center out of range"
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        return f"{label_path}:{line_no}: bbox size out of range"
    return class_id, x_center, y_center, width, height


def read_label_file(label_path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    errors: list[str] = []
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parsed = parse_label_line(line, label_path, line_no)
        if isinstance(parsed, str):
            errors.append(parsed)
        else:
            boxes.append(parsed)
    return boxes, errors


def draw_boxes(image_path: Path, label_path: Path, output_path: Path) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    height, width = image.shape[:2]
    boxes, _ = read_label_file(label_path)

    for class_id, x_center, y_center, box_width, box_height in boxes:
        x1 = int((x_center - box_width / 2.0) * width)
        y1 = int((y_center - box_height / 2.0) * height)
        x2 = int((x_center + box_width / 2.0) * width)
        y2 = int((y_center + box_height / 2.0) * height)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        color = (0, 220, 80) if class_id == 0 else (0, 128, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = P0_CLASSES[class_id]
        cv2.rectangle(image, (x1, max(0, y1 - 22)), (min(width - 1, x1 + 150), y1), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 4, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), image))


def make_contact_sheet(preview_paths: list[Path], output_path: Path, thumb_width: int) -> bool:
    thumbs: list[np.ndarray] = []
    for path in preview_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = thumb_width / max(1, width)
        resized = cv2.resize(image, (thumb_width, max(1, int(height * scale))))
        title_bar = np.full((28, thumb_width, 3), 245, dtype=np.uint8)
        cv2.putText(
            title_bar,
            path.name[:36],
            (6, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(np.vstack([title_bar, resized]))
    if not thumbs:
        return False

    cols = min(4, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    cell_height = max(thumb.shape[0] for thumb in thumbs)
    canvas = np.full((rows * cell_height, cols * thumb_width, 3), 255, dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        row = index // cols
        col = index % cols
        y = row * cell_height
        x = col * thumb_width
        canvas[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), canvas))


def validate_and_preview(args: argparse.Namespace) -> DatasetReport:
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "previews"
    report = DatasetReport(root=str(root), output_dir=str(output_dir))

    labeled_candidates: list[tuple[str, Path, Path]] = []
    requested_splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    for split in requested_splits:
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images = list_images(images_dir)
        labels = {path.stem: path for path in sorted(labels_dir.glob("*.txt"))} if labels_dir.exists() else {}

        split_stats = SplitStats(images=len(images), labels=len(labels))
        report.splits[split] = split_stats

        for stem, image_path in images.items():
            label_path = labels.get(stem)
            if label_path is None:
                report.missing_label_files.append(str(image_path))
                continue
            boxes, errors = read_label_file(label_path)
            report.invalid_label_lines.extend(errors)
            if boxes:
                split_stats.labeled_images += 1
                labeled_candidates.append((split, image_path, label_path))
            split_stats.boxes += len(boxes)
            for class_id, *_ in boxes:
                class_name = P0_CLASSES[class_id]
                report.class_counts[class_name] = report.class_counts.get(class_name, 0) + 1
                if class_id != 0:
                    report.non_road_obstacle_labels.append(str(label_path))

        for stem, label_path in labels.items():
            if stem not in images:
                report.label_without_image.append(str(label_path))

    rng = random.Random(args.seed)
    sample_count = min(args.sample_count, len(labeled_candidates))
    samples = rng.sample(labeled_candidates, sample_count) if sample_count else []

    preview_paths: list[Path] = []
    for index, (split, image_path, label_path) in enumerate(samples, start=1):
        output_path = preview_dir / f"{index:02d}_{split}_{image_path.stem}.jpg"
        if draw_boxes(image_path, label_path, output_path):
            preview_paths.append(output_path)
            report.sampled_images.append(str(output_path))

    contact_sheet = output_dir / "contact_sheet.jpg"
    if make_contact_sheet(preview_paths, contact_sheet, args.thumb_width):
        report.contact_sheet = str(contact_sheet)

    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = asdict(report)
    serializable["splits"] = {key: asdict(value) for key, value in report.splits.items()}
    serializable["check_ok"] = report.check_ok
    (output_dir / "report.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.txt").write_text(build_summary(report), encoding="utf-8")
    return report


def build_summary(report: DatasetReport) -> str:
    lines = [
        "Mendeley road_obstacle dataset check",
        f"ROOT={report.root}",
        f"OUTPUT_DIR={report.output_dir}",
    ]
    for split, stats in report.splits.items():
        lines.append(
            f"{split}: images={stats.images} labels={stats.labels} "
            f"labeled_images={stats.labeled_images} boxes={stats.boxes}"
        )
    for class_name, count in sorted(report.class_counts.items()):
        lines.append(f"{class_name}: {count}")
    lines.extend(
        [
            f"MISSING_LABEL_FILES={len(report.missing_label_files)}",
            f"LABEL_WITHOUT_IMAGE={len(report.label_without_image)}",
            f"INVALID_LABEL_LINES={len(report.invalid_label_lines)}",
            f"NON_ROAD_OBSTACLE_LABELS={len(set(report.non_road_obstacle_labels))}",
            f"SAMPLED_IMAGES={len(report.sampled_images)}",
            f"CONTACT_SHEET={report.contact_sheet}",
            f"CHECK_OK={'true' if report.check_ok else 'false'}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    report = validate_and_preview(args)
    print(build_summary(report), end="")
    return 0 if report.check_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
