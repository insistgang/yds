from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


P0_CLASSES = {
    "road_obstacle": 0,
    "stairs": 1,
    "ramp": 2,
    "blind_road_occupied": 3,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


CATALOG = [
    {
        "id": "zenodo_accessibility_barriers",
        "priority": 1,
        "name": "Image Dataset of Accessibility Barriers",
        "url": "https://zenodo.org/records/6382090",
        "license": "CC BY 4.0",
        "source_format": "CVAT XML + images",
        "recommended_p0_use": ["stairs", "ramp"],
        "mapping": {"stairs": "stairs", "ramps": "ramp", "ramp": "ramp"},
        "notes": "Optional steps->stairs requires manual review.",
    },
    {
        "id": "mendeley_obstacles_avoidance",
        "priority": 2,
        "name": "Obstacles Avoidance Assistance for Visually Impaired",
        "url": "https://data.mendeley.com/datasets/xwhnp82rhk/1",
        "license": "CC BY 4.0",
        "source_format": "YOLO",
        "recommended_p0_use": ["road_obstacle"],
        "mapping": {
            "pole": "road_obstacle",
            "fence": "road_obstacle",
            "bump": "road_obstacle",
            "hole": "road_obstacle",
        },
        "notes": "Collapse source obstacle classes to road_obstacle after QA.",
    },
    {
        "id": "roboflow_tactile_paving_blind_assist",
        "priority": 3,
        "name": "Tactile-Paving-Blind-Assist",
        "url": "https://universe.roboflow.com/labeling-qcirk/tactile-paving-blind-assist",
        "license": "CC BY 4.0",
        "source_format": "Roboflow object detection",
        "recommended_p0_use": ["blind_road_context_only"],
        "mapping": {},
        "notes": "Tactile paving context is not blind_road_occupied. Do not map directly.",
    },
    {
        "id": "roboflow_stairs_detection",
        "priority": 4,
        "name": "Roboflow Universe Stairs_Detection candidates",
        "url": "https://universe.roboflow.com/stair-detection-itfjj",
        "license": "check_each_project_version_before_use",
        "source_format": "Roboflow object detection",
        "recommended_p0_use": ["stairs"],
        "mapping": {"stairs": "stairs", "upstair": "review_optional_stairs", "downstair": "review_optional_stairs"},
        "notes": "Project quality and license can vary; confirm version license before use.",
    },
    {
        "id": "github_obstacle_dataset_candidate",
        "priority": 5,
        "name": "TW0521/Obstacle-Dataset",
        "url": "https://github.com/TW0521/Obstacle-Dataset",
        "license": "unclear",
        "source_format": "VOC + YOLO",
        "recommended_p0_use": ["road_obstacle_candidate_only"],
        "mapping": {
            "reflective_cone": "road_obstacle",
            "warning_column": "road_obstacle",
            "spherical_roadblock": "road_obstacle",
            "pole": "road_obstacle",
            "ashcan": "road_obstacle",
        },
        "notes": "Do not use in official baseline until license is confirmed.",
    },
    {
        "id": "huggingface_projectsidewalk_rampnet",
        "priority": 6,
        "name": "Project Sidewalk RampNet Dataset",
        "url": "https://huggingface.co/datasets/projectsidewalk/rampnet-dataset",
        "license": "MIT",
        "source_format": "Hugging Face parquet",
        "recommended_p0_use": ["ramp_research_candidate"],
        "mapping": {"curb_ramp_points_normalized": "requires_custom_conversion_to_ramp"},
        "notes": "Large street-view dataset; not first priority for Jetson P0 baseline.",
    },
]


@dataclass(slots=True)
class ConversionStats:
    images_seen: int = 0
    images_written: int = 0
    boxes_written: int = 0
    boxes_skipped: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare public datasets for LinkAble P0 YOLO experiments without adding P1 classes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create public dataset staging directories and catalog.")
    init_parser.add_argument("--root", default="datasets/public_p0_sources")

    catalog_parser = subparsers.add_parser("catalog", help="Print or write the public dataset catalog.")
    catalog_parser.add_argument("--output", default="")

    cvat_parser = subparsers.add_parser("convert-cvat", help="Convert CVAT XML boxes to P0 YOLO labels.")
    cvat_parser.add_argument("--xml", required=True)
    cvat_parser.add_argument("--image-root", required=True)
    cvat_parser.add_argument("--output-root", default="datasets/public_p0_yolo")
    cvat_parser.add_argument("--source-id", default="zenodo_accessibility_barriers")
    cvat_parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    cvat_parser.add_argument("--include-steps-as-stairs", action="store_true")
    cvat_parser.add_argument("--dry-run", action="store_true")

    yolo_parser = subparsers.add_parser("convert-yolo", help="Remap a YOLO dataset to LinkAble P0 labels.")
    yolo_parser.add_argument("--images-dir", required=True)
    yolo_parser.add_argument("--labels-dir", required=True)
    yolo_parser.add_argument("--class-names", required=True, help="Comma-separated source class names by id.")
    yolo_parser.add_argument("--map", action="append", default=[], help="Mapping like pole=road_obstacle.")
    yolo_parser.add_argument("--output-root", default="datasets/public_p0_yolo")
    yolo_parser.add_argument("--source-id", required=True)
    yolo_parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    yolo_parser.add_argument("--dry-run", action="store_true")

    yaml_parser = subparsers.add_parser("write-data-yaml", help="Write P0-only YOLO data.yaml.")
    yaml_parser.add_argument("--output-root", default="datasets/public_p0_yolo")

    summary_parser = subparsers.add_parser("summary", help="Count labels in a converted P0 YOLO dataset.")
    summary_parser.add_argument("--root", default="datasets/public_p0_yolo")

    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def init_dirs(root: Path) -> None:
    for rel in ("raw", "converted_yolo", "_manifests"):
        target = root / rel
        target.mkdir(parents=True, exist_ok=True)
        keep = target / ".gitkeep"
        keep.touch(exist_ok=True)
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Public P0 Dataset Staging\n\n"
            "Public datasets are staged here for cold-start experiments only. "
            "Do not commit downloaded images or derived labels.\n",
            encoding="utf-8",
        )
    write_json(root / "sources.json", CATALOG)


def p0_label_id(label: str) -> int:
    if label not in P0_CLASSES:
        raise ValueError(f"not a P0 label: {label}")
    return P0_CLASSES[label]


def ensure_yolo_dirs(root: Path, split: str) -> tuple[Path, Path]:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def find_image(image_root: Path, name: str) -> Path | None:
    candidates = [image_root / name, image_root / Path(name).name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    stem = Path(name).stem
    for suffix in IMAGE_SUFFIXES:
        matches = list(image_root.rglob(stem + suffix))
        if matches:
            return matches[0]
    return None


def yolo_box_from_cvat(box: ET.Element, width: float, height: float) -> tuple[float, float, float, float]:
    xtl = float(box.attrib["xtl"])
    ytl = float(box.attrib["ytl"])
    xbr = float(box.attrib["xbr"])
    ybr = float(box.attrib["ybr"])
    x_center = ((xtl + xbr) / 2.0) / width
    y_center = ((ytl + ybr) / 2.0) / height
    box_width = (xbr - xtl) / width
    box_height = (ybr - ytl) / height
    return x_center, y_center, box_width, box_height


def convert_cvat(args: argparse.Namespace) -> ConversionStats:
    xml_path = Path(args.xml)
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)
    image_out, label_out = ensure_yolo_dirs(output_root, args.split)

    label_map = {"stairs": "stairs", "stair": "stairs", "ramps": "ramp", "ramp": "ramp"}
    if args.include_steps_as_stairs:
        label_map.update({"steps": "stairs", "step": "stairs"})

    tree = ET.parse(xml_path)
    root = tree.getroot()
    stats = ConversionStats()

    for image in root.findall("image"):
        stats.images_seen += 1
        name = image.attrib["name"]
        width = float(image.attrib["width"])
        height = float(image.attrib["height"])
        yolo_lines: list[str] = []

        for box in image.findall("box"):
            source_label = box.attrib.get("label", "").strip()
            p0_label = label_map.get(source_label)
            if not p0_label:
                stats.boxes_skipped += 1
                continue
            values = yolo_box_from_cvat(box, width, height)
            class_id = p0_label_id(p0_label)
            yolo_lines.append(
                f"{class_id} " + " ".join(f"{value:.6f}" for value in values)
            )

        if not yolo_lines:
            continue

        source_image = find_image(image_root, name)
        if source_image is None:
            print(f"[WARN] image not found for annotation: {name}")
            continue

        target_name = f"{args.source_id}_{Path(source_image).name}"
        image_target = image_out / target_name
        label_target = label_out / (Path(target_name).stem + ".txt")

        if not args.dry_run:
            shutil.copy2(source_image, image_target)
            label_target.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        stats.images_written += 1
        stats.boxes_written += len(yolo_lines)

    return stats


def parse_map(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--map must be source=p0_label, got: {value}")
        source, target = value.split("=", 1)
        target = target.strip()
        if target not in P0_CLASSES:
            raise SystemExit(f"mapping target must be one of {sorted(P0_CLASSES)}, got: {target}")
        result[source.strip()] = target
    return result


def find_label_for_image(labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / (image_path.stem + ".txt")


def convert_yolo(args: argparse.Namespace) -> ConversionStats:
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_root = Path(args.output_root)
    image_out, label_out = ensure_yolo_dirs(output_root, args.split)
    class_names = [name.strip() for name in args.class_names.split(",")]
    source_to_p0 = parse_map(args.map)
    stats = ConversionStats()

    for image_path in sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
        stats.images_seen += 1
        label_path = find_label_for_image(labels_dir, image_path)
        if not label_path.exists():
            continue

        out_lines: list[str] = []
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5:
                stats.boxes_skipped += 1
                continue
            try:
                source_id = int(float(parts[0]))
                source_name = class_names[source_id]
            except (ValueError, IndexError):
                stats.boxes_skipped += 1
                continue
            p0_label = source_to_p0.get(source_name)
            if not p0_label:
                stats.boxes_skipped += 1
                continue
            out_lines.append(" ".join([str(p0_label_id(p0_label)), *parts[1:5]]))

        if not out_lines:
            continue

        target_name = f"{args.source_id}_{image_path.name}"
        image_target = image_out / target_name
        label_target = label_out / (Path(target_name).stem + ".txt")
        if not args.dry_run:
            shutil.copy2(image_path, image_target)
            label_target.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        stats.images_written += 1
        stats.boxes_written += len(out_lines)

    return stats


def write_data_yaml(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {idx}: {name}" for name, idx in P0_CLASSES.items())
    text = (
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names}\n"
    )
    target = output_root / "data.yaml"
    target.write_text(text, encoding="utf-8")
    return target


def summarize_yolo(root: Path) -> dict[str, int]:
    counts = {name: 0 for name in P0_CLASSES}
    labels_root = root / "labels"
    if not labels_root.exists():
        return counts
    id_to_name = {idx: name for name, idx in P0_CLASSES.items()}
    for label_file in labels_root.rglob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            name = id_to_name.get(class_id)
            if name:
                counts[name] += 1
    return counts


def print_stats(stats: ConversionStats) -> None:
    print(f"IMAGES_SEEN={stats.images_seen}")
    print(f"IMAGES_WRITTEN={stats.images_written}")
    print(f"BOXES_WRITTEN={stats.boxes_written}")
    print(f"BOXES_SKIPPED={stats.boxes_skipped}")


def main() -> int:
    args = parse_args()
    if args.command == "init":
        init_dirs(Path(args.root))
        print(f"PUBLIC_P0_SOURCES_READY={Path(args.root)}")
        return 0
    if args.command == "catalog":
        if args.output:
            write_json(Path(args.output), CATALOG)
            print(f"CATALOG_WRITTEN={args.output}")
        else:
            print(json.dumps(CATALOG, ensure_ascii=False, indent=2))
        return 0
    if args.command == "convert-cvat":
        stats = convert_cvat(args)
        print_stats(stats)
        return 0
    if args.command == "convert-yolo":
        stats = convert_yolo(args)
        print_stats(stats)
        return 0
    if args.command == "write-data-yaml":
        target = write_data_yaml(Path(args.output_root))
        print(f"DATA_YAML_WRITTEN={target}")
        return 0
    if args.command == "summary":
        counts = summarize_yolo(Path(args.root))
        for name in P0_CLASSES:
            print(f"{name}: {counts[name]}")
        return 0
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
