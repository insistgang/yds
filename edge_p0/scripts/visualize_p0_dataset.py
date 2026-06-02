from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_NAMES = ["blind_road_occupied", "stairs", "ramp", "road_obstacle"]
COLORS = {
    0: (236, 84, 84),
    1: (242, 172, 69),
    2: (79, 161, 105),
    3: (72, 134, 213),
}


@dataclass(frozen=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class ImageRecord:
    split: str
    image_path: Path
    label_path: Path
    boxes: tuple[Box, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a LinkAble P0 YOLO dataset.")
    parser.add_argument("--dataset-root", default="edge_p0/datasets/p0_yolo")
    parser.add_argument("--output-dir", default="edge_p0/runs/dataset_visualization/p0_yolo")
    parser.add_argument("--samples-per-split", type=int, default=24)
    parser.add_argument("--samples-per-class", type=int, default=24)
    parser.add_argument("--thumb-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_names(dataset_root: Path) -> list[str]:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        return DEFAULT_NAMES

    try:
        import yaml  # type: ignore

        config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        names = config.get("names")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    except Exception:
        pass

    names: list[str] = []
    in_names = False
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names and line.startswith("-"):
            names.append(line[1:].strip())
        elif in_names and line and not line.startswith("#"):
            break
    return names or DEFAULT_NAMES


def label_path_for(image_path: Path, dataset_root: Path, split: str) -> Path:
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def read_boxes(label_path: Path, class_count: int) -> tuple[Box, ...]:
    boxes: list[Box] = []
    if not label_path.exists():
        return tuple(boxes)

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            if class_id < 0 or class_id >= class_count:
                continue
            x_center, y_center, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            continue
        boxes.append(Box(class_id, x_center, y_center, width, height))
    return tuple(boxes)


def load_records(dataset_root: Path, names: list[str]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        if not image_dir.exists():
            continue
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            label_path = label_path_for(image_path, dataset_root, split)
            records.append(
                ImageRecord(
                    split=split,
                    image_path=image_path,
                    label_path=label_path,
                    boxes=read_boxes(label_path, len(names)),
                )
            )
    return records


def safe_open_image(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def draw_record(record: ImageRecord, names: list[str], thumb_size: int) -> Image.Image | None:
    image = safe_open_image(record.image_path)
    if image is None:
        return None

    image = ImageOps.contain(image, (thumb_size, thumb_size))
    canvas = Image.new("RGB", (thumb_size, thumb_size + 46), (245, 247, 250))
    offset_x = (thumb_size - image.width) // 2
    offset_y = (thumb_size - image.height) // 2
    canvas.paste(image, (offset_x, offset_y))

    draw = ImageDraw.Draw(canvas)
    scale_x = image.width
    scale_y = image.height
    for box in record.boxes:
        color = COLORS.get(box.class_id, (180, 80, 180))
        x1 = offset_x + (box.x_center - box.width / 2) * scale_x
        y1 = offset_y + (box.y_center - box.height / 2) * scale_y
        x2 = offset_x + (box.x_center + box.width / 2) * scale_x
        y2 = offset_y + (box.y_center + box.height / 2) * scale_y
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = names[box.class_id] if box.class_id < len(names) else str(box.class_id)
        text_bbox = draw.textbbox((x1, y1), label)
        draw.rectangle((text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2), fill=color)
        draw.text((x1, y1), label, fill=(255, 255, 255))

    title = f"{record.split} | {record.image_path.name}"
    title = title[:56]
    draw.text((8, thumb_size + 6), title, fill=(20, 25, 34))
    draw.text((8, thumb_size + 24), f"boxes: {len(record.boxes)}", fill=(80, 86, 96))
    return canvas


def make_grid(records: list[ImageRecord], names: list[str], output_path: Path, thumb_size: int, columns: int = 4) -> bool:
    tiles = [tile for record in records if (tile := draw_record(record, names, thumb_size)) is not None]
    if not tiles:
        return False

    rows = (len(tiles) + columns - 1) // columns
    tile_width, tile_height = tiles[0].size
    gap = 12
    margin = 16
    grid = Image.new(
        "RGB",
        (columns * tile_width + (columns - 1) * gap + margin * 2, rows * tile_height + (rows - 1) * gap + margin * 2),
        (232, 236, 242),
    )
    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, columns)
        x = margin + col * (tile_width + gap)
        y = margin + row * (tile_height + gap)
        grid.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, quality=92)
    return True


def collect_stats(records: list[ImageRecord], names: list[str]) -> dict[str, object]:
    split_stats: dict[str, dict[str, object]] = {}
    overall_boxes = Counter()
    overall_images = Counter()

    for split in ("train", "val", "test"):
        split_records = [record for record in records if record.split == split]
        box_counts = Counter()
        image_counts = Counter()
        empty_labels = 0
        missing_labels = 0
        boxes_per_image: list[int] = []
        for record in split_records:
            if not record.label_path.exists():
                missing_labels += 1
            if not record.boxes:
                empty_labels += 1
            seen = set()
            for box in record.boxes:
                box_counts[names[box.class_id]] += 1
                overall_boxes[names[box.class_id]] += 1
                seen.add(names[box.class_id])
            for class_name in seen:
                image_counts[class_name] += 1
                overall_images[class_name] += 1
            boxes_per_image.append(len(record.boxes))

        split_stats[split] = {
            "images": len(split_records),
            "label_files": sum(1 for record in split_records if record.label_path.exists()),
            "missing_label_files": missing_labels,
            "empty_label_files": empty_labels,
            "boxes_by_class": {name: box_counts[name] for name in names},
            "images_by_class": {name: image_counts[name] for name in names},
            "boxes_per_image_avg": round(sum(boxes_per_image) / len(boxes_per_image), 3) if boxes_per_image else 0,
        }

    return {
        "total_images": len(records),
        "splits": split_stats,
        "overall_boxes_by_class": {name: overall_boxes[name] for name in names},
        "overall_images_by_class": {name: overall_images[name] for name in names},
    }


def write_summary_csv(stats: dict[str, object], names: list[str], output_path: Path) -> None:
    splits = stats["splits"]
    assert isinstance(splits, dict)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "images", "label_files", "missing_label_files", "empty_label_files", "boxes_per_image_avg", *names])
        for split, values in splits.items():
            assert isinstance(values, dict)
            boxes_by_class = values["boxes_by_class"]
            assert isinstance(boxes_by_class, dict)
            writer.writerow(
                [
                    split,
                    values["images"],
                    values["label_files"],
                    values["missing_label_files"],
                    values["empty_label_files"],
                    values["boxes_per_image_avg"],
                    *[boxes_by_class.get(name, 0) for name in names],
                ]
            )


def plot_split_counts(stats: dict[str, object], output_path: Path) -> None:
    splits = stats["splits"]
    assert isinstance(splits, dict)
    labels = list(splits.keys())
    values = [int(splits[split]["images"]) for split in labels]  # type: ignore[index]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, color="#4b84d1")
    plt.title("Images by Split")
    plt.ylabel("Images")
    for index, value in enumerate(values):
        plt.text(index, value, str(value), ha="center", va="bottom")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_class_distribution(stats: dict[str, object], names: list[str], output_path: Path) -> None:
    splits = stats["splits"]
    assert isinstance(splits, dict)
    x = range(len(names))
    width = 0.24

    plt.figure(figsize=(11, 5))
    for offset, split in enumerate(("train", "val", "test")):
        split_stats = splits.get(split, {})
        assert isinstance(split_stats, dict)
        boxes_by_class = split_stats.get("boxes_by_class", {})
        assert isinstance(boxes_by_class, dict)
        values = [int(boxes_by_class.get(name, 0)) for name in names]
        positions = [idx + (offset - 1) * width for idx in x]
        plt.bar(positions, values, width=width, label=split)

    plt.title("YOLO Boxes by Class")
    plt.ylabel("Boxes")
    plt.xticks(list(x), names, rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_boxes_per_image(records: list[ImageRecord], output_path: Path) -> None:
    values = [len(record.boxes) for record in records]
    max_boxes = max(values) if values else 0
    bins = min(max(max_boxes, 1), 30)

    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=bins, color="#4fa169", edgecolor="white")
    plt.title("Boxes per Image")
    plt.xlabel("Boxes")
    plt.ylabel("Images")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_index(output_dir: Path, stats: dict[str, object], image_files: list[Path]) -> None:
    rows = []
    splits = stats["splits"]
    assert isinstance(splits, dict)
    for split, values in splits.items():
        assert isinstance(values, dict)
        rows.append(
            "<tr>"
            f"<td>{split}</td>"
            f"<td>{values['images']}</td>"
            f"<td>{values['label_files']}</td>"
            f"<td>{values['empty_label_files']}</td>"
            f"<td>{values['boxes_per_image_avg']}</td>"
            "</tr>"
        )

    cards = "\n".join(
        f'<section><h2>{path.stem}</h2><img src="{path.name}" alt="{path.stem}"></section>'
        for path in image_files
        if path.exists()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LinkAble P0 Dataset Visualization</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #17202a; background: #f2f4f8; }}
    header {{ padding: 24px 32px; background: #17202a; color: white; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    table {{ border-collapse: collapse; background: white; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d5dbe5; padding: 8px 12px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    section {{ margin: 0 0 28px; padding: 18px; background: white; border: 1px solid #d5dbe5; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    code {{ background: #e9edf3; padding: 2px 5px; }}
  </style>
</head>
<body>
  <header>
    <h1>LinkAble P0 Dataset Visualization</h1>
    <p>Generated summary for <code>edge_p0/datasets/p0_yolo</code></p>
  </header>
  <main>
    <table>
      <thead><tr><th>split</th><th>images</th><th>label files</th><th>empty labels</th><th>avg boxes/image</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    {cards}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    names = read_names(dataset_root)
    records = load_records(dataset_root, names)
    if not records:
        raise SystemExit(f"No images found in {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = collect_stats(records, names)
    (output_dir / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(stats, names, output_dir / "summary.csv")

    image_outputs: list[Path] = []
    for output_name, plotter in (
        ("split_image_counts.png", lambda path: plot_split_counts(stats, path)),
        ("class_distribution_boxes.png", lambda path: plot_class_distribution(stats, names, path)),
        ("boxes_per_image_hist.png", lambda path: plot_boxes_per_image(records, path)),
    ):
        output_path = output_dir / output_name
        plotter(output_path)
        image_outputs.append(output_path)

    for split in ("train", "val", "test"):
        split_records = [record for record in records if record.split == split and record.boxes]
        rng.shuffle(split_records)
        output_path = output_dir / f"sample_grid_{split}.jpg"
        if make_grid(split_records[: args.samples_per_split], names, output_path, args.thumb_size):
            image_outputs.append(output_path)

    for class_id, class_name in enumerate(names):
        class_records = [record for record in records if any(box.class_id == class_id for box in record.boxes)]
        rng.shuffle(class_records)
        output_path = output_dir / f"class_grid_{class_name}.jpg"
        if make_grid(class_records[: args.samples_per_class], names, output_path, args.thumb_size):
            image_outputs.append(output_path)

    write_index(output_dir, stats, image_outputs)

    print(f"OUTPUT_DIR={output_dir}")
    print(f"INDEX={output_dir / 'index.html'}")
    print(f"SUMMARY_JSON={output_dir / 'summary.json'}")
    print(f"SUMMARY_CSV={output_dir / 'summary.csv'}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
