from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_NAMES = ["blind_road_occupied", "stairs", "ramp", "road_obstacle"]
COLORS = {
    0: (230, 74, 74),
    1: (232, 146, 44),
    2: (63, 151, 91),
    3: (54, 119, 207),
}


@dataclass(frozen=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class Candidate:
    class_id: int
    split: str
    image_path: Path
    label_path: Path
    boxes: tuple[Box, ...]
    target_box: Box
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select clear poster samples from the P0 YOLO dataset.")
    parser.add_argument("--dataset-root", default="edge_p0/datasets/p0_yolo")
    parser.add_argument("--output-dir", default="edge_p0/runs/poster_samples/p0_yolo")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--thumb-size", type=int, default=360)
    parser.add_argument("--min-area", type=float, default=0.0, help="Minimum normalized target bbox area.")
    parser.add_argument("--max-area", type=float, default=1.0, help="Maximum normalized target bbox area.")
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
    return DEFAULT_NAMES


def read_boxes(label_path: Path, class_count: int) -> tuple[Box, ...]:
    if not label_path.exists():
        return ()

    boxes: list[Box] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
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


def iter_images(dataset_root: Path) -> list[tuple[str, Path, Path, tuple[Box, ...]]]:
    names = read_names(dataset_root)
    records: list[tuple[str, Path, Path, tuple[Box, ...]]] = []
    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.exists():
            continue
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            label_path = label_dir / f"{image_path.stem}.txt"
            boxes = read_boxes(label_path, len(names))
            if boxes:
                records.append((split, image_path, label_path, boxes))
    return records


def centeredness(box: Box) -> float:
    distance = math.dist((box.x_center, box.y_center), (0.5, 0.5))
    return max(0.0, 1.0 - distance / math.sqrt(0.5))


def score_candidate(target_box: Box, boxes: tuple[Box, ...]) -> float:
    same_class = [box for box in boxes if box.class_id == target_box.class_id]
    other_class_count = len(boxes) - len(same_class)
    target_area = target_box.area
    same_class_area = sum(box.area for box in same_class)

    # Poster samples read best when the target is large, near center, and not buried in many labels.
    size_score = min(target_area, 0.5) * 1000.0
    class_area_score = min(same_class_area, 0.75) * 160.0
    center_score = centeredness(target_box) * 70.0
    crowd_penalty = max(0, len(boxes) - 1) * 7.0
    other_class_penalty = other_class_count * 18.0
    tiny_penalty = 60.0 if target_area < 0.035 else 0.0
    edge_penalty = 30.0 if target_box.x_center < 0.12 or target_box.x_center > 0.88 else 0.0
    return size_score + class_area_score + center_score - crowd_penalty - other_class_penalty - tiny_penalty - edge_penalty


def select_candidates(
    dataset_root: Path,
    names: list[str],
    min_area: float = 0.0,
    max_area: float = 1.0,
) -> dict[int, list[Candidate]]:
    candidates: dict[int, list[Candidate]] = {class_id: [] for class_id in range(len(names))}
    fallback_candidates: dict[int, list[Candidate]] = {class_id: [] for class_id in range(len(names))}
    for split, image_path, label_path, boxes in iter_images(dataset_root):
        for class_id in range(len(names)):
            class_boxes = [box for box in boxes if box.class_id == class_id]
            if not class_boxes:
                continue
            target_box = max(class_boxes, key=lambda box: box.area)
            candidate = Candidate(
                class_id=class_id,
                split=split,
                image_path=image_path,
                label_path=label_path,
                boxes=boxes,
                target_box=target_box,
                score=score_candidate(target_box, boxes),
            )
            fallback_candidates[class_id].append(candidate)
            if min_area <= target_box.area <= max_area:
                candidates[class_id].append(candidate)

    for class_id in candidates:
        if not candidates[class_id]:
            candidates[class_id] = fallback_candidates[class_id]
        candidates[class_id].sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def draw_boxes(image: Image.Image, boxes: tuple[Box, ...], names: list[str], highlight_class_id: int | None = None) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    for box in boxes:
        color = COLORS.get(box.class_id, (160, 80, 180))
        line_width = 6 if box.class_id == highlight_class_id else 4
        x1 = (box.x_center - box.width / 2) * width
        y1 = (box.y_center - box.height / 2) * height
        x2 = (box.x_center + box.width / 2) * width
        y2 = (box.y_center + box.height / 2) * height
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = names[box.class_id] if box.class_id < len(names) else str(box.class_id)
        text_bbox = draw.textbbox((x1, y1), label)
        draw.rectangle((text_bbox[0] - 4, text_bbox[1] - 4, text_bbox[2] + 4, text_bbox[3] + 4), fill=color)
        draw.text((x1, y1), label, fill=(255, 255, 255))
    return output


def save_candidate_files(candidate: Candidate, names: list[str], output_dir: Path, ordinal: int) -> dict[str, Path]:
    class_name = names[candidate.class_id]
    prefix = f"{ordinal:02d}_{class_name}"
    original_path = output_dir / f"{prefix}_original{candidate.image_path.suffix.lower()}"
    annotated_path = output_dir / f"{prefix}_annotated.jpg"

    shutil.copy2(candidate.image_path, original_path)
    image = open_rgb(candidate.image_path)
    annotated = draw_boxes(image, candidate.boxes, names, highlight_class_id=candidate.class_id)
    annotated.save(annotated_path, quality=94)
    return {"original": original_path, "annotated": annotated_path}


def make_candidate_grid(candidates: list[Candidate], names: list[str], output_path: Path, thumb_size: int) -> None:
    tiles: list[Image.Image] = []
    for idx, candidate in enumerate(candidates, start=1):
        image = open_rgb(candidate.image_path)
        annotated = draw_boxes(image, candidate.boxes, names, highlight_class_id=candidate.class_id)
        annotated = ImageOps.contain(annotated, (thumb_size, thumb_size))
        canvas = Image.new("RGB", (thumb_size, thumb_size + 58), (246, 248, 251))
        canvas.paste(annotated, ((thumb_size - annotated.width) // 2, (thumb_size - annotated.height) // 2))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb_size + 8), f"#{idx} score={candidate.score:.1f} {candidate.split}", fill=(20, 25, 34))
        draw.text((8, thumb_size + 28), candidate.image_path.name[:50], fill=(85, 91, 101))
        tiles.append(canvas)

    if not tiles:
        return

    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    gap = 12
    margin = 16
    tile_width, tile_height = tiles[0].size
    grid = Image.new(
        "RGB",
        (columns * tile_width + (columns - 1) * gap + margin * 2, rows * tile_height + (rows - 1) * gap + margin * 2),
        (231, 236, 244),
    )
    for idx, tile in enumerate(tiles):
        row, col = divmod(idx, columns)
        grid.paste(tile, (margin + col * (tile_width + gap), margin + row * (tile_height + gap)))
    grid.save(output_path, quality=92)


def make_four_panel(selected: list[Candidate], names: list[str], output_path: Path, annotated: bool) -> None:
    panel_w, panel_h = 720, 540
    header_h = 54
    gap = 18
    margin = 24
    canvas = Image.new("RGB", (panel_w * 2 + gap + margin * 2, (panel_h + header_h) * 2 + gap + margin * 2), (239, 243, 249))
    for idx, candidate in enumerate(selected):
        image = open_rgb(candidate.image_path)
        if annotated:
            image = draw_boxes(image, candidate.boxes, names, highlight_class_id=candidate.class_id)
        image = ImageOps.contain(image, (panel_w, panel_h))
        panel = Image.new("RGB", (panel_w, panel_h + header_h), (255, 255, 255))
        panel.paste(image, ((panel_w - image.width) // 2, header_h + (panel_h - image.height) // 2))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel_w, header_h), fill=COLORS.get(candidate.class_id, (80, 80, 80)))
        draw.text((18, 17), names[candidate.class_id], fill=(255, 255, 255))
        row, col = divmod(idx, 2)
        x = margin + col * (panel_w + gap)
        y = margin + row * (panel_h + header_h + gap)
        canvas.paste(panel, (x, y))
    canvas.save(output_path, quality=94)


def write_manifest(selected: list[Candidate], names: list[str], copied_paths: dict[int, dict[str, Path]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class",
                "split",
                "score",
                "target_box_area",
                "image_path",
                "label_path",
                "poster_original",
                "poster_annotated",
            ]
        )
        for candidate in selected:
            paths = copied_paths[candidate.class_id]
            writer.writerow(
                [
                    names[candidate.class_id],
                    candidate.split,
                    f"{candidate.score:.3f}",
                    f"{candidate.target_box.area:.5f}",
                    candidate.image_path,
                    candidate.label_path,
                    paths["original"],
                    paths["annotated"],
                ]
            )


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = read_names(dataset_root)
    candidates_by_class = select_candidates(dataset_root, names, min_area=args.min_area, max_area=args.max_area)

    selected: list[Candidate] = []
    copied_paths: dict[int, dict[str, Path]] = {}
    for class_id, class_name in enumerate(names):
        candidates = candidates_by_class[class_id]
        if not candidates:
            print(f"[WARN] no candidates for {class_name}")
            continue
        top_candidates = candidates[: args.top_k]
        make_candidate_grid(top_candidates, names, output_dir / f"top_candidates_{class_name}.jpg", args.thumb_size)
        selected_candidate = top_candidates[0]
        selected.append(selected_candidate)
        copied_paths[class_id] = save_candidate_files(selected_candidate, names, output_dir, class_id + 1)

    make_four_panel(selected, names, output_dir / "poster_four_classes_annotated.jpg", annotated=True)
    make_four_panel(selected, names, output_dir / "poster_four_classes_clean.jpg", annotated=False)
    write_manifest(selected, names, copied_paths, output_dir / "selected_manifest.csv")

    print(f"OUTPUT_DIR={output_dir}")
    print(f"FOUR_PANEL_ANNOTATED={output_dir / 'poster_four_classes_annotated.jpg'}")
    print(f"FOUR_PANEL_CLEAN={output_dir / 'poster_four_classes_clean.jpg'}")
    print(f"MANIFEST={output_dir / 'selected_manifest.csv'}")
    for candidate in selected:
        paths = copied_paths[candidate.class_id]
        print(
            f"SELECTED {names[candidate.class_id]} score={candidate.score:.1f} "
            f"area={candidate.target_box.area:.3f} split={candidate.split} "
            f"original={paths['original']} annotated={paths['annotated']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
