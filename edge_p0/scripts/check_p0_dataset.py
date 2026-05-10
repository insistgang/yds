from __future__ import annotations

import argparse
import csv
from pathlib import Path


POSITIVE_TARGET_MIN = {
    "road_obstacle": 100,
    "stairs": 80,
    "ramp": 60,
    "blind_road_occupied": 80,
}
POSITIVE_TARGET_MAX = {
    "road_obstacle": 200,
    "stairs": 150,
    "ramp": 120,
    "blind_road_occupied": 150,
}
LABELS = tuple(POSITIVE_TARGET_MIN) + ("negative",)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LinkAble P0 raw dataset counts and manifest consistency.")
    parser.add_argument("--root", default="datasets/linkable_p0_raw")
    return parser.parse_args()


def count_images(root: Path, label: str) -> int:
    image_dir = root / label / "images"
    if not image_dir.exists():
        return 0
    return sum(1 for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def resolve_manifest_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = root / path
    if candidate.exists():
        return candidate
    return path


def check_manifest(root: Path) -> tuple[bool, int, list[str]]:
    manifest = root / "manifest.csv"
    if not manifest.exists():
        return False, 0, []

    missing: list[str] = []
    rows = 0
    with manifest.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows += 1
            image_path = row.get("image_path", "")
            if not image_path:
                missing.append("<empty image_path>")
                continue
            if not resolve_manifest_path(root, image_path).exists():
                missing.append(image_path)
    return True, rows, missing


def main() -> int:
    args = parse_args()
    root = Path(args.root)

    print(f"DATASET_ROOT={root}")
    counts = {label: count_images(root, label) for label in LABELS}

    for label in POSITIVE_TARGET_MIN:
        print(
            f"{label}: {counts[label]} / "
            f"target_min={POSITIVE_TARGET_MIN[label]} target_max={POSITIVE_TARGET_MAX[label]}"
        )
    print(f"negative: {counts['negative']}")

    total_positive = sum(counts[label] for label in POSITIVE_TARGET_MIN)
    print(f"TOTAL_POSITIVE={total_positive}")
    print("TOTAL_POSITIVE_TARGET=400-600")

    manifest_exists, manifest_rows, missing_paths = check_manifest(root)
    print(f"MANIFEST_EXISTS={'true' if manifest_exists else 'false'}")
    print(f"MANIFEST_ROWS={manifest_rows}")
    print(f"MANIFEST_MISSING_FILES={len(missing_paths)}")
    if missing_paths:
        print("MANIFEST_MISSING_BEGIN")
        for value in missing_paths[:50]:
            print(value)
        if len(missing_paths) > 50:
            print(f"... {len(missing_paths) - 50} more")
        print("MANIFEST_MISSING_END")

    ready = (
        manifest_exists
        and not missing_paths
        and total_positive >= 600
        and all(counts[label] >= POSITIVE_TARGET_MIN[label] for label in POSITIVE_TARGET_MIN)
    )
    print(f"DATASET_READY_FOR_BASELINE={'true' if ready else 'false'}")

    return 1 if missing_paths else 0


if __name__ == "__main__":
    raise SystemExit(main())
