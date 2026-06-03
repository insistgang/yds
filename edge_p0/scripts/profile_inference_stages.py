#!/usr/bin/env python3
"""Profile YOLO preprocess / inference / postprocess stages for local evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_OUTPUT_DIR = ROOT / "test_results" / "benchmark_stages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile YOLO stage timings and export CSV.")
    parser.add_argument("--model", default=str(ROOT / "best.pt"))
    parser.add_argument("--images", type=Path, default=ROOT / "datasets" / "p0_yolo" / "images" / "val")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def collect_images(path: Path, limit: int) -> list[Path]:
    if path.is_file():
        return [path]
    images = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)
    return images[:limit]


def main() -> int:
    args = parse_args()
    images = collect_images(args.images, args.limit)
    if not images:
        print(f"[ERROR] no images found: {args.images}", file=sys.stderr)
        return 1

    from ultralytics import YOLO
    from linkable_edge.benchmark import BenchmarkCollector

    model = YOLO(str(args.model))
    collector = BenchmarkCollector()
    rows: list[dict[str, object]] = []

    print(f"[INFO] model={args.model}")
    print(f"[INFO] images={len(images)} source={args.images}")

    for idx, image_path in enumerate(images, 1):
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        result = results[0]
        metric = collector.record_from_ultralytics(idx, result)
        preprocess_ms = float(metric.preprocess_ms)
        inference_ms = float(metric.inference_ms)
        postprocess_ms = float(metric.postprocess_ms)
        total_ms = float(metric.total_ms)
        denom = total_ms or 1.0
        rows.append(
            {
                "frame_id": idx,
                "image": image_path.name,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
                "total_ms": total_ms,
                "preprocess_pct": preprocess_ms / denom * 100,
                "inference_pct": inference_ms / denom * 100,
                "postprocess_pct": postprocess_ms / denom * 100,
            }
        )

    summary = build_summary(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "inference_stages.csv"
    json_path = args.output_dir / "inference_stages_summary.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[RESULT] CSV saved: {csv_path}")
    print(f"[RESULT] summary saved: {json_path}")
    print(
        "[SUMMARY] avg_ms "
        f"pre={summary['avg_preprocess_ms']:.2f} "
        f"infer={summary['avg_inference_ms']:.2f} "
        f"post={summary['avg_postprocess_ms']:.2f} "
        f"total={summary['avg_total_ms']:.2f}"
    )
    print(
        "[SUMMARY] avg_pct "
        f"pre={summary['avg_preprocess_pct']:.1f}% "
        f"infer={summary['avg_inference_pct']:.1f}% "
        f"post={summary['avg_postprocess_pct']:.1f}%"
    )
    return 0


def build_summary(rows: list[dict[str, object]]) -> dict[str, float | int | str]:
    count = len(rows)

    def avg(key: str) -> float:
        return sum(float(row[key]) for row in rows) / count if count else 0.0

    largest_stage = max(
        {
            "preprocess": avg("preprocess_ms"),
            "inference": avg("inference_ms"),
            "postprocess": avg("postprocess_ms"),
        }.items(),
        key=lambda item: item[1],
    )[0]
    return {
        "count": count,
        "avg_preprocess_ms": avg("preprocess_ms"),
        "avg_inference_ms": avg("inference_ms"),
        "avg_postprocess_ms": avg("postprocess_ms"),
        "avg_total_ms": avg("total_ms"),
        "avg_preprocess_pct": avg("preprocess_pct"),
        "avg_inference_pct": avg("inference_pct"),
        "avg_postprocess_pct": avg("postprocess_pct"),
        "largest_stage": largest_stage,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "frame_id",
        "image",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "total_ms",
        "preprocess_pct",
        "inference_pct",
        "postprocess_pct",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{row[key]:.4f}" if isinstance(row[key], float) else row[key]
                    for key in fieldnames
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
