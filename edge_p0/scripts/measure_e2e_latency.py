"""端到端延迟测量脚本

测量从"YOLO 检测完成"到"语音播报开始"的延迟，目标 < 1.5 秒。

用法:
    cd edge_p0
    python -m scripts.measure_e2e_latency

输出:
    - 每个视频的 avg / p50 / p99 延迟
    - 汇总 JSON 保存到 test_results/e2e_latency.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from linkable_edge.audio import PrintAudioOutput
from linkable_edge.detector import DEFAULT_COCO_LABEL_MAP, YoloDetector, YoloDetectorConfig
from linkable_edge.event_builder import EventBuilder, EventBuilderConfig
from linkable_edge.inputs import VideoFileSource
from linkable_edge.pipeline import EdgePipeline


TEST_VIDEOS_DIR = ROOT / "test_videos"
OUTPUT_PATH = ROOT / "test_results" / "e2e_latency.json"
CSV_OUTPUT_PATH = ROOT / "test_results" / "e2e_latency_events.csv"
PLOT_OUTPUT_PATH = ROOT / "test_results" / "e2e_latency_distribution.png"


@dataclass(slots=True)
class LatencyRecord:
    video: str
    frame_id: int
    label: str
    latency_ms: float


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def run_one_video(
    video_path: Path,
    detector: YoloDetector,
    pipeline: EdgePipeline,
    max_frames: int | None = None,
) -> list[LatencyRecord]:
    source = VideoFileSource(video_path)
    if not source.open():
        print(f"  [WARN] cannot open {video_path.name}")
        return []

    latencies: list[LatencyRecord] = []
    frame_id = 0
    try:
        while max_frames is None or frame_id < max_frames:
            ok, image = source.read()
            if not ok or image is None:
                break
            frame_id += 1

            t_detect_start = time.monotonic()
            frame = detector.predict_frame(image, frame_id=frame_id)
            t_detect_done = time.monotonic()

            results = pipeline.process_frame(frame, detection_done_at=t_detect_done)

            for r in results:
                if r.audio_spoken:
                    latencies.append(
                        LatencyRecord(
                            video=video_path.stem,
                            frame_id=frame_id,
                            label=r.event.label,
                            latency_ms=max(0.0, r.latency_ms),
                        )
                    )
    finally:
        source.release()

    return latencies


def stats_for(latencies: list[float]) -> dict:
    if not latencies:
        return {"count": 0, "avg_ms": 0, "p50_ms": 0, "p99_ms": 0}
    s = sorted(latencies)
    n = len(s)
    return {
        "count": n,
        "avg_ms": round(sum(s) / n, 2),
        "min_ms": round(s[0], 2),
        "max_ms": round(s[-1], 2),
        "p50_ms": round(percentile(s, 0.50), 2),
        "p99_ms": round(percentile(s, 0.99), 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure local detection-to-speech latency.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit per video for local smoke tests.")
    parser.add_argument("--video-limit", type=int, default=None, help="Optional number of videos to test.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = sorted(TEST_VIDEOS_DIR.glob("*.mp4"))
    if args.video_limit is not None:
        videos = videos[:args.video_limit]
    if not videos:
        print(f"[ERROR] no .mp4 files in {TEST_VIDEOS_DIR}")
        return 1

    print(f"[INFO] found {len(videos)} test videos in {TEST_VIDEOS_DIR}")

    model_path = ROOT / "best.pt"
    if not model_path.exists():
        model_path = "yolo11n.pt"
        print(f"[WARN] best.pt not found, falling back to {model_path}")

    detector = YoloDetector(
        YoloDetectorConfig(
            model_path=str(model_path),
            image_size=640,
            confidence_threshold=0.25,
            label_map=dict(DEFAULT_COCO_LABEL_MAP),
        )
    )

    pipeline = EdgePipeline(
        audio_output=PrintAudioOutput(),
        event_builder=EventBuilder(EventBuilderConfig(
            confidence_threshold=0.25,
            min_consecutive_frames=2,
            emit_cooldown_frames=5,
        )),
        audio_cooldown_sec=0.0,
    )

    all_results: dict[str, dict] = {}
    global_latencies: list[float] = []
    global_records: list[LatencyRecord] = []

    for video_path in videos:
        name = video_path.stem
        print(f"\n[VIDEO] {video_path.name}")
        records = run_one_video(video_path, detector, pipeline, max_frames=args.max_frames)
        latencies = [record.latency_ms for record in records]
        st = stats_for(latencies)
        all_results[name] = st
        global_latencies.extend(latencies)
        global_records.extend(records)

        if st["count"] > 0:
            print(f"  spoken={st['count']}  avg={st['avg_ms']:.1f}ms  "
                  f"p50={st['p50_ms']:.1f}ms  p99={st['p99_ms']:.1f}ms")
        else:
            print("  no spoken events (no detections triggered speech)")

    global_stats = stats_for(global_latencies)
    output = {
        "target_ms": 1500,
        "global": global_stats,
        "per_video": all_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[RESULT] saved to {OUTPUT_PATH}")
    write_latency_csv(CSV_OUTPUT_PATH, global_records)
    print(f"[RESULT] CSV saved to {CSV_OUTPUT_PATH}")
    write_latency_distribution(PLOT_OUTPUT_PATH, global_latencies)
    print(f"[RESULT] distribution plot saved to {PLOT_OUTPUT_PATH}")
    print(f"[GLOBAL] count={global_stats['count']}  avg={global_stats.get('avg_ms', 0):.1f}ms  "
          f"p50={global_stats.get('p50_ms', 0):.1f}ms  p99={global_stats.get('p99_ms', 0):.1f}ms")

    target = 1500
    avg = global_stats.get("avg_ms", 0)
    p99 = global_stats.get("p99_ms", 0)
    if avg > 0:
        status = "PASS" if p99 < target else "FAIL"
        print(f"[CHECK] target={target}ms  p99={p99:.1f}ms  {status}")

    return 0


def write_latency_csv(output_path: Path, records: list[LatencyRecord]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["video", "frame_id", "label", "latency_ms"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "video": record.video,
                    "frame_id": record.frame_id,
                    "label": record.label,
                    "latency_ms": f"{record.latency_ms:.4f}",
                }
            )


def write_latency_distribution(output_path: Path, latencies: list[float]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[WARN] matplotlib unavailable, skip latency plot: {exc}", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if latencies:
        ax.hist(latencies, bins=min(20, max(1, len(set(latencies)))), color="#2f6f9f", edgecolor="white")
        p99 = percentile(sorted(latencies), 0.99)
        ax.axvline(p99, color="#b23a48", linestyle="--", label=f"p99={p99:.2f}ms")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No spoken events", ha="center", va="center")
    ax.set_xlabel("Detection-to-speech latency (ms)")
    ax.set_ylabel("Spoken event count")
    ax.set_title("End-to-End Latency Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
