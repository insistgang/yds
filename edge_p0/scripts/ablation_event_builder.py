"""EventBuilder 消融实验：有/无去抖对比"""
import argparse
import csv
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from linkable_edge.event_builder import EventBuilder, EventBuilderConfig
from linkable_edge.detector import YoloDetector, YoloDetectorConfig
from linkable_edge.inputs import VideoFileSource


@dataclass
class AblationResult:
    config_name: str
    total_frames: int = 0
    total_detections: int = 0
    total_events: int = 0
    events_by_label: dict = field(default_factory=dict)
    events_per_100_frames: float = 0.0
    timestamps: list = field(default_factory=list)


def run_ablation(
    source_path: str,
    model_path: str,
    config: EventBuilderConfig,
    config_name: str,
    max_frames: int | None = None,
) -> AblationResult:
    result = AblationResult(config_name=config_name)

    detector = YoloDetector(YoloDetectorConfig(model_path=model_path))
    source = VideoFileSource(source_path)
    builder = EventBuilder(config=config)

    if not source.open():
        print(f"[ERROR] failed to open: {source_path}")
        return result

    frame_id = 0
    while max_frames is None or frame_id < max_frames:
        success, frame = source.read()
        if not success or frame is None:
            break

        frame_id += 1
        detections = detector.predict_frame(frame, frame_id)
        result.total_detections += len(detections.detections)
        result.total_frames += 1

        events = builder.process_frame(detections)
        result.total_events += len(events)

        for event in events:
            label = event.label
            result.events_by_label[label] = result.events_by_label.get(label, 0) + 1
            result.timestamps.append({
                "frame": frame_id,
                "label": label,
                "confidence": event.confidence,
            })

    source.release()

    if result.total_frames > 0:
        result.events_per_100_frames = (result.total_events / result.total_frames) * 100

    return result


def main():
    parser = argparse.ArgumentParser(description="Run EventBuilder ablation on local test videos.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit per video for smoke tests.")
    parser.add_argument("--video-limit", type=int, default=None, help="Optional number of videos to test.")
    args = parser.parse_args()

    videos = [ROOT / "test_videos" / f"{index}.mp4" for index in range(1, 8)]
    if args.video_limit is not None:
        videos = videos[:args.video_limit]
    model_path = str(ROOT / "best.pt")

    # 无去抖：每帧检测都触发事件
    config_no_debounce = EventBuilderConfig(
        confidence_threshold=0.55,
        min_consecutive_frames=2,
        emit_cooldown_frames=5,
        enable_debounce=False,
        enable_cooldown=False,
    )

    # 有去抖：当前配置
    config_with_debounce = EventBuilderConfig(
        confidence_threshold=0.55,
        min_consecutive_frames=2,
        emit_cooldown_frames=5,
        enable_debounce=True,
        enable_cooldown=True,
    )

    all_results = []

    for video_path in videos:
        if not video_path.exists():
            print(f"[SKIP] {video_path} not found")
            continue

        video_name = video_path.stem
        print(f"\n{'='*60}")
        print(f"Video: {video_name}")
        print(f"{'='*60}")

        # 无去抖
        print(f"[1/2] Running WITHOUT debounce...")
        r1 = run_ablation(
            str(video_path),
            model_path,
            config_no_debounce,
            f"{video_name}_no_debounce",
            max_frames=args.max_frames,
        )

        # 有去抖
        print(f"[2/2] Running WITH debounce...")
        r2 = run_ablation(
            str(video_path),
            model_path,
            config_with_debounce,
            f"{video_name}_with_debounce",
            max_frames=args.max_frames,
        )

        all_results.append((video_name, r1, r2))

        # 打印对比
        print(f"\n  无去抖: {r1.total_events} 次播报 / {r1.total_frames} 帧 = {r1.events_per_100_frames:.1f} 次/百帧")
        print(f"  有去抖: {r2.total_events} 次播报 / {r2.total_frames} 帧 = {r2.events_per_100_frames:.1f} 次/百帧")
        if r1.total_events > 0:
            reduction = (1 - r2.total_events / r1.total_events) * 100
            print(f"  减少率: {reduction:.1f}%")

    # 汇总表格
    print(f"\n\n{'='*80}")
    print("汇总表格")
    print(f"{'='*80}")
    print(f"{'视频':<10} {'无去抖播报':>10} {'有去抖播报':>10} {'减少率':>10} {'无去抖/百帧':>12} {'有去抖/百帧':>12}")
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")

    total_no = 0
    total_yes = 0

    for video_name, r1, r2 in all_results:
        total_no += r1.total_events
        total_yes += r2.total_events
        reduction = (1 - r2.total_events / r1.total_events) * 100 if r1.total_events > 0 else 0
        print(f"{video_name:<10} {r1.total_events:>10} {r2.total_events:>10} {reduction:>9.1f}% {r1.events_per_100_frames:>12.1f} {r2.events_per_100_frames:>12.1f}")

    total_reduction = (1 - total_yes / total_no) * 100 if total_no > 0 else 0
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
    print(f"{'合计':<10} {total_no:>10} {total_yes:>10} {total_reduction:>9.1f}%")

    # 保存结果
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_no_debounce": {
            "confidence_threshold": 0.55,
            "min_consecutive_frames": 2,
            "emit_cooldown_frames": 5,
            "enable_debounce": False,
            "enable_cooldown": False,
        },
        "config_with_debounce": {
            "confidence_threshold": 0.55,
            "min_consecutive_frames": 2,
            "emit_cooldown_frames": 5,
            "enable_debounce": True,
            "enable_cooldown": True,
        },
        "results": []
    }

    for video_name, r1, r2 in all_results:
        output["results"].append({
            "video": video_name,
            "no_debounce": {
                "total_events": r1.total_events,
                "total_frames": r1.total_frames,
                "events_per_100_frames": r1.events_per_100_frames,
                "by_label": r1.events_by_label,
            },
            "with_debounce": {
                "total_events": r2.total_events,
                "total_frames": r2.total_frames,
                "events_per_100_frames": r2.events_per_100_frames,
                "by_label": r2.events_by_label,
            },
            "reduction_pct": (1 - r2.total_events / r1.total_events) * 100 if r1.total_events > 0 else 0,
        })

    output_path = ROOT / "runs" / "ablation" / "event_builder_ablation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {output_path}")

    csv_path = ROOT / "runs" / "ablation" / "event_builder_ablation.csv"
    write_ablation_csv(csv_path, all_results)
    print(f"CSV已保存: {csv_path}")

    chart_path = ROOT / "runs" / "ablation" / "event_builder_ablation.png"
    write_ablation_chart(chart_path, all_results)
    print(f"柱状图已保存: {chart_path}")


def write_ablation_csv(output_path: Path, all_results: list[tuple[str, AblationResult, AblationResult]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["video", "mode", "event_count", "reduction_pct"],
        )
        writer.writeheader()
        for video_name, no_builder, with_builder in all_results:
            reduction = (
                (1 - with_builder.total_events / no_builder.total_events) * 100
                if no_builder.total_events > 0
                else 0
            )
            writer.writerow(
                {
                    "video": video_name,
                    "mode": "without_event_builder",
                    "event_count": no_builder.total_events,
                    "reduction_pct": f"{reduction:.2f}",
                }
            )
            writer.writerow(
                {
                    "video": video_name,
                    "mode": "with_event_builder",
                    "event_count": with_builder.total_events,
                    "reduction_pct": f"{reduction:.2f}",
                }
            )


def write_ablation_chart(output_path: Path, all_results: list[tuple[str, AblationResult, AblationResult]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[WARN] matplotlib unavailable, skip chart: {exc}", file=sys.stderr)
        return

    videos = [item[0] for item in all_results]
    without_counts = [item[1].total_events for item in all_results]
    with_counts = [item[2].total_events for item in all_results]
    x_positions = list(range(len(videos)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([x - width / 2 for x in x_positions], without_counts, width, label="Without EventBuilder")
    ax.bar([x + width / 2 for x in x_positions], with_counts, width, label="With EventBuilder")
    ax.set_xlabel("Video")
    ax.set_ylabel("Spoken event count")
    ax.set_title("EventBuilder Ablation: Spoken Events by Video")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(videos)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
