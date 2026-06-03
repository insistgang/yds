from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .audio import DEFAULT_MINIMAX_VOICE, AudioOutput, build_audio_output
from .detector import DEFAULT_COCO_LABEL_MAP, YoloDetector, YoloDetectorConfig
from .event_builder import EventBuilder, EventBuilderConfig
from .image_demo import parse_label_map
from .inputs import FrameSource, VideoFileSource, ImageSequenceSource
from .pipeline import EdgePipeline
from .publisher import EventPublisher, HttpPublisher, SafePublisher
from .benchmark import BenchmarkCollector


DEFAULT_PUBLISH_URL = "http://127.0.0.1:8000/api/v1/detect"


def build_detector(model_path: str, imgsz: int, conf: float, device: str | None, label_map: dict[str, str] | None) -> YoloDetector:
    return YoloDetector(
        YoloDetectorConfig(
            model_path=model_path,
            image_size=imgsz,
            confidence_threshold=conf,
            device=device,
            label_map=label_map or dict(DEFAULT_COCO_LABEL_MAP),
        )
    )


def build_event_builder(conf: float, min_frames: int = 2, cooldown_frames: int = 5) -> EventBuilder:
    return EventBuilder(
        EventBuilderConfig(
            confidence_threshold=conf,
            min_consecutive_frames=min_frames,
            emit_cooldown_frames=cooldown_frames,
        )
    )


def build_publisher(publish: bool, url: str, node_id: str) -> EventPublisher | None:
    if not publish:
        return None
    return SafePublisher(
        HttpPublisher(url=url, node_id=node_id),
        async_publish=True,
    )


def run_video_demo(
    source: FrameSource,
    detector: YoloDetector,
    audio_output: AudioOutput,
    publisher: EventPublisher | None,
    event_builder: EventBuilder,
    audio_cooldown_sec: float = 5.0,
    show: bool = False,
    max_frames: int | None = None,
    benchmark: BenchmarkCollector | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    pipeline = EdgePipeline(
        audio_output=audio_output,
        publisher=publisher,
        event_builder=event_builder,
        audio_cooldown_sec=audio_cooldown_sec,
        clock=clock,
    )

    cv2 = None
    if show:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            print("[WARN] opencv not available for display", file=sys.stderr)

    frame_id = 0
    start_at = clock()
    last_report_at = start_at

    print(f"[VIDEO] source={source.source_id} fps={source.fps}")

    try:
        while max_frames is None or frame_id < max_frames:
            ok, image = source.read()
            if not ok or image is None:
                print("[INFO] source exhausted")
                break

            frame_id += 1
            now = clock()

            frame = detector.predict_frame(image, frame_id=frame_id)

            if benchmark is not None:
                # Ultralytics speed dict is in result; grab from last result if available
                pass  # benchmark recording done inside detector if we extend it

            results = pipeline.process_frame(frame)

            for result in results:
                spoken = "spoken" if result.audio_spoken else "cooldown"
                print(f"[EVENT] {result.event.label} {spoken} -> {result.text}")

            if cv2 is not None:
                cv2.imshow("LinkAble Video P0", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_id % 30 == 0:
                report_at = clock()
                elapsed = max(report_at - last_report_at, 1e-6)
                fps = 30 / elapsed
                print(f"[STATS] fps={fps:.2f} frames={frame_id}")
                last_report_at = report_at
    finally:
        source.release()
        if cv2 is not None and hasattr(cv2, "destroyAllWindows"):
            cv2.destroyAllWindows()

    if benchmark is not None:
        summary = benchmark.summary()
        print(f"[BENCHMARK] {summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LinkAble P0 pipeline with video file or image sequence.")
    parser.add_argument("--source", required=True, help="Video file path or image directory/glob pattern")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--audio", choices=["print", "minimax", "pyttsx3"], default="print")
    parser.add_argument("--voice", default=DEFAULT_MINIMAX_VOICE)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--publish-url", default=DEFAULT_PUBLISH_URL)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--cooldown-sec", type=float, default=5.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--event-conf", type=float, default=0.25)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--node-id", default="edge-001")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--benchmark", action="store_true", help="Collect inference latency metrics")
    parser.add_argument("--output-dir", default="./runs/video_demo", help="Directory to save benchmark results")
    parser.add_argument(
        "--label-map",
        action="append",
        help='Map detector class to P0 type, e.g., bicycle=blind_road_occupied',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_path = Path(args.source)
    if source_path.is_dir() or ("*" in args.source):
        source: FrameSource = ImageSequenceSource(args.source)
    else:
        source = VideoFileSource(args.source)

    if not source.open():
        print(f"[ERROR] failed to open source: {args.source}", file=sys.stderr)
        return 1

    try:
        detector = build_detector(
            args.model, args.imgsz, args.conf, args.yolo_device, parse_label_map(args.label_map)
        )
    except Exception as exc:
        print(f"[ERROR] model load failed: {exc}", file=sys.stderr)
        return 2

    try:
        audio_output = build_audio_output(args.audio, voice=args.voice)
        publisher = build_publisher(args.publish, args.publish_url, args.node_id)
        event_builder = build_event_builder(args.event_conf)
        benchmark = BenchmarkCollector() if args.benchmark else None

        run_video_demo(
            source=source,
            detector=detector,
            audio_output=audio_output,
            publisher=publisher,
            event_builder=event_builder,
            audio_cooldown_sec=args.cooldown_sec,
            show=args.show,
            max_frames=args.max_frames,
            benchmark=benchmark,
        )

        if benchmark is not None:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            benchmark.save_json(output_dir / "benchmark.json")
            benchmark.save_csv(output_dir / "benchmark.csv")
            print(f"[INFO] benchmark saved to {output_dir}")

    except KeyboardInterrupt:
        print("[VIDEO] interrupted")
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
