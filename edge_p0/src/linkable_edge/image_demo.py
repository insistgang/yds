from __future__ import annotations

import argparse

from .audio import build_audio_output
from .detector import DEFAULT_COCO_LABEL_MAP, P0_LABELS, YoloDetector, YoloDetectorConfig
from .event_builder import EventBuilder, EventBuilderConfig
from .pipeline import EdgePipeline
from .publisher import StdoutPublisher


def parse_label_map(items: list[str] | None) -> dict[str, str]:
    label_map = dict(DEFAULT_COCO_LABEL_MAP)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid label map item: {item}")
        source, target = item.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or target not in P0_LABELS:
            raise ValueError(f"Invalid label map item: {item}")
        label_map[source] = target
    return label_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LinkAble P0 pipeline on one image with a YOLO model.")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path, for example yolo11n.pt or best.pt.")
    parser.add_argument("--source", required=True, help="Image path to run detection on.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--event-conf", type=float, default=0.25, help="Event confidence threshold.")
    parser.add_argument("--device", default=None, help="YOLO device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--audio", choices=["print", "pyttsx3"], default="print")
    parser.add_argument("--publish", action="store_true", help="Print /api/v1/detect style payload.")
    parser.add_argument("--node-id", default="edge-001")
    parser.add_argument(
        "--label-map",
        action="append",
        help="Map detector class to P0 type, for example bicycle=blind_road_occupied. Can be repeated.",
    )
    args = parser.parse_args()

    detector = YoloDetector(
        YoloDetectorConfig(
            model_path=args.model,
            image_size=args.imgsz,
            confidence_threshold=args.conf,
            device=args.device,
            label_map=parse_label_map(args.label_map),
        )
    )
    frame = detector.predict_image(args.source, frame_id=1)
    print(f"[IMAGE] detections={len(frame.detections)} source={args.source}")
    for detection in frame.detections:
        print(
            "[DETECTION] "
            f"label={detection.label} confidence={detection.confidence:.2f} "
            f"bbox={detection.bbox} direction={detection.direction}"
        )

    publisher = StdoutPublisher(node_id=args.node_id) if args.publish else None
    pipeline = EdgePipeline(
        audio_output=build_audio_output(args.audio),
        publisher=publisher,
        event_builder=EventBuilder(
            EventBuilderConfig(confidence_threshold=args.event_conf, min_consecutive_frames=1)
        ),
    )
    results = pipeline.process_frame(frame)
    if not results:
        print("[EVENT] no P0 event emitted")
        return

    for result in results:
        print(f"[EVENT] {result.event.label} -> {result.text}")


if __name__ == "__main__":
    main()
