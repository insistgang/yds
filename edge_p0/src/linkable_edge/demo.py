from __future__ import annotations

import argparse
from datetime import timedelta

from .audio import build_audio_output
from .models import Detection, FrameDetections
from .pipeline import EdgePipeline
from .publisher import StdoutPublisher


def build_mock_frames() -> list[FrameDetections]:
    base = FrameDetections(frame_id=1).timestamp
    return [
        FrameDetections(
            frame_id=1,
            timestamp=base,
            detections=[
                Detection("blind_road_occupied", 0.90, distance_m=8.2, direction="right_front"),
            ],
        ),
        FrameDetections(
            frame_id=2,
            timestamp=base + timedelta(milliseconds=100),
            detections=[
                Detection("blind_road_occupied", 0.92, distance_m=8.0, direction="right_front"),
            ],
        ),
        FrameDetections(
            frame_id=3,
            timestamp=base + timedelta(milliseconds=200),
            detections=[
                Detection("stairs", 0.88, distance_m=2.0, direction="front"),
            ],
        ),
        FrameDetections(
            frame_id=4,
            timestamp=base + timedelta(milliseconds=300),
            detections=[
                Detection("stairs", 0.91, distance_m=2.0, direction="front"),
            ],
        ),
        FrameDetections(
            frame_id=5,
            timestamp=base + timedelta(milliseconds=400),
            detections=[
                Detection("ramp", 0.86, distance_m=3.0, direction="right"),
            ],
        ),
        FrameDetections(
            frame_id=6,
            timestamp=base + timedelta(milliseconds=500),
            detections=[
                Detection("ramp", 0.88, distance_m=3.0, direction="right"),
            ],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LinkAble edge P0 mock pipeline.")
    parser.add_argument("--audio", choices=["print", "pyttsx3"], default="print")
    parser.add_argument("--publish", action="store_true", help="Print event payloads as mock publish output.")
    args = parser.parse_args()

    audio_output = build_audio_output(args.audio)
    publisher = StdoutPublisher() if args.publish else None
    pipeline = EdgePipeline(audio_output=audio_output, publisher=publisher)

    print("[DEMO] Running mock edge-side pipeline...")
    for frame in build_mock_frames():
        print(f"[FRAME] frame_id={frame.frame_id} detections={len(frame.detections)}")
        results = pipeline.process_frame(frame)
        for result in results:
            print(f"[EVENT] {result.event.label} -> {result.text}")


if __name__ == "__main__":
    main()
