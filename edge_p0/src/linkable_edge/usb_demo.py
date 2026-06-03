from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from .audio import DEFAULT_MINIMAX_VOICE, AudioOutput, build_audio_output
from .detector import DEFAULT_COCO_LABEL_MAP, YoloDetector, YoloDetectorConfig
from .event_builder import EventBuilder, EventBuilderConfig
from .image_demo import parse_label_map
from .pipeline import EdgePipeline
from .publisher import EventPublisher, HttpPublisher, SafePublisher


USB_DEVICE_IDX = 0
USB_WIDTH = 640
USB_HEIGHT = 480
USB_FPS = 30
USB_FOURCC = "MJPG"
DEFAULT_CAPTURE_DIR = Path("/tmp/linkable_capture")
DEFAULT_PUBLISH_URL = "http://127.0.0.1:8000/api/v1/detect"


@dataclass(slots=True)
class UsbDemoConfig:
    device_idx: int = USB_DEVICE_IDX
    model: str = "yolo11n.pt"
    audio: str = "minimax"
    voice: str = DEFAULT_MINIMAX_VOICE
    publish: bool = False
    publish_url: str = DEFAULT_PUBLISH_URL
    node_id: str = "edge-001"
    image_size: int = 640
    confidence_threshold: float = 0.25
    event_confidence_threshold: float = 0.25
    yolo_device: str | int | None = None
    show: bool = False
    save_interval_sec: float = 0.0
    capture_dir: Path = field(default_factory=lambda: DEFAULT_CAPTURE_DIR)
    audio_cooldown_sec: float = 5.0
    min_consecutive_frames: int = 2
    event_emit_cooldown_frames: int | None = None


@dataclass(slots=True)
class UsbDemoStats:
    frames: int = 0
    events: int = 0
    speeches: int = 0
    saved_frames: int = 0


def configure_v4l2_dynamic_framerate(device_idx: int) -> bool:
    device_node = f"/dev/video{device_idx}"
    command = [
        "v4l2-ctl",
        "-d",
        device_node,
        "-c",
        "exposure_dynamic_framerate=0",
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        print("[WARN] v4l2-ctl not found; skip exposure_dynamic_framerate=0", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"[WARN] v4l2-ctl timeout while configuring {device_node}", file=sys.stderr)
        return False

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        print(
            f"[WARN] failed to set exposure_dynamic_framerate=0 on {device_node}: {detail}",
            file=sys.stderr,
        )
        return False

    print(f"[USB] {device_node} exposure_dynamic_framerate=0")
    return True


def load_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python is not installed") from exc
    return cv2


def open_usb_capture(config: UsbDemoConfig, cv2_module: Any) -> Any:
    capture = cv2_module.VideoCapture(config.device_idx, cv2_module.CAP_V4L2)
    capture.set(cv2_module.CAP_PROP_FOURCC, cv2_module.VideoWriter_fourcc(*USB_FOURCC))
    capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, USB_WIDTH)
    capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, USB_HEIGHT)
    capture.set(cv2_module.CAP_PROP_FPS, USB_FPS)

    if not capture.isOpened():
        raise RuntimeError(
            f"camera open failed: device_idx={config.device_idx}, node=/dev/video{config.device_idx}. "
            "Check USB connection, permissions, and use /dev/video0 rather than /dev/video1."
        )

    return capture


def build_detector(config: UsbDemoConfig, label_map: dict[str, str] | None = None) -> YoloDetector:
    try:
        return YoloDetector(
            YoloDetectorConfig(
                model_path=config.model,
                image_size=config.image_size,
                confidence_threshold=config.confidence_threshold,
                device=config.yolo_device,
                label_map=label_map or dict(DEFAULT_COCO_LABEL_MAP),
            )
        )
    except Exception as exc:
        raise RuntimeError(f"model load failed: model={config.model}: {exc}") from exc


def build_event_builder(config: UsbDemoConfig) -> EventBuilder:
    emit_cooldown_frames = config.event_emit_cooldown_frames
    if emit_cooldown_frames is None:
        emit_cooldown_frames = max(1, int(config.audio_cooldown_sec * USB_FPS))

    return EventBuilder(
        EventBuilderConfig(
            confidence_threshold=config.event_confidence_threshold,
            min_consecutive_frames=config.min_consecutive_frames,
            emit_cooldown_frames=emit_cooldown_frames,
        )
    )


def build_publisher(config: UsbDemoConfig) -> EventPublisher | None:
    if not config.publish:
        return None
    return SafePublisher(
        HttpPublisher(url=config.publish_url, node_id=config.node_id),
        async_publish=True,
    )


def save_original_frame(frame: Any, frame_id: int, capture_dir: Path, cv2_module: Any) -> bool:
    capture_dir.mkdir(parents=True, exist_ok=True)
    output_path = capture_dir / f"frame_{frame_id:06d}_{int(time.time() * 1000)}.jpg"
    ok = bool(cv2_module.imwrite(str(output_path), frame))
    if ok:
        print(f"[CAPTURE] {output_path}")
    else:
        print(f"[WARN] failed to save capture frame: {output_path}", file=sys.stderr)
    return ok


def run_usb_demo(
    config: UsbDemoConfig,
    *,
    detector: Any,
    audio_output: AudioOutput,
    publisher: EventPublisher | None = None,
    cv2_module: Any | None = None,
    max_frames: int | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> UsbDemoStats:
    cv2_module = cv2_module or load_cv2()
    capture = open_usb_capture(config, cv2_module)
    pipeline = EdgePipeline(
        audio_output=audio_output,
        publisher=publisher,
        event_builder=build_event_builder(config),
        audio_cooldown_sec=config.audio_cooldown_sec,
        clock=clock,
    )
    stats = UsbDemoStats()
    frame_id = 0
    start_at = clock()
    last_report_at = start_at
    next_save_at = start_at if config.save_interval_sec > 0 else float("inf")

    print(
        "[USB] running "
        f"device=/dev/video{config.device_idx} fourcc={USB_FOURCC} "
        f"size={USB_WIDTH}x{USB_HEIGHT} fps={USB_FPS} audio={config.audio}"
    )

    try:
        while max_frames is None or stats.frames < max_frames:
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(
                    f"camera read failed: device_idx={config.device_idx}, frame_id={frame_id + 1}"
                )

            frame_id += 1
            stats.frames += 1
            now = clock()

            if now >= next_save_at:
                if save_original_frame(image, frame_id, config.capture_dir, cv2_module):
                    stats.saved_frames += 1
                next_save_at = now + config.save_interval_sec

            frame = detector.predict_frame(image, frame_id=frame_id)
            results = pipeline.process_frame(frame)
            stats.events += len(results)
            stats.speeches += sum(1 for result in results if result.audio_spoken)

            for result in results:
                spoken = "spoken" if result.audio_spoken else "cooldown"
                print(f"[EVENT] {result.event.label} {spoken} -> {result.text}")

            if config.show:
                cv2_module.imshow("LinkAble USB P0", image)
                if cv2_module.waitKey(1) & 0xFF == ord("q"):
                    break

            if stats.frames % 30 == 0:
                report_at = clock()
                elapsed = max(report_at - last_report_at, 1e-6)
                fps = 30 / elapsed
                print(
                    f"[STATS] fps={fps:.2f} frames={stats.frames} "
                    f"events={stats.events} speeches={stats.speeches}"
                )
                last_report_at = report_at
    finally:
        capture.release()
        if config.show and hasattr(cv2_module, "destroyAllWindows"):
            cv2_module.destroyAllWindows()

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LinkAble P0 pipeline with a USB camera.")
    parser.add_argument("--device-idx", type=int, default=USB_DEVICE_IDX)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--audio", choices=["print", "minimax"], default="minimax")
    parser.add_argument("--voice", default=DEFAULT_MINIMAX_VOICE)
    parser.add_argument("--publish", action="store_true", help="POST events to the configured cloud endpoint.")
    parser.add_argument('--offline', action='store_true', help='force offline mode')
    parser.add_argument("--publish-url", default=DEFAULT_PUBLISH_URL)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--save-interval", type=float, default=0.0)
    parser.add_argument("--cooldown-sec", type=float, default=5.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--event-conf", type=float, default=0.25)
    parser.add_argument("--yolo-device", default=None, help="YOLO device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--node-id", default="edge-001")
    parser.add_argument(
        "--label-map",
        action="append",
        help="Map detector class to P0 type, for example bicycle=blind_road_occupied. Can be repeated.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> UsbDemoConfig:
    return UsbDemoConfig(
        device_idx=args.device_idx,
        model=args.model,
        audio=args.audio,
        voice=args.voice,
        publish=args.publish,
        publish_url=args.publish_url,
        node_id=args.node_id,
        image_size=args.imgsz,
        confidence_threshold=args.conf,
        event_confidence_threshold=args.event_conf,
        yolo_device=args.yolo_device,
        show=args.show,
        save_interval_sec=args.save_interval,
        audio_cooldown_sec=args.cooldown_sec,
    )


def main() -> int:
    args = parse_args()
    config = config_from_args(args)

    # offline mode: disable cloud dependencies
    if args.offline:
        config.publish = False
        if config.audio == "minimax":
            print("[INFO] offline mode: forcing audio to print")
            config.audio = "print"


    configure_v4l2_dynamic_framerate(config.device_idx)

    try:
        detector = build_detector(config, label_map=parse_label_map(args.label_map))
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    try:
        audio_output = build_audio_output(config.audio, voice=config.voice)
        publisher = build_publisher(config)
        run_usb_demo(
            config,
            detector=detector,
            audio_output=audio_output,
            publisher=publisher,
        )
    except KeyboardInterrupt:
        print("[USB] interrupted")
        return 130
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
