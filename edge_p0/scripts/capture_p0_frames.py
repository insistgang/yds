from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_LABELS = (
    "road_obstacle",
    "stairs",
    "ramp",
    "blind_road_occupied",
    "negative",
)
MANIFEST_FIELDS = (
    "image_path",
    "label",
    "route_point",
    "session",
    "timestamp",
    "device",
    "width",
    "height",
    "privacy_checked",
    "notes",
)
PRIVACY_NOTICE = (
    "[PRIVACY] Do not collect clear faces, license plates, school names, "
    "name tags, or other sensitive identifiers. Delete or anonymize sensitive "
    "samples before training. This tool saves discrete images only; it does "
    "not save continuous video and does not upload images."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture discrete USB-camera images for the LinkAble P0 raw dataset."
    )
    parser.add_argument("--label", required=True, choices=ALLOWED_LABELS)
    parser.add_argument("--route-point", required=True, help="Route point such as A/B/C/D/E/F.")
    parser.add_argument("--session", required=True, help="Session id, for example 20260428_route1.")
    parser.add_argument("--output-root", default="datasets/linkable_p0_raw")
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=0.0,
        help="0 means manual capture. >0 enables low-frequency still capture, not video.",
    )
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--device-idx", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--show", action="store_true", help="Show OpenCV preview window.")
    parser.add_argument("--dry-run", action="store_true", help="Check camera and paths without saving.")
    return parser.parse_args()


def import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("OpenCV is required to capture frames: import cv2 failed") from exc
    return cv2


def ensure_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()


def open_capture(cv2, args: argparse.Namespace):
    cap = cv2.VideoCapture(args.device_idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera device index {args.device_idx}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap


def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("failed to read frame from camera")
    return frame


def safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def make_filename(args: argparse.Namespace, index: int, timestamp: str) -> str:
    return (
        f"{safe_token(args.session)}_"
        f"{safe_token(args.route_point)}_"
        f"{args.label}_"
        f"{timestamp}_"
        f"{index:04d}.jpg"
    )


def append_manifest(
    manifest_path: Path,
    image_path: Path,
    args: argparse.Namespace,
    timestamp_iso: str,
) -> None:
    with manifest_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writerow(
            {
                "image_path": image_path.as_posix(),
                "label": args.label,
                "route_point": args.route_point,
                "session": args.session,
                "timestamp": timestamp_iso,
                "device": f"/dev/video{args.device_idx}",
                "width": args.width,
                "height": args.height,
                "privacy_checked": "false",
                "notes": "",
            }
        )


def save_frame(cv2, frame, target_dir: Path, manifest_path: Path, args: argparse.Namespace, index: int) -> Path:
    now = datetime.now(timezone.utc)
    timestamp_file = now.strftime("%Y%m%dT%H%M%SZ")
    filename = make_filename(args, index, timestamp_file)
    image_path = target_dir / filename
    ok = cv2.imwrite(str(image_path), frame)
    if not ok:
        raise RuntimeError(f"failed to write image: {image_path}")
    append_manifest(manifest_path, image_path, args, now.isoformat())
    return image_path


def run_manual_no_preview(cv2, cap, target_dir: Path, manifest_path: Path, args: argparse.Namespace) -> int:
    print("[MODE] manual terminal capture: press Enter to capture, type q then Enter to quit")
    saved = 0
    while saved < args.max_images:
        cmd = input("capture> ").strip().lower()
        if cmd == "q":
            break
        frame = read_frame(cap)
        saved += 1
        path = save_frame(cv2, frame, target_dir, manifest_path, args, saved)
        print(f"[SAVED] {path}")
    return saved


def run_manual_preview(cv2, cap, target_dir: Path, manifest_path: Path, args: argparse.Namespace) -> int:
    print("[MODE] manual preview capture: press c to save, q to quit")
    saved = 0
    while saved < args.max_images:
        frame = read_frame(cap)
        cv2.imshow("LinkAble P0 capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            saved += 1
            path = save_frame(cv2, frame, target_dir, manifest_path, args, saved)
            print(f"[SAVED] {path}")
    return saved


def run_interval_capture(cv2, cap, target_dir: Path, manifest_path: Path, args: argparse.Namespace) -> int:
    print(f"[MODE] low-frequency still capture every {args.interval_sec:.2f}s; this is not video")
    saved = 0
    next_capture = time.monotonic()
    while saved < args.max_images:
        frame = read_frame(cap)
        if args.show:
            cv2.imshow("LinkAble P0 capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        now = time.monotonic()
        if now >= next_capture:
            saved += 1
            path = save_frame(cv2, frame, target_dir, manifest_path, args, saved)
            print(f"[SAVED] {path}")
            next_capture = now + args.interval_sec
        time.sleep(0.01)
    return saved


def main() -> int:
    args = parse_args()
    if args.max_images < 1:
        raise SystemExit("--max-images must be >= 1")
    if args.interval_sec < 0:
        raise SystemExit("--interval-sec must be >= 0")

    print(PRIVACY_NOTICE)
    print("[NOTICE] For 30 FPS on HP w300, run: v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0")

    output_root = Path(args.output_root)
    target_dir = output_root / args.label / "images"
    manifest_path = output_root / "manifest.csv"
    target_dir.mkdir(parents=True, exist_ok=True)
    ensure_manifest(manifest_path)

    print(f"[DATASET] output_root={output_root}")
    print(f"[DATASET] target_dir={target_dir}")
    print(f"[DATASET] manifest={manifest_path}")

    cv2 = import_cv2()
    cap = open_capture(cv2, args)
    try:
        frame = read_frame(cap)
        actual_height, actual_width = frame.shape[:2]
        print(f"[CAMERA] device=/dev/video{args.device_idx} frame={actual_width}x{actual_height}")

        if args.dry_run:
            print("[DRY_RUN] camera and output path check passed; no image saved")
            return 0

        if args.interval_sec > 0:
            saved = run_interval_capture(cv2, cap, target_dir, manifest_path, args)
        elif args.show:
            saved = run_manual_preview(cv2, cap, target_dir, manifest_path, args)
        else:
            saved = run_manual_no_preview(cv2, cap, target_dir, manifest_path, args)

        print(f"[DONE] saved_images={saved}")
        return 0
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
