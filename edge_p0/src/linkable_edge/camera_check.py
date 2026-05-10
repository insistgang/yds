from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def run_command(name: str, command: str, timeout_sec: int = 20) -> int:
    print(f"===== {name} =====")
    try:
        completed = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        print(exc.stdout or "", end="")
        print(exc.stderr or "", end="")
        print(f"TIMEOUT after {timeout_sec}s")
        return 124

    print(completed.stdout, end="")
    print(completed.stderr, end="")
    print(f"exit={completed.returncode}")
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal Jetson CSI camera diagnostics for LinkAble.")
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--sensor-mode", type=int, default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", default="~/linkable-gst-frame-cam0.jpg")
    args = parser.parse_args()

    output_path = Path(args.output).expanduser()
    output_arg = shlex.quote(str(output_path))
    source_parts = ["nvarguscamerasrc", f"sensor-id={args.sensor_id}", "num-buffers=1"]
    if args.sensor_mode is not None:
        source_parts.append(f"sensor-mode={args.sensor_mode}")
    source = " ".join(source_parts)
    caps = (
        f"video/x-raw(memory:NVMM), width=(int){args.width}, "
        f"height=(int){args.height}, format=(string)NV12, framerate=(fraction){args.fps}/1"
    )

    run_command("v4l2 devices", "v4l2-ctl --list-devices 2>&1 || true")
    run_command("device nodes", "ls -l /dev/video* /dev/media* 2>&1 || true")
    run_command("argus service", "systemctl is-active nvargus-daemon 2>&1 || true")
    run_command("argus plugin", "gst-inspect-1.0 nvarguscamerasrc 2>&1 | sed -n '1,40p' || true")
    run_command(
        "gst jpeg capture",
        f"rm -f {output_arg}; gst-launch-1.0 -e {source} ! {shlex.quote(caps)} ! "
        f"nvjpegenc ! filesink location={output_arg}",
        timeout_sec=25,
    )

    if output_path.exists():
        size = output_path.stat().st_size
        print(f"capture_file={output_path} size={size}")
        if size <= 0:
            print("capture_status=empty_file")
        else:
            print("capture_status=ok")
    else:
        print(f"capture_file={output_path} missing")


if __name__ == "__main__":
    main()
