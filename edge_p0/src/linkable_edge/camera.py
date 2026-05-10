from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class NvArgusCameraConfig:
    sensor_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    sensor_mode: int | None = None


def build_nvargus_opencv_pipeline(config: NvArgusCameraConfig | None = None) -> str:
    camera = config or NvArgusCameraConfig()
    source_parts = [
        "nvarguscamerasrc",
        f"sensor-id={camera.sensor_id}",
    ]
    if camera.sensor_mode is not None:
        source_parts.append(f"sensor-mode={camera.sensor_mode}")

    source = " ".join(source_parts)
    caps = (
        f"video/x-raw(memory:NVMM), width=(int){camera.width}, "
        f"height=(int){camera.height}, format=(string)NV12, "
        f"framerate=(fraction){camera.fps}/1"
    )
    return (
        f"{source} ! {caps} ! nvvidconv ! video/x-raw, format=(string)BGRx "
        "! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
    )
