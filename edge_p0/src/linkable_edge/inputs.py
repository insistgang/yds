from __future__ import annotations

from abc import ABC, abstractmethod
import glob
from pathlib import Path
from typing import Any, Iterator

import cv2


class FrameSource(ABC):
    """统一输入抽象：支持 USB/CSI/视频文件/图片序列 fallback"""

    @abstractmethod
    def open(self) -> bool:
        """打开源，返回是否成功"""
        ...

    @abstractmethod
    def read(self) -> tuple[bool, Any]:
        """读取一帧，返回 (success, frame)"""
        ...

    @abstractmethod
    def release(self) -> None:
        """释放资源"""
        ...

    @property
    @abstractmethod
    def source_id(self) -> str:
        """源标识符，用于日志"""
        ...

    @property
    @abstractmethod
    def fps(self) -> float:
        """源帧率"""
        ...

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(width, height)"""
        ...

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


class UsbCameraSource(FrameSource):
    """USB 摄像头源"""

    def __init__(
        self,
        device_idx: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fourcc: str = "MJPG",
    ) -> None:
        self.device_idx = device_idx
        self.width = width
        self.height = height
        self._fps = fps
        self.fourcc = fourcc
        self._cap: cv2.VideoCapture | None = None

    @property
    def source_id(self) -> str:
        return f"usb:{self.device_idx}"

    @property
    def fps(self) -> float:
        return float(self._fps)

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.device_idx, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            # fallback to default backend
            self._cap = cv2.VideoCapture(self.device_idx)
        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
        self._cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        return True

    def read(self) -> tuple[bool, Any]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class CsiCameraSource(FrameSource):
    """Jetson CSI 摄像头源（nvarguscamerasrc）"""

    def __init__(
        self,
        sensor_id: int = 0,
        sensor_mode: int = 2,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        flip_method: int = 0,
    ) -> None:
        self.sensor_id = sensor_id
        self.sensor_mode = sensor_mode
        self.width = width
        self.height = height
        self._fps = fps
        self.flip_method = flip_method
        self._cap: cv2.VideoCapture | None = None

    @property
    def source_id(self) -> str:
        return f"csi:{self.sensor_id}"

    @property
    def fps(self) -> float:
        return float(self._fps)

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def _build_pipeline(self) -> str:
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} sensor-mode={self.sensor_mode} ! "
            f"video/x-raw(memory:NVMM), width=(int){self.width}, height=(int){self.height}, "
            f"format=(string)NV12, framerate=(fraction){self._fps}/1 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            f"video/x-raw, width=(int){self.width}, height=(int){self.height}, "
            f"format=(string)BGRx ! videoconvert ! "
            f"video/x-raw, format=(string)BGR ! appsink drop=true"
        )

    def open(self) -> bool:
        pipeline = self._build_pipeline()
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        return self._cap.isOpened()

    def read(self) -> tuple[bool, Any]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoFileSource(FrameSource):
    """视频文件源"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cap: cv2.VideoCapture | None = None

    @property
    def source_id(self) -> str:
        return f"video:{self.path}"

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 30.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        if self._cap is None:
            return (640, 480)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(str(self.path))
        return self._cap.isOpened()

    def read(self) -> tuple[bool, Any]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImageSequenceSource(FrameSource):
    """图片序列源（支持通配符或目录）"""

    def __init__(self, pattern: str | Path) -> None:
        self.pattern = Path(pattern)
        self._images: list[Path] = []
        self._index = 0

    @property
    def source_id(self) -> str:
        return f"images:{self.pattern}"

    @property
    def fps(self) -> float:
        return 1.0

    @property
    def frame_size(self) -> tuple[int, int]:
        return (640, 480)

    def open(self) -> bool:
        if self.pattern.is_dir():
            exts = {"*.jpg", "*.jpeg", "*.png", "*.bmp"}
            self._images = []
            for ext in exts:
                self._images.extend(self.pattern.glob(ext))
            self._images.sort()
        else:
            self._images = [Path(p) for p in sorted(glob.glob(str(self.pattern)))]
        self._index = 0
        return len(self._images) > 0

    def read(self) -> tuple[bool, Any]:
        if self._index >= len(self._images):
            return False, None
        path = self._images[self._index]
        self._index += 1
        frame = cv2.imread(str(path))
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self._images = []
        self._index = 0


class FallbackFrameSource(FrameSource):
    """FrameSource wrapper that moves to the next source when open/read fails."""

    def __init__(self, candidates: list[tuple[str, FrameSource]]) -> None:
        self._candidates = candidates
        self._current_index = -1
        self._current: FrameSource | None = None

    @property
    def source_id(self) -> str:
        if self._current is None:
            return "fallback:none"
        return self._current.source_id

    @property
    def fps(self) -> float:
        if self._current is None:
            return 0.0
        return self._current.fps

    @property
    def frame_size(self) -> tuple[int, int]:
        if self._current is None:
            return (0, 0)
        return self._current.frame_size

    def open(self) -> bool:
        if self._current is not None:
            return True
        return self._open_next(0)

    def read(self) -> tuple[bool, Any]:
        if self._current is None and not self.open():
            return False, None

        while self._current is not None:
            ok, frame = self._current.read()
            if ok and frame is not None:
                return True, frame

            print(f"[WARN] source read failed, fallback from {self._current.source_id}")
            if not self._open_next(self._current_index + 1):
                return False, None

        return False, None

    def release(self) -> None:
        if self._current is not None:
            self._current.release()
        self._current = None
        self._current_index = -1

    def _open_next(self, start_index: int) -> bool:
        if self._current is not None:
            self._current.release()
            self._current = None

        for index in range(start_index, len(self._candidates)):
            name, source = self._candidates[index]
            try:
                if source.open():
                    self._current_index = index
                    self._current = source
                    print(f"[INFO] Using {name} source: {source.source_id}")
                    return True
                print(f"[WARN] {name} source unavailable: {source.source_id}")
            except Exception as exc:
                print(f"[WARN] {name} source failed: {exc}")

        self._current_index = -1
        return False


def auto_select_source(
    *,
    csi_sensor_id: int | None = None,
    usb_device_idx: int | None = None,
    video_path: str | Path | None = None,
    image_pattern: str | Path | None = None,
    prefer_csi: bool = True,
) -> FrameSource:
    """按优先级自动选择输入源：CSI -> USB -> 视频 -> 图片序列

    AGENTS.md 风险响应规范：CSI 相机未恢复时，自动 fallback 到下一级。
    """
    candidates: list[tuple[str, FrameSource]] = []

    if csi_sensor_id is not None:
        candidates.append(("CSI", CsiCameraSource(sensor_id=csi_sensor_id)))
    if usb_device_idx is not None:
        candidates.append(("USB", UsbCameraSource(device_idx=usb_device_idx)))
    if video_path is not None:
        candidates.append(("VIDEO", VideoFileSource(video_path)))
    if image_pattern is not None:
        candidates.append(("IMAGES", ImageSequenceSource(image_pattern)))

    # Sort by priority
    priority = {"CSI": 0, "USB": 1, "VIDEO": 2, "IMAGES": 3}
    if not prefer_csi:
        priority = {"USB": 0, "CSI": 1, "VIDEO": 2, "IMAGES": 3}
    candidates.sort(key=lambda x: priority.get(x[0], 99))

    fallback = FallbackFrameSource(candidates)
    if fallback.open():
        return fallback

    raise RuntimeError("No input source available")
