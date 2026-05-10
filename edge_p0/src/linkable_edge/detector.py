from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from .models import Detection, FrameDetections


P0_LABELS = {
    "blind_road_occupied",
    "stairs",
    "ramp",
    "road_obstacle",
}

# COCO 代理冷启动策略（Cold-start）
# 说明：在自定义模型训练完成前，使用 COCO 预训练权重代理检测通用障碍物。
# stairs / ramp / blind_road_occupied 为自定义类别，无法通过 COCO 代理检测，
# 必须在 Phase 3 通过自采数据训练自定义 YOLO 模型后支持。
DEFAULT_COCO_LABEL_MAP: dict[str, str] = {
    "backpack": "road_obstacle",
    "bench": "road_obstacle",
    "bicycle": "road_obstacle",
    "chair": "road_obstacle",
    "motorcycle": "road_obstacle",
    "person": "road_obstacle",
    "potted plant": "road_obstacle",
    "suitcase": "road_obstacle",
}


@dataclass(slots=True)
class YoloDetectorConfig:
    model_path: str = "yolo11n.pt"
    image_size: int = 640
    confidence_threshold: float = 0.25
    device: str | int | None = None
    engine: Literal["pytorch", "onnx", "tensorrt"] = "pytorch"  # 预留后端迁移
    label_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_COCO_LABEL_MAP))


def normalize_label(raw_label: str, label_map: Mapping[str, str] | None = None) -> str | None:
    label = raw_label.strip()
    if label in P0_LABELS:
        return label

    mapped = (label_map or {}).get(label)
    if mapped in P0_LABELS:
        return mapped

    return None


def infer_direction_from_bbox(bbox: tuple[int, int, int, int], image_width: int) -> str:
    if image_width <= 0:
        return "front"

    center_x = (bbox[0] + bbox[2]) / 2
    left_boundary = image_width / 3
    right_boundary = image_width * 2 / 3

    if center_x < left_boundary:
        return "left_front"
    if center_x > right_boundary:
        return "right_front"
    return "front"


class YoloDetector:
    def __init__(self, config: YoloDetectorConfig | None = None) -> None:
        self.config = config or YoloDetectorConfig()
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed") from exc

        self._model = YOLO(self.config.model_path)

    def predict_image(self, source: str | Path, frame_id: int = 1) -> FrameDetections:
        try:
            results = self._model.predict(
                source=str(source),
                imgsz=self.config.image_size,
                conf=self.config.confidence_threshold,
                device=self.config.device,
                verbose=False,
            )
        except Exception as exc:
            print(f"[WARN] inference failed for {source}: {exc}", file=sys.stderr)
            return FrameDetections(frame_id=frame_id)
        return self._frame_from_results(results, frame_id)

    def predict_frame(self, image: Any, frame_id: int) -> FrameDetections:
        try:
            results = self._model.predict(
                source=image,
                imgsz=self.config.image_size,
                conf=self.config.confidence_threshold,
                device=self.config.device,
                verbose=False,
            )
        except Exception as exc:
            # P0 演示降级：单帧失败不中断 pipeline
            print(f"[WARN] inference failed at frame {frame_id}: {exc}", file=sys.stderr)
            return FrameDetections(frame_id=frame_id)
        return self._frame_from_results(results, frame_id)

    def _frame_from_results(self, results: Any, frame_id: int) -> FrameDetections:
        frame = FrameDetections(frame_id=frame_id)
        for result in results:
            image_height, image_width = result.orig_shape[:2]
            names = result.names
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls.item())
                raw_label = str(names.get(class_id, class_id))
                label = normalize_label(raw_label, self.config.label_map)
                if label is None:
                    continue

                confidence = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                bbox = tuple(int(round(value)) for value in xyxy)
                if len(bbox) != 4:
                    continue

                frame.detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        bbox=bbox,
                        direction=infer_direction_from_bbox(bbox, image_width),
                    )
                )

        return frame

