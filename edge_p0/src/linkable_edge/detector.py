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

# mangdaojianceYOLO 16类 -> P0 4类 映射
# 说明：购买的盲道检测数据集，包含16个类别，需要映射到P0四类。
# 其中 stairs 和 ramp 在原数据集中不存在，需要从其他数据源补充训练。
MANGDAOJIANE_LABEL_MAP: dict[str, str] = {
    # 直接映射 - 盲道相关
    "blind track": "blind_road_occupied",
    # 障碍物映射 - 所有可能阻碍通行的物体
    "ashcan": "road_obstacle",
    "car": "road_obstacle",
    "bicycle": "road_obstacle",
    "person": "road_obstacle",
    "spherical_roadblock": "road_obstacle",
    "pole": "road_obstacle",
    "fire_hydrant": "road_obstacle",
    "truck": "road_obstacle",
    "dog": "road_obstacle",
    "motorbike": "road_obstacle",
    "warning_column": "road_obstacle",
    "bus": "road_obstacle",
    "tricycle": "road_obstacle",
    "reflective_cone": "road_obstacle",
    # stop_sign 不映射（P1阶段处理）
}


@dataclass(slots=True)
class YoloDetectorConfig:
    model_path: str = "best.pt"
    image_size: int = 640
    confidence_threshold: float = 0.25
    device: str | int | None = None
    engine: Literal["pytorch", "onnx", "tensorrt"] = "pytorch"  # 预留后端迁移
    label_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_COCO_LABEL_MAP))
    # 模型类型：coco（COCO预训练）、mangdaojiance（购买的16类模型）、p0_custom（P0四类自定义模型）
    model_type: Literal["coco", "mangdaojiance", "p0_custom"] = "coco"


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
        
        # 根据模型类型自动选择标签映射
        if self.config.model_type == "mangdaojiance":
            # 使用 mangdaojianceYOLO 的类别映射
            if not self.config.label_map or self.config.label_map == DEFAULT_COCO_LABEL_MAP:
                self.config = YoloDetectorConfig(
                    model_path=self.config.model_path,
                    image_size=self.config.image_size,
                    confidence_threshold=self.config.confidence_threshold,
                    device=self.config.device,
                    engine=self.config.engine,
                    label_map=MANGDAOJIANE_LABEL_MAP,
                    model_type=self.config.model_type,
                )
        elif self.config.model_type == "p0_custom":
            # P0 自定义模型，不需要标签映射（直接输出P0四类）
            self.config = YoloDetectorConfig(
                model_path=self.config.model_path,
                image_size=self.config.image_size,
                confidence_threshold=self.config.confidence_threshold,
                device=self.config.device,
                engine=self.config.engine,
                label_map={},  # 空映射，直接使用模型输出
                model_type=self.config.model_type,
            )
        
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

