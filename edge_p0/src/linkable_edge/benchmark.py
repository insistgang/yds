from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json


@dataclass
class InferenceMetrics:
    frame_id: int
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0
    end_to_end_ms: float = 0.0


class BenchmarkCollector:
    """推理性能指标收集、汇总、持久化（JSON/CSV）

    AGENTS.md 指标：
    - 推理帧率 >= 30 FPS
    - 单帧推理延迟 < 100 ms
    - 检测到播报响应 < 1.5 s
    """

    def __init__(self) -> None:
        self.records: list[InferenceMetrics] = []

    def record_from_ultralytics(self, frame_id: int, result: Any) -> InferenceMetrics:
        """从 Ultralytics result.speed 字典提取指标"""
        speed = getattr(result, "speed", None) or {}
        metric = InferenceMetrics(
            frame_id=frame_id,
            preprocess_ms=speed.get("preprocess", 0.0),
            inference_ms=speed.get("inference", 0.0),
            postprocess_ms=speed.get("postprocess", 0.0),
            total_ms=sum(speed.values()) if speed else 0.0,
        )
        self.records.append(metric)
        return metric

    def record(self, metric: InferenceMetrics) -> None:
        self.records.append(metric)

    def summary(self) -> dict[str, float | int]:
        if not self.records:
            return {"count": 0}

        preprocess_vals = [r.preprocess_ms for r in self.records]
        inference_vals = [r.inference_ms for r in self.records]
        postprocess_vals = [r.postprocess_ms for r in self.records]
        total_vals = [r.total_ms for r in self.records]
        e2e_vals = [r.end_to_end_ms for r in self.records if r.end_to_end_ms > 0]

        def _stats(vals: list[float]) -> dict[str, float]:
            if not vals:
                return {"avg": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p99": 0.0}
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            return {
                "avg": sum(vals_sorted) / n,
                "min": vals_sorted[0],
                "max": vals_sorted[-1],
                "p50": vals_sorted[n // 2],
                "p99": vals_sorted[int(n * 0.99)],
            }

        return {
            "count": len(self.records),
            "fps": 1000.0 / (sum(total_vals) / len(total_vals)) if total_vals else 0.0,
            "preprocess_ms": _stats(preprocess_vals),
            "inference_ms": _stats(inference_vals),
            "postprocess_ms": _stats(postprocess_vals),
            "total_ms": _stats(total_vals),
            "end_to_end_ms": _stats(e2e_vals),
        }

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": self.summary(),
                    "records": [
                        {
                            "frame_id": r.frame_id,
                            "preprocess_ms": r.preprocess_ms,
                            "inference_ms": r.inference_ms,
                            "postprocess_ms": r.postprocess_ms,
                            "total_ms": r.total_ms,
                            "end_to_end_ms": r.end_to_end_ms,
                        }
                        for r in self.records
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def save_csv(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["frame_id,preprocess_ms,inference_ms,postprocess_ms,total_ms,end_to_end_ms"]
        for r in self.records:
            lines.append(
                f"{r.frame_id},{r.preprocess_ms},{r.inference_ms},"
                f"{r.postprocess_ms},{r.total_ms},{r.end_to_end_ms}"
            )
        path.write_text("\n".join(lines), encoding="utf-8")
