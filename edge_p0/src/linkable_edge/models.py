from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int] | None = None
    distance_m: float | None = None
    direction: str | None = None


@dataclass(slots=True)
class FrameDetections:
    frame_id: int
    timestamp: datetime = field(default_factory=utc_now)
    detections: list[Detection] = field(default_factory=list)


@dataclass(slots=True)
class DetectionEvent:
    event_id: str
    label: str
    confidence: float
    timestamp: datetime
    priority: int
    source_frame_ids: list[int] = field(default_factory=list)
    bbox: tuple[int, int, int, int] | None = None
    distance_m: float | None = None
    direction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload

    def to_api_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["type"] = payload.pop("label")
        return payload
