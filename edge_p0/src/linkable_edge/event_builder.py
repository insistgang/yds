from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .models import Detection, DetectionEvent, FrameDetections


# P1 reserved labels - not triggered in P0 phase (AGENTS.md 7.2)
P1_LABELS = frozenset({"traffic_red", "traffic_green", "crosswalk"})

PRIORITY_BY_LABEL = {
    "stairs": 100,
    "traffic_red": 95,
    "road_obstacle": 80,
    "blind_road_occupied": 70,
    "ramp": 60,
    "crosswalk": 40,
}


@dataclass(slots=True)
class EventBuilderConfig:
    confidence_threshold: float = 0.55
    min_consecutive_frames: int = 2
    emit_cooldown_frames: int = 5


@dataclass(slots=True)
class _EventState:
    streak: int = 0
    last_detection: Detection | None = None
    last_frame_id: int = -1
    last_emitted_frame_id: int = -10_000


@dataclass(slots=True)
class EventBuilder:
    config: EventBuilderConfig = field(default_factory=EventBuilderConfig)
    _states: Dict[str, _EventState] = field(default_factory=dict)

    def process_frame(self, frame: FrameDetections) -> list[DetectionEvent]:
        emitted: list[DetectionEvent] = []
        strongest_by_label: dict[str, Detection] = {}

        for detection in frame.detections:
            if detection.label in P1_LABELS:
                continue
            if detection.confidence < self.config.confidence_threshold:
                continue
            current = strongest_by_label.get(detection.label)
            if current is None or detection.confidence > current.confidence:
                strongest_by_label[detection.label] = detection

        seen_labels = set(strongest_by_label)

        for label, detection in strongest_by_label.items():
            state = self._states.setdefault(label, _EventState())
            if state.last_frame_id == frame.frame_id - 1:
                state.streak += 1
            else:
                state.streak = 1

            state.last_detection = detection
            state.last_frame_id = frame.frame_id

            if state.streak < self.config.min_consecutive_frames:
                continue

            if frame.frame_id - state.last_emitted_frame_id < self.config.emit_cooldown_frames:
                continue

            state.last_emitted_frame_id = frame.frame_id
            emitted.append(
                DetectionEvent(
                    event_id=f"{label}-{frame.frame_id}",
                    label=label,
                    confidence=detection.confidence,
                    timestamp=frame.timestamp,
                    priority=PRIORITY_BY_LABEL.get(label, 50),
                    source_frame_ids=[frame.frame_id],
                    bbox=detection.bbox,
                    distance_m=detection.distance_m,
                    direction=detection.direction,
                )
            )

        for label in list(self._states.keys()):
            state = self._states[label]
            if label not in seen_labels and state.last_frame_id != frame.frame_id:
                state.streak = 0
                # 长期未出现的标签清理，防止内存泄漏
                if frame.frame_id - state.last_frame_id > self.config.emit_cooldown_frames * 10:
                    del self._states[label]

        return sorted(emitted, key=lambda event: event.priority, reverse=True)

