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
    enable_debounce: bool = True
    enable_cooldown: bool = True
    state_ttl_frames: int = 50


@dataclass(slots=True)
class _EventState:
    streak: int = 0
    last_detection: Detection | None = None
    last_frame_id: int = -1
    last_emitted_frame_id: int = -10_000
    source_frame_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class EventBuilder:
    config: EventBuilderConfig = field(default_factory=EventBuilderConfig)
    _states: Dict[str, _EventState] = field(default_factory=dict)

    def process_frame(self, frame: FrameDetections) -> list[DetectionEvent]:
        emitted: list[DetectionEvent] = []
        strongest_by_label: dict[str, Detection] = {}
        min_consecutive_frames = (
            max(1, self.config.min_consecutive_frames)
            if self.config.enable_debounce
            else 1
        )
        emit_cooldown_frames = (
            max(0, self.config.emit_cooldown_frames)
            if self.config.enable_cooldown
            else 0
        )

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
                state.source_frame_ids.append(frame.frame_id)
            else:
                state.streak = 1
                state.source_frame_ids = [frame.frame_id]

            max_frame_history = max(self.config.state_ttl_frames, min_consecutive_frames)
            if len(state.source_frame_ids) > max_frame_history:
                state.source_frame_ids = state.source_frame_ids[-max_frame_history:]

            state.last_detection = detection
            state.last_frame_id = frame.frame_id

            if state.streak < min_consecutive_frames:
                continue

            if emit_cooldown_frames > 0 and frame.frame_id - state.last_emitted_frame_id < emit_cooldown_frames:
                continue

            state.last_emitted_frame_id = frame.frame_id
            source_frame_ids = state.source_frame_ids[-min_consecutive_frames:]
            emitted.append(
                DetectionEvent(
                    event_id=f"{label}-{frame.frame_id}",
                    label=label,
                    confidence=detection.confidence,
                    timestamp=frame.timestamp,
                    priority=PRIORITY_BY_LABEL.get(label, 50),
                    source_frame_ids=source_frame_ids,
                    bbox=detection.bbox,
                    distance_m=detection.distance_m,
                    direction=detection.direction,
                )
            )

        for label in list(self._states.keys()):
            state = self._states[label]
            if label not in seen_labels and state.last_frame_id != frame.frame_id:
                state.streak = 0
                state.source_frame_ids = []
                # 长期未出现的标签清理，防止内存泄漏
                if frame.frame_id - state.last_frame_id > self.config.state_ttl_frames:
                    del self._states[label]

        return sorted(emitted, key=lambda event: event.priority, reverse=True)

