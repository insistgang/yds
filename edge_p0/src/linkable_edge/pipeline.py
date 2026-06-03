from __future__ import annotations

from dataclasses import dataclass, field
import sys
import time
from typing import Callable

from .audio import AudioOutput
from .event_builder import EventBuilder, EventBuilderConfig
from .models import DetectionEvent, FrameDetections
from .publisher import EventPublisher
from .semantics import render_event_text


@dataclass(slots=True)
class PipelineResult:
    event: DetectionEvent
    text: str
    audio_spoken: bool = True
    t_detection_done: float = 0.0
    t_speak_start: float = 0.0
    latency_ms: float = 0.0


@dataclass(slots=True)
class EdgePipeline:
    audio_output: AudioOutput
    publisher: EventPublisher | None = None
    event_builder: EventBuilder = field(default_factory=lambda: EventBuilder(EventBuilderConfig()))
    audio_cooldown_sec: float = 0.0
    clock: Callable[[], float] = time.monotonic
    _last_audio_at_by_label: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # Preload all template audio caches on startup (AGENTS.md 7.4)
        if hasattr(self.audio_output, "preload"):
            from .semantics import ALL_TEMPLATE_TEXTS
            self.audio_output.preload(ALL_TEMPLATE_TEXTS)

    def process_frame(
        self, frame: FrameDetections, detection_done_at: float | None = None
    ) -> list[PipelineResult]:
        t_enter = self.clock()
        events = self.event_builder.process_frame(frame)
        if not events:
            return []

        if self.publisher is not None:
            self.publisher.publish(events)

        results: list[PipelineResult] = []
        # AGENTS.md 低信息密度原则：一帧只播报最高优先级事件
        audio_event = events[0] if events else None
        t_ref = detection_done_at if detection_done_at is not None else t_enter
        for event in events:
            text = render_event_text(event)
            audio_spoken = False
            t_speak = 0.0
            latency_ms = 0.0
            if event is audio_event and self._should_speak(event):
                t_speak = self.clock()
                latency_ms = (t_speak - t_ref) * 1000
                try:
                    self.audio_output.speak(text)
                    audio_spoken = True
                except Exception as exc:
                    print(f"[WARN] audio failed for {event.label}: {exc}", file=sys.stderr)
                    self._last_audio_at_by_label.pop(event.label, None)
            else:
                if event is not audio_event:
                    print(f"[DEBUG] audio skipped for {event.label} (p={event.priority}): "
                          f"lower than {audio_event.label} (p={audio_event.priority})")
            results.append(PipelineResult(
                event=event,
                text=text,
                audio_spoken=audio_spoken,
                t_detection_done=t_ref,
                t_speak_start=t_speak,
                latency_ms=latency_ms,
            ))
        return results

    def _should_speak(self, event: DetectionEvent) -> bool:
        if self.audio_cooldown_sec <= 0:
            return True

        now = self.clock()
        last_audio_at = self._last_audio_at_by_label.get(event.label)
        if last_audio_at is not None and now - last_audio_at < self.audio_cooldown_sec:
            return False

        self._last_audio_at_by_label[event.label] = now
        return True