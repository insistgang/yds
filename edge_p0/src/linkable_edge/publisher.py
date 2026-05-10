from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from urllib import request

from .models import DetectionEvent


def build_detection_payload(
    events: Iterable[DetectionEvent],
    node_id: str = "edge-001",
    location: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event_list = list(events)
    timestamp = event_list[0].timestamp.isoformat() if event_list else None
    return {
        "node_id": node_id,
        "timestamp": timestamp,
        "location": dict(location) if location is not None else None,
        "events": [event.to_api_dict() for event in event_list],
    }


class EventPublisher:
    def publish(self, events: Iterable[DetectionEvent]) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class StdoutPublisher(EventPublisher):
    node_id: str = "edge-001"
    location: Mapping[str, Any] | None = None

    def publish(self, events: Iterable[DetectionEvent]) -> None:
        payload = build_detection_payload(events, node_id=self.node_id, location=self.location)
        print(f"[PUBLISH] {json.dumps(payload, ensure_ascii=False)}")


@dataclass(slots=True)
class HttpPublisher(EventPublisher):
    url: str
    node_id: str = "edge-001"
    location: Mapping[str, Any] | None = None
    timeout_sec: float = 2.0

    def publish(self, events: Iterable[DetectionEvent]) -> None:
        payload = build_detection_payload(events, node_id=self.node_id, location=self.location)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_sec) as response:
            response.read()
