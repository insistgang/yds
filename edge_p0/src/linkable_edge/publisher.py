from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import uuid
from dataclasses import dataclass, field
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


@dataclass(slots=True)
class AsyncHttpPublisher(EventPublisher):
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
        thread = threading.Thread(target=self._send, args=(req,), daemon=True)
        thread.start()

    def _send(self, req: request.Request) -> None:
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as response:
                response.read()
        except Exception as exc:
            print(f"[WARN] async publish failed: {exc}", file=sys.stderr)


@dataclass(slots=True)
class SafePublisher(EventPublisher):
    inner: EventPublisher
    cache_dir: Path = field(default_factory=lambda: Path(".tmp/linkable_publisher_cache"))
    retry_cached: bool = True
    async_publish: bool = False

    def publish(self, events: Iterable[DetectionEvent]) -> None:
        event_list = list(events)
        if not event_list:
            return

        if self.async_publish:
            thread = threading.Thread(target=self._publish_with_cache, args=(event_list,), daemon=True)
            thread.start()
            return

        self._publish_with_cache(event_list)

    def _publish_with_cache(self, event_list: list[DetectionEvent]) -> None:

        if self.retry_cached:
            self.flush_cache()

        try:
            self.inner.publish(event_list)
        except Exception as exc:
            self._cache_events(event_list, exc)
            print(f"[WARN] publish failed; cached locally: {exc}", file=sys.stderr)

    def flush_cache(self) -> None:
        url = getattr(self.inner, "url", None)
        if not url or not self.cache_dir.exists():
            return

        for path in sorted(self.cache_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self._post_payload(url, payload)
                path.unlink()
                print(f"[INFO] flushed cached event payload: {path.name}")
            except Exception as exc:
                print(f"[WARN] cached publish retry failed for {path.name}: {exc}", file=sys.stderr)
                return

    def _cache_events(self, events: list[DetectionEvent], exc: Exception) -> None:
        node_id = getattr(self.inner, "node_id", "edge-001")
        location = getattr(self.inner, "location", None)
        payload = build_detection_payload(events, node_id=node_id, location=location)
        payload["_cached_at"] = datetime.now(timezone.utc).isoformat()
        payload["_cache_reason"] = str(exc)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.cache_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}-{uuid.uuid4().hex}.json"
        tmp_path = output_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(output_path)

    def _post_payload(self, url: str, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout_sec = getattr(self.inner, "timeout_sec", 2.0)
        with request.urlopen(req, timeout=timeout_sec) as response:
            response.read()
