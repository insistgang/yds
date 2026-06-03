from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from linkable_edge.models import DetectionEvent
from linkable_edge.publisher import EventPublisher, SafePublisher, build_detection_payload


class PublisherPayloadTests(unittest.TestCase):
    def test_builds_detect_api_payload(self) -> None:
        event = DetectionEvent(
            event_id="evt-1",
            label="road_obstacle",
            confidence=0.88,
            timestamp=datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
            priority=80,
            distance_m=None,
            direction="front",
        )

        payload = build_detection_payload(
            [event],
            node_id="edge-test",
            location={"lat": 31.0, "lng": 121.0, "source": "mobile_gps"},
        )

        self.assertEqual(payload["node_id"], "edge-test")
        self.assertEqual(payload["timestamp"], "2026-04-27T12:00:00+00:00")
        self.assertEqual(payload["location"]["source"], "mobile_gps")
        self.assertEqual(payload["events"][0]["type"], "road_obstacle")
        self.assertNotIn("label", payload["events"][0])

    def test_safe_publisher_caches_events_when_delegate_fails(self) -> None:
        class FailingPublisher(EventPublisher):
            node_id = "edge-cache-test"
            location = {"site": "anonymous-demo"}

            def publish(self, events) -> None:  # type: ignore[no-untyped-def]
                raise RuntimeError("offline")

        event = DetectionEvent(
            event_id="evt-cache",
            label="road_obstacle",
            confidence=0.88,
            timestamp=datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
            priority=80,
            direction="front",
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            publisher = SafePublisher(FailingPublisher(), cache_dir=cache_dir)

            publisher.publish([event])

            cached_files = list(cache_dir.glob("*.json"))
            self.assertEqual(len(cached_files), 1)
            cached_text = cached_files[0].read_text(encoding="utf-8")
            self.assertIn("edge-cache-test", cached_text)
            self.assertIn("road_obstacle", cached_text)
            self.assertIn("_cache_reason", cached_text)


if __name__ == "__main__":
    unittest.main()
