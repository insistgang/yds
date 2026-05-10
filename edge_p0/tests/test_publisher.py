from __future__ import annotations

from datetime import datetime, timezone
import unittest

from linkable_edge.models import DetectionEvent
from linkable_edge.publisher import build_detection_payload


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


if __name__ == "__main__":
    unittest.main()
