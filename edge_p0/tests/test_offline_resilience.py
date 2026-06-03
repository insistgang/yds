from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from linkable_edge.models import DetectionEvent
from linkable_edge.publisher import EventPublisher, SafePublisher
from linkable_edge.report_gen import generate_weekly_report


class OfflineResilienceTests(unittest.TestCase):
    def test_offline_report_generation_and_event_cache_do_not_crash(self) -> None:
        class OfflinePublisher(EventPublisher):
            node_id = "offline-node"
            location = {"site": "local-test"}

            def publish(self, events) -> None:  # type: ignore[no-untyped-def]
                raise OSError("network unreachable")

        event = DetectionEvent(
            event_id="evt-offline",
            label="road_obstacle",
            confidence=0.91,
            timestamp=datetime(2026, 6, 3, 8, 0, tzinfo=timezone.utc),
            priority=80,
            direction="front",
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            publisher = SafePublisher(OfflinePublisher(), cache_dir=cache_dir)
            publisher.publish([event])

            report = generate_weekly_report(
                [
                    {
                        "event_id": event.event_id,
                        "timestamp": event.timestamp.isoformat(),
                        "type": event.label,
                        "confidence": event.confidence,
                        "location": "B点",
                    }
                ],
                period="2026-06-03",
                generated_at=datetime(2026, 6, 3, 8, 5),
            )

            self.assertIn("LinkAble 周度无障碍环境监测报告", report)
            cached_files = list(cache_dir.glob("*.json"))
            self.assertEqual(len(cached_files), 1)
            self.assertIn("network unreachable", cached_files[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
