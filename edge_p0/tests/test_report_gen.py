from __future__ import annotations

from datetime import datetime
import unittest

from linkable_edge.report_gen import generate_weekly_report, summarize_events


class ReportGenTests(unittest.TestCase):
    def test_summarizes_structured_events_without_images(self) -> None:
        events = [
            {
                "event_id": "evt-1",
                "timestamp": "2026-06-01T08:10:00+08:00",
                "type": "blind_road_occupied",
                "confidence": 0.9,
                "location": "A点",
            },
            {
                "event_id": "evt-2",
                "timestamp": "2026-06-01T08:20:00+08:00",
                "type": "road_obstacle",
                "confidence": 0.8,
                "location": "A点",
            },
        ]

        summary = summarize_events(events)

        self.assertEqual(summary["total_events"], 2)
        self.assertEqual(summary["by_type"]["blind_road_occupied"], 1)
        self.assertEqual(summary["by_location"]["A点"], 2)
        self.assertEqual(summary["peak_hour"]["hour"], 8)

    def test_generates_markdown_report(self) -> None:
        report = generate_weekly_report(
            [
                {
                    "timestamp": "2026-06-01T08:10:00+08:00",
                    "type": "road_obstacle",
                    "confidence": 0.8,
                    "location": "B点",
                }
            ],
            period="2026-06-01 ~ 2026-06-07",
            generated_at=datetime(2026, 6, 3, 1, 0),
        )

        self.assertIn("LinkAble 周度无障碍环境监测报告", report)
        self.assertIn("路面障碍", report)
        self.assertIn("B点", report)
        self.assertNotIn("image", report.lower())


if __name__ == "__main__":
    unittest.main()
