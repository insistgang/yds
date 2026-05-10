from __future__ import annotations

from datetime import datetime, timezone
import unittest

from linkable_edge.models import DetectionEvent
from linkable_edge.semantics import render_event_text


class SemanticsTests(unittest.TestCase):
    def _event(self, label: str, direction: str | None = None, distance_m: float | None = None) -> DetectionEvent:
        return DetectionEvent(
            event_id="evt-test",
            label=label,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            priority=80,
            distance_m=distance_m,
            direction=direction,
        )

    def test_blind_road_message_contains_distance_and_direction(self) -> None:
        event = DetectionEvent(
            event_id="evt-1",
            label="blind_road_occupied",
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            priority=70,
            distance_m=8.0,
            direction="right_front",
        )

        text = render_event_text(event)
        self.assertIn("前方8米", text)
        self.assertIn("右前方", text)

    def test_ramp_message_fallback_without_direction(self) -> None:
        event = DetectionEvent(
            event_id="evt-2",
            label="ramp",
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            priority=60,
            distance_m=2.0,
        )

        text = render_event_text(event)
        self.assertEqual(text, "前方2米有坡道。")

    def test_road_obstacle_without_direction_uses_front_location(self) -> None:
        text = render_event_text(self._event("road_obstacle"))

        self.assertEqual(text, "前方有障碍，请注意避让。")

    def test_road_obstacle_front_or_center_direction_uses_front_location(self) -> None:
        for direction in ("front", "center"):
            with self.subTest(direction=direction):
                text = render_event_text(self._event("road_obstacle", direction=direction))

                self.assertEqual(text, "前方有障碍，请注意避让。")

    def test_road_obstacle_left_front_direction_uses_clean_location(self) -> None:
        text = render_event_text(self._event("road_obstacle", direction="left_front"))

        self.assertEqual(text, "左前方有障碍，请注意避让。")
        self.assertNotIn("前方左前方", text)

    def test_road_obstacle_right_front_direction_uses_clean_location(self) -> None:
        text = render_event_text(self._event("road_obstacle", direction="right_front"))

        self.assertEqual(text, "右前方有障碍，请注意避让。")
        self.assertNotIn("前方右前方", text)

    def test_stairs_front_direction_does_not_repeat_prefix(self) -> None:
        event = DetectionEvent(
            event_id="evt-3",
            label="stairs",
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            priority=100,
            distance_m=2.0,
            direction="front",
        )

        text = render_event_text(event)
        self.assertEqual(text, "前方2米有台阶，请减速。")

    def test_traffic_red_message_uses_existing_semantic_renderer(self) -> None:
        event = DetectionEvent(
            event_id="evt-4",
            label="traffic_red",
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            priority=95,
            distance_m=5.0,
            direction="front",
        )

        text = render_event_text(event)
        self.assertEqual(text, "前方5米是红灯，请在路口前等待。")

    def test_crosswalk_message_uses_existing_semantic_renderer(self) -> None:
        event = DetectionEvent(
            event_id="evt-5",
            label="crosswalk",
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            priority=40,
            distance_m=4.0,
            direction="front",
        )

        text = render_event_text(event)
        self.assertEqual(text, "前方4米有人行横道，请沿横道通过。")


if __name__ == "__main__":
    unittest.main()
