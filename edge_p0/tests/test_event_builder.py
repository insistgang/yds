from __future__ import annotations

import unittest

from linkable_edge.event_builder import EventBuilder, EventBuilderConfig
from linkable_edge.models import Detection, FrameDetections


class EventBuilderTests(unittest.TestCase):
    def test_emits_after_min_consecutive_frames(self) -> None:
        builder = EventBuilder(EventBuilderConfig(min_consecutive_frames=2, emit_cooldown_frames=5))
        frame1 = FrameDetections(frame_id=1, detections=[Detection("stairs", 0.90)])
        frame2 = FrameDetections(frame_id=2, detections=[Detection("stairs", 0.91)])

        self.assertEqual(builder.process_frame(frame1), [])
        events = builder.process_frame(frame2)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].label, "stairs")

    def test_respects_cooldown(self) -> None:
        builder = EventBuilder(EventBuilderConfig(min_consecutive_frames=1, emit_cooldown_frames=3))
        frame1 = FrameDetections(frame_id=1, detections=[Detection("road_obstacle", 0.95)])
        frame2 = FrameDetections(frame_id=2, detections=[Detection("road_obstacle", 0.95)])
        frame4 = FrameDetections(frame_id=4, detections=[Detection("road_obstacle", 0.95)])

        first = builder.process_frame(frame1)
        second = builder.process_frame(frame2)
        third = builder.process_frame(frame4)

        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])
        self.assertEqual(len(third), 1)

    def test_source_frame_ids_include_confirming_frames(self) -> None:
        builder = EventBuilder(EventBuilderConfig(min_consecutive_frames=2, emit_cooldown_frames=5))
        frame1 = FrameDetections(frame_id=1, detections=[Detection("stairs", 0.90)])
        frame2 = FrameDetections(frame_id=2, detections=[Detection("stairs", 0.91)])

        builder.process_frame(frame1)
        events = builder.process_frame(frame2)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source_frame_ids, [1, 2])

    def test_can_disable_debounce_and_cooldown_for_ablation(self) -> None:
        builder = EventBuilder(
            EventBuilderConfig(
                min_consecutive_frames=5,
                emit_cooldown_frames=99,
                enable_debounce=False,
                enable_cooldown=False,
            )
        )
        frame1 = FrameDetections(frame_id=1, detections=[Detection("road_obstacle", 0.95)])
        frame2 = FrameDetections(frame_id=2, detections=[Detection("road_obstacle", 0.95)])

        first = builder.process_frame(frame1)
        second = builder.process_frame(frame2)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0].source_frame_ids, [1])
        self.assertEqual(second[0].source_frame_ids, [2])

    def test_filters_p1_reserved_labels(self) -> None:
        builder = EventBuilder(EventBuilderConfig(min_consecutive_frames=1))
        frame = FrameDetections(
            frame_id=1,
            detections=[
                Detection("traffic_red", 0.99),
                Detection("crosswalk", 0.99),
            ],
        )

        self.assertEqual(builder.process_frame(frame), [])


if __name__ == "__main__":
    unittest.main()
