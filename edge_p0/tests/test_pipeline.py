from __future__ import annotations

import unittest

from linkable_edge.audio import AudioOutput
from linkable_edge.event_builder import EventBuilder, EventBuilderConfig
from linkable_edge.models import Detection, FrameDetections
from linkable_edge.pipeline import EdgePipeline


class FakeAudio(AudioOutput):
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


def fixed_clock(values: list[float]):
    remaining = list(values)

    def clock() -> float:
        if remaining:
            return remaining.pop(0)
        return values[-1]

    return clock


class EdgePipelineTests(unittest.TestCase):
    def test_speaks_only_highest_priority_event_per_frame(self) -> None:
        audio = FakeAudio()
        pipeline = EdgePipeline(
            audio_output=audio,
            event_builder=EventBuilder(
                EventBuilderConfig(
                    confidence_threshold=0.25,
                    min_consecutive_frames=1,
                    emit_cooldown_frames=0,
                )
            ),
        )
        frame = FrameDetections(
            frame_id=1,
            detections=[
                Detection("ramp", 0.95, direction="front"),
                Detection("stairs", 0.90, direction="front"),
            ],
        )

        results = pipeline.process_frame(frame)

        self.assertEqual([r.event.label for r in results], ["stairs", "ramp"])
        self.assertEqual(audio.spoken, ["前方有台阶，请减速。"])
        self.assertTrue(results[0].audio_spoken)
        self.assertFalse(results[1].audio_spoken)

    def test_records_detection_to_speak_latency(self) -> None:
        audio = FakeAudio()
        pipeline = EdgePipeline(
            audio_output=audio,
            event_builder=EventBuilder(
                EventBuilderConfig(
                    confidence_threshold=0.25,
                    min_consecutive_frames=1,
                    emit_cooldown_frames=0,
                )
            ),
            clock=fixed_clock([10.0, 10.006]),
        )
        frame = FrameDetections(
            frame_id=1,
            detections=[Detection("road_obstacle", 0.95, direction="front")],
        )

        results = pipeline.process_frame(frame, detection_done_at=10.001)

        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].latency_ms, 5.0, places=3)


if __name__ == "__main__":
    unittest.main()
