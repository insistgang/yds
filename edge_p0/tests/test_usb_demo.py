from __future__ import annotations

import unittest

from linkable_edge.audio import AudioOutput
from linkable_edge.models import Detection, FrameDetections
from linkable_edge.publisher import EventPublisher
from linkable_edge.usb_demo import USB_FPS, USB_HEIGHT, USB_WIDTH, UsbDemoConfig, run_usb_demo


class FakeCapture:
    def __init__(self, frames: list[object]) -> None:
        self.frames = list(frames)
        self.set_calls: list[tuple[int, object]] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: object) -> bool:
        self.set_calls.append((prop, value))
        return True

    def read(self) -> tuple[bool, object | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class FakeCv2:
    CAP_V4L2 = 200
    CAP_PROP_FOURCC = 6
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5

    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture
        self.video_capture_args: tuple[int, int] | None = None
        self.saved_paths: list[str] = []

    def VideoCapture(self, device_idx: int, backend: int) -> FakeCapture:
        self.video_capture_args = (device_idx, backend)
        return self.capture

    def VideoWriter_fourcc(self, *chars: str) -> int:
        self.fourcc_chars = chars
        return 1196444237

    def imwrite(self, path: str, frame: object) -> bool:
        self.saved_paths.append(path)
        return True


class FakeDetector:
    def __init__(self) -> None:
        self.calls: list[tuple[object, int]] = []

    def predict_frame(self, image: object, frame_id: int) -> FrameDetections:
        self.calls.append((image, frame_id))
        return FrameDetections(
            frame_id=frame_id,
            detections=[Detection("road_obstacle", 0.95, bbox=(10, 20, 100, 200), direction="front")],
        )


class FakeAudio(AudioOutput):
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakePublisher(EventPublisher):
    def __init__(self) -> None:
        self.events = []

    def publish(self, events) -> None:  # type: ignore[no-untyped-def]
        self.events.extend(list(events))


def fixed_clock(values: list[float]):
    remaining = list(values)

    def clock() -> float:
        if remaining:
            return remaining.pop(0)
        return values[-1]

    return clock


class UsbDemoTests(unittest.TestCase):
    def test_reads_camera_frame_and_publishes_pipeline_event(self) -> None:
        capture = FakeCapture(frames=["frame-1"])
        cv2 = FakeCv2(capture)
        detector = FakeDetector()
        audio = FakeAudio()
        publisher = FakePublisher()
        config = UsbDemoConfig(
            audio="print",
            save_interval_sec=0,
            audio_cooldown_sec=5,
            min_consecutive_frames=1,
            event_emit_cooldown_frames=1,
        )

        stats = run_usb_demo(
            config,
            detector=detector,
            audio_output=audio,
            publisher=publisher,
            cv2_module=cv2,
            max_frames=1,
            clock=fixed_clock([0, 0, 0]),
        )

        self.assertEqual(cv2.video_capture_args, (0, FakeCv2.CAP_V4L2))
        self.assertIn((FakeCv2.CAP_PROP_FOURCC, 1196444237), capture.set_calls)
        self.assertIn((FakeCv2.CAP_PROP_FRAME_WIDTH, USB_WIDTH), capture.set_calls)
        self.assertIn((FakeCv2.CAP_PROP_FRAME_HEIGHT, USB_HEIGHT), capture.set_calls)
        self.assertIn((FakeCv2.CAP_PROP_FPS, USB_FPS), capture.set_calls)
        self.assertEqual(detector.calls, [("frame-1", 1)])
        self.assertEqual(len(publisher.events), 1)
        self.assertEqual(publisher.events[0].label, "road_obstacle")
        self.assertEqual(audio.spoken, ["前方有障碍，请注意避让。"])
        self.assertEqual(stats.frames, 1)
        self.assertEqual(stats.events, 1)
        self.assertEqual(stats.speeches, 1)
        self.assertTrue(capture.released)

    def test_audio_cooldown_suppresses_repeated_event_type(self) -> None:
        capture = FakeCapture(frames=["frame-1", "frame-2"])
        cv2 = FakeCv2(capture)
        detector = FakeDetector()
        audio = FakeAudio()
        publisher = FakePublisher()
        config = UsbDemoConfig(
            audio="print",
            save_interval_sec=0,
            audio_cooldown_sec=5,
            min_consecutive_frames=1,
            event_emit_cooldown_frames=1,
        )

        stats = run_usb_demo(
            config,
            detector=detector,
            audio_output=audio,
            publisher=publisher,
            cv2_module=cv2,
            max_frames=2,
            clock=fixed_clock([0, 0, 0, 1, 1]),
        )

        self.assertEqual(len(publisher.events), 2)
        self.assertEqual(stats.events, 2)
        self.assertEqual(stats.speeches, 1)
        self.assertEqual(audio.spoken, ["前方有障碍，请注意避让。"])


if __name__ == "__main__":
    unittest.main()
