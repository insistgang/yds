from __future__ import annotations

import unittest
from typing import Any

from linkable_edge.inputs import FallbackFrameSource, FrameSource


class FakeSource(FrameSource):
    def __init__(self, source_id: str, open_ok: bool, frames: list[Any]) -> None:
        self._source_id = source_id
        self.open_ok = open_ok
        self.frames = list(frames)
        self.released = False
        self.open_calls = 0

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def fps(self) -> float:
        return 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        return (640, 480)

    def open(self) -> bool:
        self.open_calls += 1
        return self.open_ok

    def read(self) -> tuple[bool, Any]:
        if not self.frames:
            return False, None
        frame = self.frames.pop(0)
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self.released = True


class FallbackFrameSourceTests(unittest.TestCase):
    def test_falls_back_when_current_source_read_fails(self) -> None:
        bad = FakeSource("bad", True, [None])
        good = FakeSource("good", True, ["frame-1"])
        source = FallbackFrameSource([("BAD", bad), ("GOOD", good)])

        self.assertTrue(source.open())
        ok, frame = source.read()

        self.assertTrue(ok)
        self.assertEqual(frame, "frame-1")
        self.assertTrue(bad.released)
        self.assertEqual(source.source_id, "good")

    def test_skips_unavailable_source_on_open(self) -> None:
        unavailable = FakeSource("unavailable", False, [])
        good = FakeSource("good", True, ["frame-1"])
        source = FallbackFrameSource([("BAD", unavailable), ("GOOD", good)])

        self.assertTrue(source.open())

        self.assertEqual(unavailable.open_calls, 1)
        self.assertEqual(good.open_calls, 1)
        self.assertEqual(source.source_id, "good")


if __name__ == "__main__":
    unittest.main()
