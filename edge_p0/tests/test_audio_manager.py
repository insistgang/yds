from __future__ import annotations

import threading
import time
import unittest

from linkable_edge.audio import AudioOutput
from linkable_edge.audio_manager import AudioManager


class RecordingAudioOutput(AudioOutput):
    def __init__(self) -> None:
        self.events: list[str] = []

    def speak(self, text: str) -> None:
        self.events.append(f"start:{text}")
        time.sleep(0.02)
        self.events.append(f"end:{text}")


class AudioManagerTests(unittest.TestCase):
    def test_audio_lock_serializes_concurrent_speak_calls(self) -> None:
        backend = RecordingAudioOutput()
        manager = AudioManager.__new__(AudioManager)
        manager._lock = threading.Lock()
        manager._fallback_chain = [backend]
        manager._cache_dir = None

        threads = [
            threading.Thread(target=manager.speak, args=("A",)),
            threading.Thread(target=manager.speak, args=("B",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertIn(
            backend.events,
            [
                ["start:A", "end:A", "start:B", "end:B"],
                ["start:B", "end:B", "start:A", "end:A"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
