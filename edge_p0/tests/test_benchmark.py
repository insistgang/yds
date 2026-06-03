from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from linkable_edge.benchmark import BenchmarkCollector, InferenceMetrics


class BenchmarkCollectorTests(unittest.TestCase):
    def test_saves_end_to_end_latency_in_json_and_csv(self) -> None:
        collector = BenchmarkCollector()
        collector.record(
            InferenceMetrics(
                frame_id=1,
                preprocess_ms=2.0,
                inference_ms=20.0,
                postprocess_ms=3.0,
                total_ms=25.0,
                end_to_end_ms=4.5,
            )
        )

        summary = collector.summary()
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["end_to_end_ms"]["avg"], 4.5)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            json_path = out_dir / "benchmark.json"
            csv_path = out_dir / "benchmark.csv"

            collector.save_json(json_path)
            collector.save_csv(csv_path)

            self.assertIn('"end_to_end_ms": 4.5', json_path.read_text(encoding="utf-8"))
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("frame_id,preprocess_ms,inference_ms,postprocess_ms,total_ms,end_to_end_ms", csv_text)
            self.assertIn("1,2.0,20.0,3.0,25.0,4.5", csv_text)


if __name__ == "__main__":
    unittest.main()
