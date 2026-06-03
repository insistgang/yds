from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_video_report_dashboard.py"
SPEC = importlib.util.spec_from_file_location("build_video_report_dashboard", SCRIPT_PATH)
build_video_report_dashboard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = build_video_report_dashboard
SPEC.loader.exec_module(build_video_report_dashboard)


class VideoReportDashboardTests(unittest.TestCase):
    def test_builds_json_and_static_dashboard_outputs(self) -> None:
        analysis = build_video_report_dashboard.VideoAnalysis(
            video_name="1",
            video_path="test_videos/1.mp4",
            fps=30.0,
            frame_size=(1280, 720),
            total_frames=60,
            total_detections=4,
            total_events=1,
            duration_sec=2.0,
        )
        analysis.events_by_type["road_obstacle"] = 1
        analysis.timeline_by_10s["0"] = 1
        analysis.events.append(
            {
                "event_id": "1-road_obstacle-2",
                "video": "1",
                "type": "road_obstacle",
                "type_cn": "路面障碍",
                "confidence": 0.9,
                "direction": "front",
                "message": "前方有障碍，请注意避让。",
                "frame_id": 2,
                "video_time_sec": 0.067,
                "bbox": [1, 2, 3, 4],
                "source_frame_ids": [1, 2],
            }
        )

        args = argparse.Namespace(
            videos_dir=Path("test_videos"),
            video_glob="*.mp4",
            model="best.pt",
            conf=0.25,
            event_conf=0.55,
            min_frames=2,
            cooldown_frames=5,
            max_frames=None,
        )
        report = build_video_report_dashboard.build_report([analysis], args=args, elapsed_sec=1.2)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            build_video_report_dashboard.write_outputs(report, output_dir)
            full_report = json.loads((output_dir / "video_report.json").read_text(encoding="utf-8"))
            html = (output_dir / "dashboard.html").read_text(encoding="utf-8")

        self.assertEqual(full_report["summary"]["video_count"], 1)
        self.assertEqual(full_report["summary"]["total_events"], 1)
        self.assertIn("LinkAble 本地视频数据大屏", html)
        self.assertIn("road_obstacle", html)


if __name__ == "__main__":
    unittest.main()
