from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_report_sample.py"
SPEC = importlib.util.spec_from_file_location("generate_report_sample", SCRIPT_PATH)
generate_report_sample = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = generate_report_sample
SPEC.loader.exec_module(generate_report_sample)


class GenerateReportSampleTests(unittest.TestCase):
    def test_call_mimo_uses_token_plan_chat_completions(self) -> None:
        events = [
            {
                "event_id": "evt-1",
                "timestamp": "2026-06-01T08:10:00+08:00",
                "type": "road_obstacle",
                "confidence": 0.8,
                "location": "A点",
            }
        ]
        stats = {
            "total_events": 1,
            "by_type": {"road_obstacle": 1},
            "by_location": {"A点": 1},
            "by_day": {"2026-06-01": 1},
            "avg_confidence": 0.8,
            "peak_hour": {"hour": 8, "count": 1},
            "worst_location": {"name": "A点", "count": 1},
        }

        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "# LinkAble 周度无障碍环境监测报告"}}]
        }

        with mock.patch("httpx.post", return_value=response) as post_mock:
            report = generate_report_sample._call_mimo(
                "token-plan-key",
                events,
                stats,
                base_url="https://token-plan-cn.xiaomimimo.com/v1",
                model="mino-2.5-pro",
            )

        self.assertIn("LinkAble", report)
        _, kwargs = post_mock.call_args
        self.assertEqual(
            post_mock.call_args.args[0],
            "https://token-plan-cn.xiaomimimo.com/v1/chat/completions",
        )
        self.assertEqual(kwargs["headers"]["api-key"], "token-plan-key")
        self.assertEqual(kwargs["json"]["model"], "mimo-v2.5-pro")
        self.assertEqual(kwargs["json"]["temperature"], 0.3)
        self.assertEqual(kwargs["json"]["messages"][0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
