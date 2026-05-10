from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest import mock
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tts_minimax_test.py"
SPEC = importlib.util.spec_from_file_location("tts_minimax_test", SCRIPT_PATH)
tts_minimax_test = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = tts_minimax_test
SPEC.loader.exec_module(tts_minimax_test)


class MiniMaxTtsTests(unittest.TestCase):
    def test_cache_hit_skips_api_request(self) -> None:
        params = tts_minimax_test.TtsParams()
        alert = tts_minimax_test.build_alert_cases()[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            output_path = tts_minimax_test.cache_path(cache_dir, alert.text, "female-tianmei", params)
            output_path.write_bytes(b"cached mp3")

            with mock.patch.object(tts_minimax_test, "call_minimax_tts") as call_mock:
                result = tts_minimax_test.synthesize_one(
                    alert=alert,
                    voice_id="female-tianmei",
                    params=params,
                    cache_dir=cache_dir,
                    api_key="test-key",
                    group_id="test-group",
                    api_url="https://api.minimaxi.com/v1/t2a_v2",
                    timeout_sec=1,
                )

            self.assertTrue(result.cache_hit)
            self.assertEqual(result.file_path, output_path)
            call_mock.assert_not_called()

    @mock.patch.object(tts_minimax_test.requests, "post")
    def test_constructs_request_and_writes_mp3_on_cache_miss(self, post_mock: mock.Mock) -> None:
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "data": {"audio": b"mp3-data".hex()},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        post_mock.return_value = response

        params = tts_minimax_test.TtsParams(speed=1.15, emotion="calm")
        alert = tts_minimax_test.build_alert_cases()[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            result = tts_minimax_test.synthesize_one(
                alert=alert,
                voice_id="female-shaonv",
                params=params,
                cache_dir=Path(temp_dir),
                api_key="test-key",
                group_id="test-group",
                api_url="https://api.minimaxi.com/v1/t2a_v2",
                timeout_sec=1,
            )
            audio_bytes = result.file_path.read_bytes()

        self.assertFalse(result.cache_hit)
        self.assertGreaterEqual(result.latency_ms, 0)
        post_mock.assert_called_once()
        _, kwargs = post_mock.call_args
        self.assertEqual(kwargs["params"], {"GroupId": "test-group"})
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["json"]["model"], "speech-2.8-hd")
        self.assertEqual(kwargs["json"]["text"], alert.text)
        self.assertEqual(kwargs["json"]["voice_setting"]["voice_id"], "female-shaonv")
        self.assertEqual(kwargs["json"]["voice_setting"]["speed"], 1.15)
        self.assertEqual(kwargs["json"]["voice_setting"]["emotion"], "calm")
        self.assertEqual(kwargs["json"]["audio_setting"]["format"], "mp3")
        self.assertEqual(audio_bytes, b"mp3-data")

    @mock.patch.object(tts_minimax_test.requests, "post")
    def test_omits_group_id_when_missing(self, post_mock: mock.Mock) -> None:
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "data": {"audio": b"mp3-data".hex()},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        post_mock.return_value = response

        tts_minimax_test.call_minimax_tts(
            text="前方2米有障碍，请注意避让。",
            voice_id="female-tianmei",
            params=tts_minimax_test.TtsParams(),
            api_key="test-key",
            group_id=None,
            api_url="https://api.minimaxi.com/v1/t2a_v2",
            timeout_sec=1,
        )

        _, kwargs = post_mock.call_args
        self.assertIsNone(kwargs["params"])

    def test_omits_empty_emotion(self) -> None:
        payload = tts_minimax_test.build_request_payload(
            text="前方2米有障碍，请注意避让。",
            voice_id="female-tianmei",
            params=tts_minimax_test.TtsParams(),
        )

        self.assertNotIn("emotion", payload["voice_setting"])

    def test_estimates_hd_and_turbo_paygo_cost(self) -> None:
        alerts = tts_minimax_test.build_alert_cases()

        turbo_cost = tts_minimax_test.estimate_paygo_cost_yuan(alerts, 4, "speech-02-turbo")
        hd_cost = tts_minimax_test.estimate_paygo_cost_yuan(alerts, 4, "speech-2.8-hd")

        self.assertAlmostEqual(turbo_cost, 0.1104, places=4)
        self.assertAlmostEqual(hd_cost, 0.1932, places=4)


if __name__ == "__main__":
    unittest.main()
