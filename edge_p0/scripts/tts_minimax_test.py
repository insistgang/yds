from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from linkable_edge.models import DetectionEvent
from linkable_edge.semantics import render_event_text


DEFAULT_API_URL = "https://api.minimaxi.com/v1/t2a_v2"
DEFAULT_MODEL = "speech-2.8-hd"
DEFAULT_CACHE_DIR = Path("~/.cache/linkable_edge/tts_minimax").expanduser()
LINKABLE_DEFAULT_VOICE = "presenter_female"
DEFAULT_VOICES = [
    "female-tianmei",
    "female-shaonv",
    "presenter_female",
    "female-yujie",
]


@dataclass(slots=True)
class AlertCase:
    event_tag: str
    event: DetectionEvent

    @property
    def text(self) -> str:
        return render_event_text(self.event)


@dataclass(slots=True)
class TtsParams:
    model: str = DEFAULT_MODEL
    speed: float = 1.0
    vol: float = 1.0
    pitch: int = 0
    emotion: str = ""
    sample_rate: int = 32000
    bitrate: int = 128000
    audio_format: str = "mp3"
    channel: int = 1
    language_boost: str = "Chinese"


@dataclass(slots=True)
class SynthesisResult:
    voice: str
    event_tag: str
    latency_ms: int
    cache_hit: bool
    file_path: Path
    ok: bool = True


def build_alert_cases() -> list[AlertCase]:
    now = datetime.now(timezone.utc)
    return [
        AlertCase(
            "road_obstacle",
            DetectionEvent(
                event_id="tts-road-obstacle",
                label="road_obstacle",
                confidence=0.92,
                timestamp=now,
                priority=80,
                distance_m=2.0,
                direction="front",
            ),
        ),
        AlertCase(
            "blind_path_block",
            DetectionEvent(
                event_id="tts-blind-path-block",
                label="blind_road_occupied",
                confidence=0.93,
                timestamp=now,
                priority=70,
                distance_m=8.0,
                direction="right_front",
            ),
        ),
        AlertCase(
            "stairs_down",
            DetectionEvent(
                event_id="tts-stairs-down",
                label="stairs",
                confidence=0.91,
                timestamp=now,
                priority=100,
                distance_m=2.0,
                direction="front",
            ),
        ),
        AlertCase(
            "traffic_red",
            DetectionEvent(
                event_id="tts-traffic-red",
                label="traffic_red",
                confidence=0.95,
                timestamp=now,
                priority=95,
                distance_m=5.0,
                direction="front",
            ),
        ),
        AlertCase(
            "crosswalk",
            DetectionEvent(
                event_id="tts-crosswalk",
                label="crosswalk",
                confidence=0.90,
                timestamp=now,
                priority=40,
                distance_m=4.0,
                direction="front",
            ),
        ),
    ]


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}，请先设置后再运行。")
    return value


def optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value or None


def build_request_payload(text: str, voice_id: str, params: TtsParams) -> dict[str, Any]:
    voice_setting: dict[str, Any] = {
        "voice_id": voice_id,
        "speed": params.speed,
        "vol": params.vol,
        "pitch": params.pitch,
    }
    if params.emotion:
        voice_setting["emotion"] = params.emotion

    payload: dict[str, Any] = {
        "model": params.model,
        "text": text,
        "stream": False,
        "voice_setting": voice_setting,
        "audio_setting": {
            "sample_rate": params.sample_rate,
            "bitrate": params.bitrate,
            "format": params.audio_format,
            "channel": params.channel,
        },
        "language_boost": params.language_boost,
        "subtitle_enable": False,
        "output_format": "hex",
    }
    return payload


def cache_key(text: str, voice_id: str, params: TtsParams) -> str:
    identity = {
        "model": params.model,
        "voice_id": voice_id,
        "speed": params.speed,
        "vol": params.vol,
        "pitch": params.pitch,
        "emotion": params.emotion,
        "sample_rate": params.sample_rate,
        "bitrate": params.bitrate,
        "format": params.audio_format,
        "channel": params.channel,
        "language_boost": params.language_boost,
        "text": text,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_path(cache_dir: Path, text: str, voice_id: str, params: TtsParams) -> Path:
    return cache_dir / f"{cache_key(text, voice_id, params)}.{params.audio_format}"


def format_base_resp(response_json: dict[str, Any]) -> str:
    base_resp = response_json.get("base_resp")
    if base_resp is None:
        return "base_resp 缺失"
    return json.dumps(base_resp, ensure_ascii=False)


def call_minimax_tts(
    text: str,
    voice_id: str,
    params: TtsParams,
    api_key: str,
    group_id: str | None,
    api_url: str,
    timeout_sec: float,
) -> bytes:
    payload = build_request_payload(text, voice_id, params)
    request_params = {"GroupId": group_id} if group_id else None
    response = requests.post(
        api_url,
        params=request_params,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_sec,
    )

    try:
        response_json = response.json()
    except ValueError as exc:
        raise RuntimeError(f"MiniMax 返回非 JSON 响应，HTTP {response.status_code}: {response.text[:300]}") from exc

    base_resp = response_json.get("base_resp") or {}
    if response.status_code >= 400 or base_resp.get("status_code") not in (0, "0", None):
        raise RuntimeError(f"MiniMax 请求失败：{format_base_resp(response_json)}")

    data = response_json.get("data") or {}
    audio_hex = data.get("audio")
    if not audio_hex:
        raise RuntimeError(f"MiniMax 响应缺少 data.audio：{format_base_resp(response_json)}")

    try:
        return bytes.fromhex(audio_hex)
    except ValueError as exc:
        raise RuntimeError(f"MiniMax 返回的 data.audio 不是合法 hex：{format_base_resp(response_json)}") from exc


def synthesize_one(
    alert: AlertCase,
    voice_id: str,
    params: TtsParams,
    cache_dir: Path,
    api_key: str,
    group_id: str | None,
    api_url: str,
    timeout_sec: float,
) -> SynthesisResult:
    text = alert.text
    output_path = cache_path(cache_dir, text, voice_id, params)
    start = time.perf_counter()

    if output_path.exists() and output_path.stat().st_size > 0:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return SynthesisResult(voice_id, alert.event_tag, latency_ms, True, output_path)

    audio_bytes = call_minimax_tts(
        text=text,
        voice_id=voice_id,
        params=params,
        api_key=api_key,
        group_id=group_id,
        api_url=api_url,
        timeout_sec=timeout_sec,
    )
    output_path.write_bytes(audio_bytes)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return SynthesisResult(voice_id, alert.event_tag, latency_ms, False, output_path)


def resolve_player(requested_player: str) -> list[str] | None:
    if requested_player != "auto":
        executable = shutil.which(requested_player)
        if executable is None:
            raise RuntimeError(f"找不到播放器 {requested_player}。")
        return [executable]

    mpg123 = shutil.which("mpg123")
    if mpg123 is not None:
        return [mpg123, "-q"]

    aplay = shutil.which("aplay")
    if aplay is not None:
        return [aplay]

    return None


def play_file(file_path: Path, requested_player: str) -> None:
    command = resolve_player(requested_player)
    if command is None:
        print(f"[WARN] 未找到 mpg123/aplay，跳过播放：{file_path}", file=sys.stderr)
        return

    completed = subprocess.run([*command, str(file_path)], text=True)
    if completed.returncode != 0:
        print(f"[WARN] 播放失败 rc={completed.returncode}: {file_path}", file=sys.stderr)


def print_table(results: list[SynthesisResult]) -> None:
    print("voice | event_tag | latency_ms | cache_hit | file_path")
    print("--- | --- | ---: | --- | ---")
    for result in results:
        cache_hit = "true" if result.cache_hit else "false"
        print(f"{result.voice} | {result.event_tag} | {result.latency_ms} | {cache_hit} | {result.file_path}")


def paygo_price_yuan_per_10k(model: str) -> float:
    if model.endswith("-turbo"):
        return 2.0
    if model.endswith("-hd"):
        return 3.5
    return 2.0


def estimate_paygo_cost_yuan(alerts: list[AlertCase], request_count_by_text: int, model: str) -> float:
    total_units = 0
    for alert in alerts:
        for char in alert.text:
            total_units += 2 if "\u4e00" <= char <= "\u9fff" else 1
    total_units *= request_count_by_text
    return total_units / 10_000 * paygo_price_yuan_per_10k(model)


def export_named_files(results: list[SynthesisResult], export_dir: Path, model: str) -> None:
    for result in results:
        target_dir = export_dir / model / result.voice
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.file_path, target_dir / f"{result.event_tag}.mp3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 LinkAble 告警文案的 MiniMax TTS 合成和本地播放效果。")
    parser.add_argument("--voice", action="append", help="只测试指定音色；可重复传入。默认遍历候选音色。")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--vol", type=float, default=1.0)
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--emotion", default="", help="可选手动情绪，例如 calm、happy；默认留空由模型自动判断。")
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--bitrate", type=int, default=128000)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--language-boost", default="Chinese")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--export-dir", type=Path, default=None, help="可选：把缓存 mp3 复制成 voice/event_tag.mp3 便于试听。")
    parser.add_argument("--api-url", default=os.environ.get("MINIMAX_API_URL", DEFAULT_API_URL))
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--no-play", action="store_true", help="只合成不播放，适合离线批量生成。")
    parser.add_argument("--player", default="auto", help="播放器：auto、mpg123 或 aplay。默认 auto。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        api_key = require_env("MINIMAX_API_KEY")
        group_id = optional_env("MINIMAX_GROUP_ID")
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    params = TtsParams(
        model=args.model,
        speed=args.speed,
        vol=args.vol,
        pitch=args.pitch,
        emotion=args.emotion,
        sample_rate=args.sample_rate,
        bitrate=args.bitrate,
        channel=args.channel,
        language_boost=args.language_boost,
    )
    voices = args.voice or DEFAULT_VOICES
    alerts = build_alert_cases()
    cache_dir = args.cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: list[SynthesisResult] = []
    failures = 0
    for voice_id in voices:
        for alert in alerts:
            try:
                result = synthesize_one(
                    alert=alert,
                    voice_id=voice_id,
                    params=params,
                    cache_dir=cache_dir,
                    api_key=api_key,
                    group_id=group_id,
                    api_url=args.api_url,
                    timeout_sec=args.timeout_sec,
                )
                results.append(result)
                if not args.no_play:
                    play_file(result.file_path, args.player)
            except Exception as exc:
                failures += 1
                print(f"[ERROR] voice={voice_id} event_tag={alert.event_tag} {exc}", file=sys.stderr)

    print_table(results)
    cold = [result.latency_ms for result in results if not result.cache_hit]
    hits = [result.latency_ms for result in results if result.cache_hit]
    if cold:
        print(f"cold_avg_ms={sum(cold) / len(cold):.1f}")
    if hits:
        print(f"cache_hit_avg_ms={sum(hits) / len(hits):.1f}")
    if args.export_dir is not None:
        export_named_files(results, args.export_dir.expanduser(), params.model)
        print(f"export_dir={args.export_dir.expanduser() / params.model}")
    print(f"paygo_cost_estimate_yuan={estimate_paygo_cost_yuan(alerts, len(voices), params.model):.4f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
