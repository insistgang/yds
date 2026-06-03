#!/usr/bin/env python3
"""Validate LinkAble P0 TTS cache coverage without allowing online fallback."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from linkable_edge import audio as audio_module  # noqa: E402
from linkable_edge.audio import (  # noqa: E402
    DEFAULT_MINIMAX_CACHE_DIR,
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_MINIMAX_VOICE,
    MiniMaxAudioOutput,
    MiniMaxTtsParams,
    minimax_cache_path,
)
from linkable_edge.audio_manager import AudioManager  # noqa: E402
from linkable_edge.semantics import ALL_TEMPLATE_TEXTS  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "test_results" / "tts_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate all P0 template audio files are cached offline.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MINIMAX_CACHE_DIR)
    parser.add_argument("--voice", default=DEFAULT_MINIMAX_VOICE)
    parser.add_argument("--model", default=DEFAULT_MINIMAX_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser()
    params = MiniMaxTtsParams(model=args.model)
    rows = build_rows(cache_dir, args.voice, params)

    preload_pass, preload_errors = validate_preload_without_api(cache_dir, args.voice, params)
    offline_speak_errors, played_paths = validate_speak_without_api(cache_dir, args.voice, params, rows)

    for row in rows:
        row["offline_speak_status"] = "PASS" if row["text"] not in offline_speak_errors else "FAIL"
        row["error"] = offline_speak_errors.get(row["text"], "")
        row["status"] = "PASS" if row["cache_status"] == "PASS" and row["offline_speak_status"] == "PASS" else "FAIL"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "tts_cache_validation.csv"
    json_path = args.output_dir / "tts_cache_validation.json"
    write_csv(csv_path, rows)

    pass_count = sum(1 for row in rows if row["status"] == "PASS")
    summary = {
        "template_count": len(rows),
        "pass_count": pass_count,
        "failed_count": len(rows) - pass_count,
        "cache_hit_count": sum(1 for row in rows if row["cache_status"] == "PASS"),
        "offline_speak_pass_count": sum(1 for row in rows if row["offline_speak_status"] == "PASS"),
        "offline_speak_count": len(played_paths),
        "online_api_allowed": False,
        "online_fallback_count": 0 if not offline_speak_errors else None,
        "preload_pass": preload_pass,
        "preload_errors": preload_errors,
        "voice": args.voice,
        "model": args.model,
        "cache_dir": str(cache_dir),
    }
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[RESULT] CSV saved: {csv_path}")
    print(f"[RESULT] JSON saved: {json_path}")
    print(
        f"[SUMMARY] templates={summary['template_count']} pass={summary['pass_count']} "
        f"cache_hits={summary['cache_hit_count']} offline_speak={summary['offline_speak_pass_count']} "
        f"preload_pass={summary['preload_pass']}"
    )
    return 0 if summary["failed_count"] == 0 and preload_pass else 1


def build_rows(cache_dir: Path, voice: str, params: MiniMaxTtsParams) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for text in ALL_TEMPLATE_TEXTS:
        path = minimax_cache_path(cache_dir, text, voice, params)
        cached = path.exists() and path.stat().st_size > 0
        rows.append(
            {
                "text": text,
                "cache_file": path.name,
                "cache_path": str(path),
                "size_bytes": path.stat().st_size if cached else 0,
                "cache_status": "PASS" if cached else "FAIL",
            }
        )
    return rows


def validate_preload_without_api(cache_dir: Path, voice: str, params: MiniMaxTtsParams) -> tuple[bool, list[str]]:
    old_key = os.environ.pop("MINIMAX_API_KEY", None)
    old_group = os.environ.pop("MINIMAX_GROUP_ID", None)
    errors: list[str] = []
    try:
        manager = AudioManager(voice=voice, model=params.model, cache_dir=cache_dir)
        manager.preload(ALL_TEMPLATE_TEXTS)
        checker = MiniMaxAudioOutput(voice_id=voice, params=params, cache_dir=cache_dir, api_key=None, group_id=None)
        for text in ALL_TEMPLATE_TEXTS:
            try:
                checker._get_audio_path(text)
            except Exception as exc:  # noqa: BLE001 - report exact preload miss
                errors.append(f"{text}: {exc}")
    finally:
        restore_env(old_key, old_group)
    return not errors, errors


def validate_speak_without_api(
    cache_dir: Path,
    voice: str,
    params: MiniMaxTtsParams,
    rows: list[dict[str, object]],
) -> tuple[dict[str, str], list[str]]:
    old_key = os.environ.pop("MINIMAX_API_KEY", None)
    old_group = os.environ.pop("MINIMAX_GROUP_ID", None)
    original_player = audio_module.play_audio_file
    played_paths: list[str] = []
    errors: dict[str, str] = {}

    def fake_play(file_path: Path, requested_player: str = "auto") -> None:  # noqa: ARG001
        played_paths.append(str(file_path))

    audio_module.play_audio_file = fake_play
    try:
        tts = MiniMaxAudioOutput(
            voice_id=voice,
            params=params,
            cache_dir=cache_dir,
            api_key=None,
            group_id=None,
            player="auto",
        )
        for row in rows:
            text = str(row["text"])
            try:
                tts.speak(text)
            except Exception as exc:  # noqa: BLE001 - no silent fallback
                errors[text] = str(exc)
    finally:
        audio_module.play_audio_file = original_player
        restore_env(old_key, old_group)

    return errors, played_paths


def restore_env(old_key: str | None, old_group: str | None) -> None:
    if old_key is not None:
        os.environ["MINIMAX_API_KEY"] = old_key
    if old_group is not None:
        os.environ["MINIMAX_GROUP_ID"] = old_group


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "text",
        "cache_file",
        "cache_status",
        "offline_speak_status",
        "status",
        "size_bytes",
        "cache_path",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
