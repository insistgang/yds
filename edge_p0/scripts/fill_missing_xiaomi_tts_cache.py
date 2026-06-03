#!/usr/bin/env python3
"""Fill missing LinkAble cache files with Xiaomi MiMo TTS fallback audio.

This intentionally writes to the existing LinkAble cache filenames so the
runtime can stay fully offline after generation. Existing non-empty cache files
are never regenerated.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from linkable_edge.audio import (  # noqa: E402
    DEFAULT_MINIMAX_CACHE_DIR,
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_MINIMAX_VOICE,
    MiniMaxTtsParams,
    minimax_cache_path,
)
from linkable_edge.semantics import ALL_TEMPLATE_TEXTS  # noqa: E402


DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_MODEL = "mimo-v2.5-tts"
DEFAULT_VOICE = "mimo_default"
DEFAULT_OUTPUT_DIR = ROOT / "test_results" / "tts_cache"
DEFAULT_STYLE_PROMPT = "请使用清晰、稳定、语速适中的中文女声，语气平稳，适合作为无障碍通行提示。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill missing LinkAble TTS cache files via Xiaomi MiMo TTS.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MINIMAX_CACHE_DIR)
    parser.add_argument("--target-voice", default=DEFAULT_MINIMAX_VOICE)
    parser.add_argument("--target-model", default=DEFAULT_MINIMAX_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("XIAOMI_MIMO_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--mimo-model", default=DEFAULT_MODEL)
    parser.add_argument("--mimo-voice", default=DEFAULT_VOICE)
    parser.add_argument("--style-prompt", default=DEFAULT_STYLE_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = (
        os.environ.get("XIAOMI_MIMO_API_KEY")
        or os.environ.get("MIMO_API_KEY")
        or os.environ.get("TOKEN_PLAN_API_KEY")
    )
    if not api_key:
        print(
            "[ERROR] missing API key; set XIAOMI_MIMO_API_KEY, MIMO_API_KEY, or TOKEN_PLAN_API_KEY",
            file=sys.stderr,
        )
        return 1

    cache_dir = args.cache_dir.expanduser()
    target_params = MiniMaxTtsParams(model=args.target_model)
    missing_texts = [
        text
        for text in ALL_TEMPLATE_TEXTS
        if not is_cached(minimax_cache_path(cache_dir, text, args.target_voice, target_params))
    ]

    print(
        f"[INFO] templates={len(ALL_TEMPLATE_TEXTS)} missing={len(missing_texts)} "
        f"provider=xiaomi_mimo model={args.mimo_model} voice={args.mimo_voice}"
    )

    rows: list[dict[str, Any]] = []
    for idx, text in enumerate(missing_texts, 1):
        output_path = minimax_cache_path(cache_dir, text, args.target_voice, target_params)
        started = time.perf_counter()
        try:
            if is_cached(output_path):
                status = "SKIP"
                size_bytes = output_path.stat().st_size
                error = ""
            else:
                wav_bytes = call_mimo_tts(
                    base_url=args.base_url,
                    api_key=api_key,
                    model=args.mimo_model,
                    voice=args.mimo_voice,
                    text=text,
                    style_prompt=args.style_prompt,
                    timeout_sec=args.timeout_sec,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                convert_wav_bytes_to_mp3(wav_bytes, output_path)
                status = "PASS" if is_cached(output_path) else "FAIL"
                size_bytes = output_path.stat().st_size if output_path.exists() else 0
                error = ""
            elapsed_ms = (time.perf_counter() - started) * 1000
            print(f"[{idx}/{len(missing_texts)}] {status}: {text} -> {output_path.name}")
        except Exception as exc:  # noqa: BLE001 - report every failed text explicitly
            elapsed_ms = (time.perf_counter() - started) * 1000
            status = "FAIL"
            size_bytes = output_path.stat().st_size if output_path.exists() else 0
            error = str(exc)
            print(f"[{idx}/{len(missing_texts)}] FAIL: {text} -> {exc}", file=sys.stderr)

        rows.append(
            {
                "text": text,
                "cache_file": output_path.name,
                "cache_path": str(output_path),
                "status": status,
                "size_bytes": size_bytes,
                "elapsed_ms": round(elapsed_ms, 1),
                "provider": "xiaomi_mimo",
                "provider_model": args.mimo_model,
                "provider_voice": args.mimo_voice,
                "error": error,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "tts_cache_fill_missing_xiaomi.csv"
    json_path = args.output_dir / "tts_cache_fill_missing_xiaomi.json"
    summary = {
        "template_count": len(ALL_TEMPLATE_TEXTS),
        "missing_before_fill": len(missing_texts),
        "filled_count": sum(1 for row in rows if row["status"] in ("PASS", "SKIP")),
        "failed_count": sum(1 for row in rows if row["status"] == "FAIL"),
        "target_voice": args.target_voice,
        "target_model": args.target_model,
        "provider": "xiaomi_mimo",
        "provider_model": args.mimo_model,
        "provider_voice": args.mimo_voice,
        "cache_dir": str(cache_dir),
        "base_url": args.base_url,
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)

    print(f"[RESULT] JSON saved: {json_path}")
    print(f"[RESULT] CSV saved: {csv_path}")
    return 0 if summary["failed_count"] == 0 else 1


def is_cached(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def call_mimo_tts(
    *,
    base_url: str,
    api_key: str,
    model: str,
    voice: str,
    text: str,
    style_prompt: str,
    timeout_sec: float,
) -> bytes:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": style_prompt},
            {"role": "assistant", "content": text},
        ],
        "audio": {
            "format": "wav",
            "voice": voice,
        },
    }
    response = requests.post(
        url,
        headers={
            "api-key": api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_sec,
    )
    try:
        response_json = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Xiaomi MiMo returned non-JSON response, HTTP {response.status_code}") from exc

    if response.status_code >= 400:
        raise RuntimeError(f"Xiaomi MiMo HTTP {response.status_code}: {json.dumps(response_json, ensure_ascii=False)}")

    if "error" in response_json:
        raise RuntimeError(f"Xiaomi MiMo error: {json.dumps(response_json['error'], ensure_ascii=False)}")

    try:
        audio_data = response_json["choices"][0]["message"]["audio"]["data"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Xiaomi MiMo response missing choices[0].message.audio.data") from exc

    try:
        return base64.b64decode(audio_data)
    except ValueError as exc:
        raise RuntimeError("Xiaomi MiMo audio data is not valid base64") from exc


def convert_wav_bytes_to_mp3(wav_bytes: bytes, output_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="linkable_mimo_tts_") as tmp:
        wav_path = Path(tmp) / "input.wav"
        wav_path.write_bytes(wav_bytes)
        completed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-ar",
                "32000",
                "-ac",
                "1",
                "-b:a",
                "128k",
                str(output_path),
            ],
            text=True,
            capture_output=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg mp3 conversion failed: {completed.stderr.strip()}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "text",
        "cache_file",
        "cache_path",
        "status",
        "size_bytes",
        "elapsed_ms",
        "provider",
        "provider_model",
        "provider_voice",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
