#!/usr/bin/env python3
"""Fill only missing MiniMax TTS cache files for ALL_TEMPLATE_TEXTS."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from linkable_edge.audio import (  # noqa: E402
    DEFAULT_MINIMAX_API_URL,
    DEFAULT_MINIMAX_CACHE_DIR,
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_MINIMAX_VOICE,
    MiniMaxAudioOutput,
    MiniMaxTtsParams,
    minimax_cache_path,
)
from linkable_edge.semantics import ALL_TEMPLATE_TEXTS  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "test_results" / "tts_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate only missing LinkAble P0 TTS cache files.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MINIMAX_CACHE_DIR)
    parser.add_argument("--voice", default=DEFAULT_MINIMAX_VOICE)
    parser.add_argument("--model", default=DEFAULT_MINIMAX_MODEL)
    parser.add_argument("--api-url", default=DEFAULT_MINIMAX_API_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    return parser.parse_args()


def is_cached(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser()
    params = MiniMaxTtsParams(model=args.model)
    output = MiniMaxAudioOutput(
        voice_id=args.voice,
        params=params,
        cache_dir=cache_dir,
        api_url=args.api_url,
        timeout_sec=args.timeout_sec,
    )

    rows: list[dict[str, object]] = []
    missing_texts = [
        text for text in ALL_TEMPLATE_TEXTS if not is_cached(minimax_cache_path(cache_dir, text, args.voice, params))
    ]

    print(
        f"[INFO] templates={len(ALL_TEMPLATE_TEXTS)} "
        f"missing={len(missing_texts)} voice={args.voice} model={args.model}"
    )
    if not missing_texts:
        print("[INFO] no missing cache files")

    for idx, text in enumerate(missing_texts, 1):
        expected_path = minimax_cache_path(cache_dir, text, args.voice, params)
        started = time.perf_counter()
        try:
            actual_path = output._get_audio_path(text)
            elapsed_ms = (time.perf_counter() - started) * 1000
            status = "PASS" if actual_path == expected_path and is_cached(actual_path) else "FAIL"
            error = ""
            size_bytes = actual_path.stat().st_size if actual_path.exists() else 0
            print(f"[{idx}/{len(missing_texts)}] {status}: {text} -> {actual_path.name}")
        except Exception as exc:  # noqa: BLE001 - report every failed text explicitly
            elapsed_ms = (time.perf_counter() - started) * 1000
            actual_path = expected_path
            status = "FAIL"
            error = str(exc)
            size_bytes = 0
            print(f"[{idx}/{len(missing_texts)}] FAIL: {text} -> {exc}", file=sys.stderr)

        rows.append(
            {
                "text": text,
                "cache_file": actual_path.name,
                "cache_path": str(actual_path),
                "status": status,
                "size_bytes": size_bytes,
                "elapsed_ms": round(elapsed_ms, 1),
                "error": error,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "tts_cache_fill_missing.json"
    csv_path = args.output_dir / "tts_cache_fill_missing.csv"
    summary = {
        "template_count": len(ALL_TEMPLATE_TEXTS),
        "missing_before_fill": len(missing_texts),
        "filled_count": sum(1 for row in rows if row["status"] == "PASS"),
        "failed_count": sum(1 for row in rows if row["status"] != "PASS"),
        "voice": args.voice,
        "model": args.model,
        "cache_dir": str(cache_dir),
        "api_url": args.api_url,
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)

    print(f"[RESULT] JSON saved: {json_path}")
    print(f"[RESULT] CSV saved: {csv_path}")
    return 0 if summary["failed_count"] == 0 else 1


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["text", "cache_file", "cache_path", "status", "size_bytes", "elapsed_ms", "error"]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
