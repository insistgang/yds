#!/usr/bin/env python3
"""Check whether ALL_TEMPLATE_TEXTS has one cached TTS file per template."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

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


DEFAULT_OUTPUT_DIR = ROOT / "test_results" / "tts_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local MiniMax TTS cache coverage for P0 templates.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MINIMAX_CACHE_DIR)
    parser.add_argument("--voice", default=DEFAULT_MINIMAX_VOICE)
    parser.add_argument("--model", default=DEFAULT_MINIMAX_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = MiniMaxTtsParams(model=args.model)
    rows = []
    for text in ALL_TEMPLATE_TEXTS:
        path = minimax_cache_path(args.cache_dir.expanduser(), text, args.voice, params)
        cached = path.exists() and path.stat().st_size > 0
        rows.append(
            {
                "text": text,
                "cache_path": str(path),
                "cached": cached,
                "size_bytes": path.stat().st_size if cached else 0,
            }
        )

    cached_count = sum(1 for row in rows if row["cached"])
    missing = [row for row in rows if not row["cached"]]
    summary = {
        "template_count": len(ALL_TEMPLATE_TEXTS),
        "unique_template_count": len(set(ALL_TEMPLATE_TEXTS)),
        "cached_count": cached_count,
        "missing_count": len(missing),
        "cache_dir": str(args.cache_dir.expanduser()),
        "voice": args.voice,
        "model": args.model,
        "strict_one_to_one": cached_count == len(ALL_TEMPLATE_TEXTS) and len(set(ALL_TEMPLATE_TEXTS)) == len(ALL_TEMPLATE_TEXTS),
        "missing_texts": [row["text"] for row in missing],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "tts_cache_check.json"
    csv_path = args.output_dir / "tts_cache_check.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)

    print(f"[RESULT] JSON saved: {json_path}")
    print(f"[RESULT] CSV saved: {csv_path}")
    print(
        f"[SUMMARY] templates={summary['template_count']} unique={summary['unique_template_count']} "
        f"cached={summary['cached_count']} missing={summary['missing_count']} "
        f"strict_one_to_one={summary['strict_one_to_one']}"
    )
    if missing:
        print("[WARN] cache/template mismatch; missing cached audio for:")
        for row in missing[:10]:
            print(f"  - {row['text']}")
        if len(missing) > 10:
            print(f"  ... {len(missing) - 10} more")

    return 0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "cached", "size_bytes", "cache_path"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
