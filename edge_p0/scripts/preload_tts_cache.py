"""预加载全部 P0 模板 TTS 缓存"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from linkable_edge.audio import MiniMaxAudioOutput, MiniMaxTtsParams, DEFAULT_MINIMAX_VOICE
from linkable_edge.semantics import ALL_TEMPLATE_TEXTS


def main():
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("ERROR: MINIMAX_API_KEY not set")
        return 1

    tts = MiniMaxAudioOutput(
        voice_id=DEFAULT_MINIMAX_VOICE,
        params=MiniMaxTtsParams(),
        api_key=api_key,
    )

    total = len(ALL_TEMPLATE_TEXTS)
    cached = 0
    failed = 0

    for i, text in enumerate(ALL_TEMPLATE_TEXTS, 1):
        try:
            path = tts._get_audio_path(text)
            if path.exists():
                cached += 1
                print(f"[{i}/{total}] OK: {text[:30]}... -> {path.name}")
            else:
                failed += 1
                print(f"[{i}/{total}] FAIL: {text[:30]}...")
        except Exception as e:
            failed += 1
            print(f"[{i}/{total}] ERROR: {text[:30]}... -> {e}")

    print(f"\nDone: {cached}/{total} cached, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
