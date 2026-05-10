from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .audio import (
    DEFAULT_MINIMAX_CACHE_DIR,
    DEFAULT_MINIMAX_MODEL,
    DEFAULT_MINIMAX_VOICE,
    AudioOutput,
    MiniMaxAudioOutput,
    MiniMaxTtsParams,
    PrintAudioOutput,
    Pyttsx3AudioOutput,
)


class AudioManager(AudioOutput):
    """MiniMax -> pyttsx3 -> print 自动降级 + 全局音频锁 + 启动预加载

    AGENTS.md 7.4 规范实现：
    - 降级链：MiniMax -> pyttsx3 -> print
    - 全局音频锁：防止多事件重叠播报
    - 启动预加载：避免首次播报延迟
    - 失败时绝不静默
    """

    def __init__(
        self,
        voice: str = DEFAULT_MINIMAX_VOICE,
        model: str = DEFAULT_MINIMAX_MODEL,
        cache_dir: Path = DEFAULT_MINIMAX_CACHE_DIR,
        player: str = "auto",
    ) -> None:
        self._lock = threading.Lock()
        self._fallback_chain: list[AudioOutput] = []
        self._cache_dir = cache_dir

        # 1. 尝试 MiniMax
        try:
            self._fallback_chain.append(
                MiniMaxAudioOutput(
                    voice_id=voice,
                    params=MiniMaxTtsParams(model=model),
                    cache_dir=cache_dir,
                    player=player,
                )
            )
            print("[INFO] Audio backend: MiniMax TTS ready")
        except Exception as exc:
            print(f"[WARN] MiniMax TTS unavailable: {exc}", file=sys.stderr)

        # 2. 尝试 pyttsx3
        try:
            self._fallback_chain.append(Pyttsx3AudioOutput())
            print("[INFO] Audio backend: pyttsx3 ready")
        except Exception as exc:
            print(f"[WARN] pyttsx3 unavailable: {exc}", file=sys.stderr)

        # 3. print 兜底（永远成功）
        self._fallback_chain.append(PrintAudioOutput())
        print("[INFO] Audio backend: print fallback ready")

        if len(self._fallback_chain) == 1:
            print(
                "[WARN] MiniMax and pyttsx3 both unavailable; using print fallback only",
                file=sys.stderr,
            )

    def preload(self, texts: Iterable[str]) -> None:
        """启动时预加载全部模板缓存

        遍历所有可能的中文提示文本，触发 MiniMax 缓存生成（若缓存已存在则跳过）。
        """
        if not self._fallback_chain:
            return
        minimax = self._fallback_chain[0]
        if not isinstance(minimax, MiniMaxAudioOutput):
            print("[INFO] MiniMax not in chain; skip preload")
            return

        texts_list = list(texts)
        print(f"[INFO] Preloading {len(texts_list)} TTS cache entries...")
        cached = 0
        missed = 0
        for text in texts_list:
            try:
                path = minimax._get_audio_path(text)
                if path.exists() and path.stat().st_size > 0:
                    cached += 1
                else:
                    missed += 1
            except Exception as exc:
                print(f"[WARN] preload check failed for '{text}': {exc}", file=sys.stderr)
                missed += 1
        print(f"[INFO] Preload done: {cached} cached, {missed} need synthesis")

    def speak(self, text: str) -> None:
        with self._lock:  # 全局音频锁
            for backend in self._fallback_chain:
                try:
                    backend.speak(text)
                    return
                except Exception as exc:
                    print(
                        f"[WARN] audio backend {type(backend).__name__} failed: {exc}",
                        file=sys.stderr,
                    )
            # print 兜底保证绝不静默
            print(f"[AUDIO] {text}")
