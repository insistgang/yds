from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MINIMAX_API_URL = "https://api.minimaxi.com/v1/t2a_v2"
DEFAULT_MINIMAX_MODEL = "speech-2.8-hd"
DEFAULT_MINIMAX_CACHE_DIR = Path("~/.cache/linkable_edge/tts_minimax").expanduser()
DEFAULT_MINIMAX_VOICE = "presenter_female"


class AudioOutput:
    def speak(self, text: str) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class PrintAudioOutput(AudioOutput):
    prefix: str = "[AUDIO]"

    def speak(self, text: str) -> None:
        print(f"{self.prefix} {text}")


class Pyttsx3AudioOutput(AudioOutput):
    def __init__(self) -> None:
        try:
            import pyttsx3  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyttsx3 is not installed") from exc
        self._engine = pyttsx3.init()

    def speak(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()


@dataclass(slots=True)
class MiniMaxTtsParams:
    model: str = DEFAULT_MINIMAX_MODEL
    speed: float = 1.0
    vol: float = 1.0
    pitch: int = 0
    emotion: str = ""
    sample_rate: int = 32000
    bitrate: int = 128000
    audio_format: str = "mp3"
    channel: int = 1
    language_boost: str = "Chinese"


def build_minimax_request_payload(text: str, voice_id: str, params: MiniMaxTtsParams) -> dict[str, Any]:
    voice_setting: dict[str, Any] = {
        "voice_id": voice_id,
        "speed": params.speed,
        "vol": params.vol,
        "pitch": params.pitch,
    }
    if params.emotion:
        voice_setting["emotion"] = params.emotion

    return {
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


def minimax_cache_key(text: str, voice_id: str, params: MiniMaxTtsParams) -> str:
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


def minimax_cache_path(cache_dir: Path, text: str, voice_id: str, params: MiniMaxTtsParams) -> Path:
    return cache_dir / f"{minimax_cache_key(text, voice_id, params)}.{params.audio_format}"


def call_minimax_tts(
    text: str,
    voice_id: str,
    params: MiniMaxTtsParams,
    api_key: str,
    group_id: str | None,
    api_url: str,
    timeout_sec: float,
) -> bytes:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is not installed; cannot call MiniMax TTS") from exc

    payload = build_minimax_request_payload(text, voice_id, params)
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
        raise RuntimeError(f"MiniMax returned non-JSON response, HTTP {response.status_code}") from exc

    base_resp = response_json.get("base_resp") or {}
    if response.status_code >= 400 or base_resp.get("status_code") not in (0, "0", None):
        raise RuntimeError(f"MiniMax request failed: {json.dumps(base_resp, ensure_ascii=False)}")

    audio_hex = (response_json.get("data") or {}).get("audio")
    if not audio_hex:
        raise RuntimeError(f"MiniMax response missing data.audio: {json.dumps(base_resp, ensure_ascii=False)}")

    try:
        return bytes.fromhex(audio_hex)
    except ValueError as exc:
        raise RuntimeError("MiniMax response data.audio is not valid hex") from exc


def resolve_player(requested_player: str) -> list[str] | None:
    if requested_player != "auto":
        executable = shutil.which(requested_player)
        if executable is None:
            raise RuntimeError(f"Audio player not found: {requested_player}")
        return [executable]

    mpg123 = shutil.which("mpg123")
    if mpg123 is not None:
        return [mpg123, "-q"]

    aplay = shutil.which("aplay")
    if aplay is not None:
        return [aplay]

    return None


def play_audio_file(file_path: Path, requested_player: str = "auto") -> None:
    command = resolve_player(requested_player)
    if command is None:
        print(f"[WARN] No mpg123/aplay found; synthesized audio kept at {file_path}", file=sys.stderr)
        return

    completed = subprocess.run([*command, str(file_path)], text=True)
    if completed.returncode != 0:
        print(f"[WARN] Audio playback failed rc={completed.returncode}: {file_path}", file=sys.stderr)


@dataclass(slots=True)
class MiniMaxAudioOutput(AudioOutput):
    voice_id: str = DEFAULT_MINIMAX_VOICE
    params: MiniMaxTtsParams = field(default_factory=MiniMaxTtsParams)
    cache_dir: Path = field(default_factory=lambda: DEFAULT_MINIMAX_CACHE_DIR)
    api_key: str | None = None
    group_id: str | None = None
    api_url: str = DEFAULT_MINIMAX_API_URL
    timeout_sec: float = 30.0
    player: str = "auto"

    def speak(self, text: str) -> None:
        audio_path = self._get_audio_path(text)
        play_audio_file(audio_path, self.player)

    def _get_audio_path(self, text: str) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = minimax_cache_path(self.cache_dir, text, self.voice_id, self.params)
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path

        api_key = self.api_key or os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise RuntimeError("MINIMAX_API_KEY is missing and no cached audio exists")

        group_id = self.group_id if self.group_id is not None else os.environ.get("MINIMAX_GROUP_ID")
        audio_bytes = call_minimax_tts(
            text=text,
            voice_id=self.voice_id,
            params=self.params,
            api_key=api_key,
            group_id=group_id,
            api_url=self.api_url,
            timeout_sec=self.timeout_sec,
        )
        output_path.write_bytes(audio_bytes)
        return output_path


def build_audio_output(
    name: str,
    *,
    voice: str = DEFAULT_MINIMAX_VOICE,
    minimax_model: str = DEFAULT_MINIMAX_MODEL,
    minimax_cache_dir: Path = DEFAULT_MINIMAX_CACHE_DIR,
    player: str = "auto",
) -> AudioOutput:
    if name == "print":
        return PrintAudioOutput()
    if name == "pyttsx3":
        return Pyttsx3AudioOutput()
    if name == "minimax":
        return MiniMaxAudioOutput(
            voice_id=voice,
            params=MiniMaxTtsParams(model=minimax_model),
            cache_dir=minimax_cache_dir,
            player=player,
        )
    raise ValueError(f"Unsupported audio backend: {name}")
