"""Decode supported audio containers to a stable PCM input for the models."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import subprocess
import tempfile
import wave


AUDIO_SUFFIXES = frozenset({".wav", ".wave", ".mp3", ".m4a", ".mp4", ".flac",
                            ".ogg", ".opus", ".aac", ".aif", ".aiff", ".wma", ".webm"})


def _command(*args: str, timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True,
                              timeout=timeout)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("音声を読み込めませんでした。対応する音声ファイルを選んでください") from exc


def probe_audio(path: Path, *, max_duration: float | None = None) -> float:
    """Return duration after checking that an uploaded file has an audio stream."""
    result = _command("ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
                      "-show_entries", "format=duration:stream=codec_type",
                      "-of", "json", str(path), timeout=20)
    info = json.loads(result.stdout)
    if not any(stream.get("codec_type") == "audio" for stream in info.get("streams", [])):
        raise ValueError("音声トラックが見つかりません")
    try:
        duration = float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("音声の長さを確認できませんでした") from exc
    if not 0 < duration < float("inf"):
        raise ValueError("空の音声ファイルです")
    if max_duration is not None and duration > max_duration:
        raise ValueError("音源は15分以内にしてください")
    return duration


def decode_audio(path: Path, destination: Path, *, max_duration: float | None = None) -> None:
    """Write mono 16 kHz PCM WAV. The caller owns the destination directory."""
    limit = ("-t", str(max_duration + 1)) if max_duration is not None else ()
    _command("ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
             "-i", str(path), *limit, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", "-y", str(destination), timeout=3600)
    try:
        with wave.open(str(destination), "rb") as audio:
            if audio.getnframes() == 0:
                raise ValueError("空の音声ファイルです")
            if max_duration is not None and audio.getnframes() / audio.getframerate() > max_duration:
                raise ValueError("音源は15分以内にしてください")
    except (wave.Error, EOFError) as exc:
        raise ValueError("音声を読み込めませんでした") from exc


def _is_pcm_wav(path: Path) -> bool:
    if path.suffix.lower() not in {".wav", ".wave"}:
        return False
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() > 0 and audio.getframerate() >= 8000
    except (wave.Error, EOFError):
        return False


@contextmanager
def decoded_audio(path: Path):
    """Keep compressed-media decoding private for one analysis call."""
    if _is_pcm_wav(path):
        yield path
        return
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError("対応する音声ファイルを選んでください")
    with tempfile.TemporaryDirectory(prefix="soramimic-score-audio-") as directory:
        destination = Path(directory) / "input.wav"
        decode_audio(path, destination)
        yield destination
