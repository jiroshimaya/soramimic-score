"""Optional, on-demand VOICEVOX singing preview from a Score document."""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import wave

from .document import ScoreDocument
from .japanese import kana_to_moras, mora_vowel


FRAME_RATE = 93.75
STYLE_ID = 6000  # VOICEVOX:波音リツ (sing)


def _post(base: str, endpoint: str, body: dict) -> bytes:
    url = f"{base.rstrip('/')}/{endpoint}?{urlencode({'speaker': STYLE_ID})}"
    request = Request(url, data=json.dumps(body, ensure_ascii=False).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=300) as response:
        return response.read()


def _score_for_slots(slots: list, offset: int) -> dict:
    cursor = 0
    notes = []
    previous_vowel = "ア"
    for slot in slots:
        start = max(cursor, round(slot.start_sec * FRAME_RATE) - offset)
        if cursor == 0 and start < 3:
            start = 3
        if 0 < start - cursor < 3:
            start = cursor
        end = max(start + 3, round(slot.end_sec * FRAME_RATE) - offset)
        if start > cursor:
            notes.append({"key": None, "frame_length": start - cursor, "lyric": ""})
        moras = list(kana_to_moras(slot.kana))
        lyrics = []
        for mora in moras:
            if mora == "ー":
                lyrics.append(previous_vowel)
            else:
                lyrics.append(mora)
                previous_vowel = {"a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ"}.get(
                    mora_vowel(mora), previous_vowel)
        lyrics = lyrics or [previous_vowel]
        # Very short notes cannot hold multiple VOICEVOX phonemes.
        count = min(len(lyrics), max(1, (end - start) // 4))
        for part, lyric in enumerate(lyrics[:count]):
            a = start + (end - start) * part // count
            b = start + (end - start) * (part + 1) // count
            notes.append({"key": slot.midi_pitch, "frame_length": b - a,
                          "lyric": lyric})
        cursor = end
    notes.append({"key": None, "frame_length": 24, "lyric": ""})
    return {"notes": notes}


def synthesize(document: ScoreDocument, output: Path, *, engine_url: str,
               duration_sec: float, on_progress=None,
               excluded_utterance_ids: frozenset[str] = frozenset()) -> None:
    """Synthesize short utterance chunks and place them on the original clock."""
    by_line = defaultdict(list)
    for slot in document.score.synthesis_plan:
        if slot.utterance_id in excluded_utterance_ids:
            continue
        by_line[slot.utterance_id].append(slot)
    groups = sorted((sorted(slots, key=lambda slot: slot.start_sec)
                     for slots in by_line.values()), key=lambda slots: slots[0].start_sec)
    if not groups:
        raise ValueError("歌唱音符がありません")
    sample_rate = channels = width = None
    pcm = None
    for index, slots in enumerate(groups):
        # Leave a short leading rest for the VOICEVOX singing API.
        offset = max(0, round(slots[0].start_sec * FRAME_RATE) - 28)
        query = json.loads(_post(engine_url, "sing_frame_audio_query",
                                 _score_for_slots(slots, offset)))
        chunk = _post(engine_url, "frame_synthesis", query)
        with wave.open(BytesIO(chunk), "rb") as wav:
            fmt = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
            if pcm is None:
                sample_rate, channels, width = fmt
                pcm = bytearray(round(duration_sec * sample_rate) * channels * width)
            elif fmt != (sample_rate, channels, width):
                raise ValueError("VOICEVOX音声形式が途中で変わりました")
            samples = wav.readframes(wav.getnframes())
        start_byte = round(offset / FRAME_RATE * sample_rate) * channels * width
        end_byte = min(len(pcm), start_byte + len(samples))
        if end_byte > start_byte:
            pcm[start_byte:end_byte] = samples[:end_byte - start_byte]
        if on_progress:
            on_progress(index + 1, len(groups))
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def mix_accompaniment(vocal: Path, accompaniment: Path) -> None:
    """Overlay the separated instrumental on the synthesized voice."""
    mixed = vocal.with_name(vocal.stem + "-mixed.wav")
    try:
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(vocal),
            "-i", str(accompaniment), "-filter_complex",
            "[1:a]volume=0.7[bgm];[0:a][bgm]amix=inputs=2:duration=first:normalize=0[out]",
            "-map", "[out]", "-c:a", "pcm_s16le", str(mixed),
        ], check=True, capture_output=True)
        mixed.replace(vocal)
    finally:
        mixed.unlink(missing_ok=True)
