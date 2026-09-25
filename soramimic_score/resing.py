"""On-demand PrettyPitch singing preview from a Score document."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from .document import ScoreDocument
from .japanese import kana_to_moras, mora_vowel


TICKS_PER_SECOND = 960  # 120 BPM, 480 ticks per beat
MORA_ADDITIONS = ("リュ ry U", "リョ ry O", "ヴィ v I")


def _runtime() -> tuple[Path, Path, Path]:
    root_value = os.environ.get("PRETTYPITCH_ROOT")
    if not root_value:
        raise RuntimeError("PRETTYPITCH_ROOT が設定されていません")
    root = Path(root_value).expanduser().resolve()
    python = Path(os.environ.get("PRETTYPITCH_PYTHON", str(root / ".venv/bin/python"))).expanduser().absolute()
    leapsinger = Path(os.environ.get("PRETTYPITCH_LEAPSINGER_ROOT", str(root.parent / "LeapSinger"))).expanduser().resolve()
    for required in (root / "svs/render.py", root / "dict/ja.mora",
                     root / "checkpoints/f0_3singer.pt", root / "checkpoints/cons_dur_3singer.pt",
                     root / "checkpoints/nhv_v3_2_1.onnx", leapsinger / "infer.py", python):
        if not required.is_file():
            raise RuntimeError(f"PrettyPitch の必要ファイルがありません: {required}")
    return root, python, leapsinger


def available() -> bool:
    """Report whether the external renderer can be offered in this service."""
    try:
        root, _, _ = _runtime()
    except RuntimeError:
        return False
    acoustic = root / "checkpoints/leapsinger"
    return acoustic.is_dir() and next(acoustic.rglob("3speaker_gan2d.pth"), None) is not None


def _ust(document: ScoreDocument, excluded: frozenset[str]) -> bytes:
    chunks = ["[#VERSION]\nUST Version1.2\n", "[#SETTING]\nTempo=120\n"]
    cursor = index = 0
    previous_vowel = "ア"
    for slot in sorted(document.score.synthesis_plan, key=lambda item: (item.start_sec, item.id)):
        if slot.utterance_id in excluded:
            continue
        start = max(cursor, round(slot.start_sec * TICKS_PER_SECOND))
        end = max(start + 1, round(slot.end_sec * TICKS_PER_SECOND))
        if start > cursor:
            chunks.append(f"[#{index:04d}]\nLength={start - cursor}\nLyric=R\nNoteNum=60\n")
            index += 1
        moras = kana_to_moras(slot.kana)
        lyric = "".join(moras) or previous_vowel
        if lyric == "ー":
            lyric = previous_vowel
        else:
            previous_vowel = {"a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ"}.get(
                mora_vowel(moras[-1]) if moras else None, previous_vowel)
        chunks.append(f"[#{index:04d}]\nLength={end - start}\nLyric={lyric}\nNoteNum={slot.midi_pitch}\n")
        cursor = end
        index += 1
    if not index:
        raise ValueError("歌唱音符がありません")
    chunks.append("[#TRACKEND]\n")
    return "".join(chunks).encode("cp932")


def synthesize(document: ScoreDocument, output: Path, *, duration_sec: float,
               on_progress=None, excluded_utterance_ids: frozenset[str] = frozenset()) -> None:
    """Render estimated notes with PrettyPitch and match the original duration."""
    root, python, leapsinger = _runtime()
    work = output.parent / "prettypitch"
    work.mkdir(mode=0o700, exist_ok=True)
    ust = work / "score.ust"
    raw = work / "vocal.wav"
    mora_table = work / "ja.mora"
    ust.write_bytes(_ust(document, excluded_utterance_ids))
    base = (root / "dict/ja.mora").read_text(encoding="utf-8")
    present = {line.split(maxsplit=1)[0] for line in base.splitlines()
               if line.strip() and not line.lstrip().startswith("#")}
    additions = [line for line in MORA_ADDITIONS if line.split()[0] not in present]
    mora_table.write_text(base.rstrip() + "\n" + "\n".join(additions) + "\n", encoding="utf-8")
    if on_progress:
        on_progress(0, 1)
    process = subprocess.run([
        str(python), "-m", "svs.render", str(ust), "-o", str(raw),
        "--spk_id", "2", "--device", os.environ.get("PRETTYPITCH_DEVICE", "cuda"),
        "--seed", "0", "--mora_table", str(mora_table),
        "--leapsinger_root", str(leapsinger),
    ], cwd=root, capture_output=True, text=True, timeout=900)
    if process.returncode or not raw.is_file() or not raw.stat().st_size:
        raise RuntimeError("PrettyPitch の歌唱合成に失敗しました: " + process.stderr[-1000:])
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(raw),
        "-af", f"apad=whole_dur={duration_sec:.3f},atrim=duration={duration_sec:.3f}",
        "-c:a", "pcm_s16le", str(output),
    ], check=True, capture_output=True, timeout=60)
    if on_progress:
        on_progress(1, 1)


def mix_accompaniment(vocal: Path, accompaniment: Path) -> None:
    """Overlay the separated instrumental on the synthesized voice."""
    mixed = vocal.with_name(vocal.stem + "-mixed.wav")
    try:
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(vocal),
            "-i", str(accompaniment), "-filter_complex",
            "[1:a]volume=0.45[bgm];[0:a][bgm]amix=inputs=2:duration=first:normalize=0,"
            "alimiter=limit=0.95:attack=5:release=50:level=0:latency=1[out]",
            "-map", "[out]", "-c:a", "pcm_s16le", str(mixed),
        ], check=True, capture_output=True)
        mixed.replace(vocal)
    finally:
        mixed.unlink(missing_ok=True)
