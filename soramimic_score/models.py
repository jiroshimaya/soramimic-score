"""Optional local inference backends; importing this module loads no models."""
from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import contextmanager
import math
import logging
from pathlib import Path
import tempfile

from .audio import AudioAdapters, AlignedMora, LyricLine, MelodyNote
from .audio import _run_adapter
from .acoustic import KANA_MODEL, release_memory as _release, separate_vocals, transcribe_kana_views
from .japanese import kana_to_moras, katakana
from .readings import acoustic_windows, dictionary_readings, select_acoustic_reading

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    sheetsage_model: Path
    sheetsage_base: Path
    whisper_model: str = "large-v3"
    ctc_model: str = "reazon-research/japanese-wav2vec2-base-rs35kh"
    device: str = "cpu"
    local_files_only: bool = False
    separate_vocals: bool = True
    acoustic_readings: bool = True
    demucs_checkpoint: Path | None = None
    kana_model: str = KANA_MODEL

    def validate(self):
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        for directory in (self.sheetsage_model, self.sheetsage_base):
            for filename in ("config.json", "model.safetensors", "LICENSE"):
                if not (Path(directory) / filename).is_file():
                    raise ValueError(f"Model directory is missing {filename}: {directory}")


def read_melody_lab(path: Path) -> tuple[MelodyNote, ...]:
    notes = []
    for number, row in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not row.strip():
            continue
        fields = row.split()
        if len(fields) != 3:
            raise ValueError(f"Invalid melody LAB row {number}")
        start, end, pitch = float(fields[0]), float(fields[1]), int(fields[2])
        if not math.isfinite(start + end) or not 0 <= start < end or not 0 <= pitch <= 127:
            raise ValueError(f"Invalid melody LAB values on row {number}")
        notes.append(MelodyNote(start, end, pitch, "sheetsage2-vocal"))
    notes.sort(key=lambda n: (n.start_sec, n.end_sec, n.midi_pitch))
    normalized = []
    for note in notes:
        if normalized and normalized[-1].end_sec > note.start_sec:
            if normalized[-1].start_sec >= note.start_sec:
                raise ValueError("Conflicting simultaneous melody notes")
            normalized[-1] = replace(normalized[-1], end_sec=note.start_sec)
        normalized.append(note)
    return tuple(normalized)


@contextmanager
def prepared_adapters(path: Path, config: ModelConfig):
    """Keep a private temporary vocal stem alive for exactly one analysis."""
    config.validate()
    with tempfile.TemporaryDirectory(prefix="soramimic-score-vocals-") as directory:
        vocals = None
        if config.separate_vocals:
            vocals = Path(directory) / "vocals.wav"
            _run_adapter("vocal separation", separate_vocals, path, vocals, config)
        yield create_adapters(config, vocals_path=vocals)


def create_adapters(config: ModelConfig, *, vocals_path: Path | None = None) -> AudioAdapters:
    """Low-level model adapters; use prepared_adapters to manage separation."""
    config.validate()

    def recognize(path):
        logger.info("歌詞を認識しています")
        from faster_whisper import WhisperModel
        model = WhisperModel(config.whisper_model, device=config.device,
                             compute_type="int8" if config.device == "cpu" else "float16",
                             local_files_only=config.local_files_only)
        try:
            segments, info = model.transcribe(str(path), language="ja", vad_filter=False,
                                               condition_on_previous_text=False)
            lines = []
            previous_end = 0.0
            for segment in segments:
                start = max(previous_end, float(segment.start), 0.0)
                end = min(float(info.duration), float(segment.end))
                if segment.text.strip() and end > start:
                    lines.append(LyricLine(segment.text.strip(), start, end))
                    previous_end = end
            return tuple(lines)
        finally:
            del model
            _release()

    def recover_window(path, start, end):
        """Retry a melody-supported credit span without the surrounding song."""
        import librosa
        from faster_whisper import WhisperModel

        samples, _ = librosa.load(str(path), sr=16000, mono=True,
                                  offset=start, duration=end - start)
        if len(samples) == 0:
            return ()
        model = WhisperModel(config.whisper_model, device=config.device,
                             compute_type="int8" if config.device == "cpu" else "float16",
                             local_files_only=config.local_files_only)
        try:
            segments, _ = model.transcribe(samples, language="ja", vad_filter=False,
                                           condition_on_previous_text=False)
            lines = []
            for segment in segments:
                onset = max(start, start + float(segment.start))
                offset = min(end, start + float(segment.end))
                if segment.text.strip() and offset > onset:
                    lines.append(LyricLine(segment.text.strip(), onset, offset))
            return tuple(lines)
        finally:
            del model
            _release()

    def align(path, lines, readings):
        logger.info("発音時刻を推定しています")
        import librosa
        import numpy as np
        import torch
        import torchaudio.functional as taf
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        audio, rate = librosa.load(str(vocals_path or path), sr=16000, mono=True)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Audio must contain finite samples")
        processor = AutoProcessor.from_pretrained(config.ctc_model,
                                                  local_files_only=config.local_files_only)
        model = Wav2Vec2ForCTC.from_pretrained(config.ctc_model,
                                              local_files_only=config.local_files_only).eval().to(config.device)
        try:
            stride = math.prod(model.config.conv_stride)
            padded = np.pad(audio, (8000, 8000))
            logits = []
            # Keep 20-second cores with two seconds of acoustic context at joins.
            for pos in range(0, len(padded), 320000):
                first, last = max(0, pos - 32000), min(len(padded), pos + 352000)
                values = processor(padded[first:last], sampling_rate=rate,
                                   return_tensors="pt").input_values.to(config.device)
                with torch.inference_mode():
                    chunk = model(values).logits[0].cpu()
                lo = (pos - first) // stride
                # Even when right context reaches EOF, keep only this core.
                # Otherwise the final context is duplicated by the next core.
                hi = min(len(chunk), lo + 320000 // stride)
                logits.append(chunk[lo:hi])
            probs = torch.log_softmax(torch.cat(logits), dim=-1)
            vocab = processor.tokenizer.get_vocab()
            blank = model.config.pad_token_id
            # Both scripts denote the same acoustic token; combine their mass.
            aliases = {}
            for char, index in vocab.items():
                normalized = katakana(char)
                if len(normalized) == 1 and kana_to_moras(normalized):
                    aliases.setdefault(normalized, []).append(index)
            token_ids = {}
            for char, indices in aliases.items():
                target = vocab.get(char, indices[0])
                combined = torch.logsumexp(probs[:, indices], dim=1)
                probs[:, indices] = -torch.inf
                probs[:, target] = combined
                token_ids[char] = target
            moras = [kana_to_moras(reading.kana) for reading in readings]
            timed = all(line.start_sec is not None for line in lines)
            groups = ([([index], line.start_sec, line.end_sec)
                       for index, line in enumerate(lines)] if timed
                      else [(list(range(len(lines))), 0.0, len(audio) / rate)])
            output = []
            for indices, start, end in groups:
                lo = max(0, math.ceil((start + .5) * rate / stride - 1e-9))
                hi = min(len(probs), math.ceil((end + .5) * rate / stride - 1e-9))
                targets, owners = [], []
                for li in indices:
                    for mi, mora in enumerate(moras[li]):
                        for char in mora:
                            if char not in token_ids:
                                raise ValueError(f"CTC vocabulary does not support {char!r}")
                            targets.append(token_ids[char])
                            owners.append((li, mi))
                required = len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))
                if not targets or hi - lo < required:
                    raise ValueError("Lyrics do not fit the available acoustic alignment frames")
                alignment, scores = taf.forced_align(probs[lo:hi].unsqueeze(0).float(),
                                                     torch.tensor([targets]), blank=blank)
                spans = taf.merge_tokens(alignment[0], scores[0].exp(), blank=blank)
                grouped = {}
                for span, owner in zip(spans, owners, strict=True):
                    grouped.setdefault(owner, []).append(span)
                for (li, mi), spans in grouped.items():
                    onset = max(start, (spans[0].start + lo) * stride / rate - .5)
                    offset = min(end, (spans[-1].end + lo) * stride / rate - .5)
                    if offset <= onset:
                        raise ValueError("CTC produced an empty mora interval")
                    confidence = min(float(span.score) for span in spans)
                    output.append(AlignedMora(li, mi, moras[li][mi], onset, offset,
                                              confidence, "reazon-kana-ctc" +
                                              ("/demucs-htdemucs" if vocals_path else "")))
            return tuple(output)
        finally:
            del model
            _release()

    def melody(path):
        logger.info("音高を推定しています")
        import librosa
        import numpy as np
        import torch
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            str(Path(config.sheetsage_model).resolve()),
            base_model_path=str(Path(config.sheetsage_base).resolve()),
            trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16 if config.device == "cuda" else torch.float32,
        ).eval().to(config.device)
        try:
            samples, rate = librosa.load(str(path), sr=None, mono=True)
            if not len(samples) or not np.isfinite(samples).all():
                raise ValueError("Audio must contain finite samples")
            with tempfile.TemporaryDirectory(prefix="soramimic-score-") as directory:
                with torch.inference_mode():
                    model.transcribe(samples, sampling_rate=rate, output_dir=directory,
                                     melody_only=True)
                return read_melody_lab(Path(directory) / "melody_vocal.lab")
        finally:
            del model
            _release()

    def select_readings(path, lines):
        defaults = dictionary_readings(path, lines)
        candidates = tuple(reading.candidates for reading in defaults)
        ambiguous = [index for index, row in enumerate(candidates) if len(row) > 1]
        if not ambiguous:
            return defaults
        if all(line.start_sec is not None and line.end_sec is not None for line in lines):
            line_windows = [(line.start_sec, line.end_sec) for line in lines]
        else:
            # Locate supplied lyrics without recognizing or rewriting their text.
            # Re-align after pronunciation selection; these times are only context.
            coarse = align(path, lines, defaults)
            line_windows = [(min(m.start_sec for m in coarse if m.line_index == index),
                             max(m.end_sec for m in coarse if m.line_index == index))
                            for index in range(len(lines))]
        import librosa
        duration = librosa.get_duration(path=str(path))
        windows, assignments = [], {}
        for index in ambiguous:
            selected_windows = acoustic_windows(*line_windows[index], duration)
            assignments[index] = tuple(range(len(windows), len(windows) + len(selected_windows)))
            windows.extend(selected_windows)
        paths = {"mix": path}
        if vocals_path is not None:
            paths["vocals"] = vocals_path
        transcripts = transcribe_kana_views(paths, windows, config)
        result = list(defaults)
        for index in ambiguous:
            views = {view: "".join(rows[i] for i in assignments[index])
                     for view, rows in transcripts.items()}
            selection = select_acoustic_reading(candidates[index], views)
            result[index] = replace(selection, detail={
                **defaults[index].detail, **selection.detail,
                "windows_sec": [list(windows[i]) for i in assignments[index]],
                "model": config.kana_model,
                "vocal_separator": "demucs-htdemucs" if vocals_path else None,
            })
        return tuple(result)

    def lyric_reading(text):
        return dictionary_readings(None, (LyricLine(text),))[0].kana

    return AudioAdapters(select_readings if config.acoustic_readings else dictionary_readings,
                         align, melody, recognize, lyric_reading, recover_window)
