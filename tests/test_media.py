import math
from contextlib import contextmanager
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from soramimic_score.audio import (AlignedMora, AudioAdapters, LyricLine,
                                   MelodyNote, ReadingSelection, analyze_audio)
from soramimic_score.media import decode_audio, decoded_audio, probe_audio


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class AudioMediaTests(unittest.TestCase):
    def test_mp3_decodes_to_pcm_for_library_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "voice.wav"
            mp3 = root / "voice.mp3"
            with wave.open(str(original), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"".join(struct.pack("<h", int(7000 * math.sin(i / 8)))
                                            for i in range(16000)))
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(original),
                            "-y", str(mp3)], check=True)
            self.assertGreater(probe_audio(mp3), .9)
            with decoded_audio(mp3) as prepared:
                self.assertEqual(prepared.suffix, ".wav")
                with wave.open(str(prepared)) as audio:
                    self.assertEqual(audio.getframerate(), 16000)
                    self.assertGreater(audio.getnframes(), 15000)
            with self.assertRaisesRegex(ValueError, "15分"):
                probe_audio(mp3, max_duration=.5)
            decoded = root / "decoded.wav"
            decode_audio(mp3, decoded)
            self.assertTrue(decoded.is_file())

            @contextmanager
            def prepared(path, _config):
                with wave.open(str(path)) as audio:
                    self.assertEqual(audio.getframerate(), 16000)
                yield AudioAdapters(
                    lambda _path, _lines: (ReadingSelection("ソラ", "test", .9),),
                    lambda _path, _lines, _readings: (
                        AlignedMora(0, 0, "ソ", .1, .3, .8),
                        AlignedMora(0, 1, "ラ", .3, .5, .8)),
                    lambda _path: (MelodyNote(.1, .5, 60),),
                    lambda _path: (LyricLine("空", .1, .5),),
                )

            with patch("soramimic_score.models.prepared_adapters", prepared):
                score = analyze_audio(mp3, model_config=SimpleNamespace(separate_vocals=False))
            self.assertEqual(score.score.canonical_text, "空")
