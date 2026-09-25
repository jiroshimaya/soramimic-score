import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from soramimic_score.resing import _ust, mix_accompaniment, synthesize
from tests.test_document import ScoreDocumentTests


class ResingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
    def test_separated_accompaniment_is_mixed_with_synthesized_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            voice = Path(directory) / "voice.wav"
            backing = Path(directory) / "backing.wav"
            for path, value in ((voice, 1000), (backing, 1000)):
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(struct.pack("<h", value) * 2400)
            mix_accompaniment(voice, backing)
            with wave.open(str(voice)) as wav:
                samples = struct.unpack("<2400h", wav.readframes(2400))
            self.assertEqual(len(samples), 2400)
            self.assertGreater(max(samples), 1400)
            self.assertLess(max(samples), 1500)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
    def test_loud_mix_is_limited_without_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            voice = Path(directory) / "voice.wav"
            backing = Path(directory) / "backing.wav"
            for path in (voice, backing):
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(struct.pack("<h", 30000) * 2400)
            mix_accompaniment(voice, backing)
            with wave.open(str(voice)) as wav:
                samples = struct.unpack("<2400h", wav.readframes(2400))
            self.assertLessEqual(max(samples), 31200)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
    def test_short_accompaniment_does_not_silently_leave_the_song_unbacked(self):
        with tempfile.TemporaryDirectory() as directory:
            voice = Path(directory) / "voice.wav"
            backing = Path(directory) / "backing.wav"
            for path, frames in ((voice, 48000), (backing, 2400)):
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(b"\x00\x00" * frames)
            with self.assertRaisesRegex(ValueError, "伴奏音声が"):
                mix_accompaniment(voice, backing)
            self.assertTrue(voice.is_file())

    def test_score_has_original_timeline_and_mora_lyrics(self):
        score = _ust(ScoreDocumentTests().score_document(), frozenset()).decode("cp932")
        self.assertIn("Lyric=カ", score)
        self.assertIn("Length=384", score)
        self.assertIn("NoteNum=", score)
        self.assertIn("[#TRACKEND]", score)

    def test_on_demand_audio_uses_configured_prettypitch(self):
        document = ScoreDocumentTests().score_document()
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            for relative in ("svs/render.py", "dict/ja.mora", "checkpoints/f0_3singer.pt",
                             "checkpoints/cons_dur_3singer.pt", "checkpoints/nhv_v3_2_1.onnx",
                             "../LeapSinger/infer.py", ".venv/bin/python"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ア a A\n" if relative == "dict/ja.mora" else "")
            output = Path(directory) / "resung.wav"
            def run(command, **kwargs):
                calls.append(command)
                if "svs.render" in command:
                    with wave.open(str(root.parent / "prettypitch/vocal.wav"), "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(24000)
                        wav.writeframes(b"\x01\x00" * 2400)
                else:
                    with wave.open(str(output), "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(24000)
                        wav.writeframes(b"\x01\x00" * 48000)
                return type("Result", (), {"returncode": 0, "stderr": ""})()
            with patch.dict("os.environ", {"PRETTYPITCH_ROOT": str(root)}), patch(
                "soramimic_score.resing.subprocess.run", side_effect=run
            ):
                synthesize(document, output, duration_sec=2)
            with wave.open(str(output)) as wav:
                self.assertEqual(wav.getnframes(), 48000)
                self.assertIn(b"\x01\x00", wav.readframes(wav.getnframes()))
        self.assertEqual(calls[0][1:3], ["-m", "svs.render"])
