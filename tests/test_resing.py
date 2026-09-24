import io
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from soramimic_score.resing import _score_for_slots, mix_accompaniment, synthesize
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

    def test_score_has_lead_rest_and_mora_lyrics(self):
        slots = ScoreDocumentTests().score_document().score.synthesis_plan
        score = _score_for_slots(list(slots), 0)
        self.assertIsNone(score["notes"][0]["key"])
        self.assertTrue(all(note["frame_length"] >= 3 for note in score["notes"]))

    def test_on_demand_audio_uses_original_timeline(self):
        document = ScoreDocumentTests().score_document()
        calls = []

        def post(_base, endpoint, body):
            calls.append(endpoint)
            if endpoint == "sing_frame_audio_query":
                return b"{}"
            out = io.BytesIO()
            with wave.open(out, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b"\x01\x00" * 2400)
            return out.getvalue()

        with tempfile.TemporaryDirectory() as directory, patch(
            "soramimic_score.resing._post", side_effect=post
        ):
            output = Path(directory) / "resung.wav"
            synthesize(document, output, engine_url="http://127.0.0.1:50021", duration_sec=2)
            with wave.open(str(output)) as wav:
                self.assertEqual(wav.getnframes(), 48000)
                self.assertIn(b"\x01\x00", wav.readframes(wav.getnframes()))
        self.assertEqual(calls, ["sing_frame_audio_query", "frame_synthesis"])
