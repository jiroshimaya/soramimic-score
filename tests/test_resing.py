import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from soramimic_score.resing import _score_for_slots, synthesize
from tests.test_document import ScoreDocumentTests


class ResingTests(unittest.TestCase):
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
