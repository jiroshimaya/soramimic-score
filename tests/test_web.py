import io
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

from fastapi.testclient import TestClient

from soramimic_score.web import create_app
from tests.test_document import ScoreDocumentTests


def wav_bytes():
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 1600)
    return out.getvalue()


class ScoreWebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.document = ScoreDocumentTests().score_document()
        self.received = []

        def analyze(path, *, model_config, lyrics):
            self.received.append((Path(path).name, lyrics))
            return self.document

        self.client = TestClient(create_app(data_root=Path(self.temporary.name),
                                            analyzer=analyze, public=True))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temporary.cleanup()

    def submit(self, lyrics=""):
        return self.client.post("/api/jobs", files={"audio": ("song.wav", wav_bytes(), "audio/wav")},
                                data={"lyrics": lyrics})

    def test_upload_analysis_preview_and_exports(self):
        with patch.dict("os.environ", {"SORAMIMIC_SCORE_SHEETSAGE_MODEL": "a",
                                    "SORAMIMIC_SCORE_SHEETSAGE_BASE": "b"}):
            response = self.submit("カ")
            self.assertEqual(response.status_code, 200)
            job = response.json()["id"]
            for _ in range(100):
                state = self.client.get(f"/api/jobs/{job}").json()["state"]
                if state == "done":
                    break
                time.sleep(.02)
            self.assertEqual(state, "done")
        self.assertEqual(self.received, [("input.wav", ("カ",))])
        score = self.client.get(f"/api/jobs/{job}/score").json()
        self.assertEqual(score["lines"][0]["text"], "カ")
        self.assertTrue(score["notes"])
        self.assertEqual(len({m["id"] for m in score["moras"]}), len(score["moras"]))
        self.assertEqual(self.client.get(f"/api/jobs/{job}/audio").status_code, 200)
        for format in ("json", "mid", "musicxml", "srt", "lrc"):
            response = self.client.get(f"/api/jobs/{job}/download/{format}")
            self.assertEqual(response.status_code, 200, format)
            self.assertTrue(response.content)
        ElementTree.fromstring(self.client.get(f"/api/jobs/{job}/download/musicxml").content)
        self.assertTrue(self.client.get(f"/api/jobs/{job}/download/mid").content.startswith(b"MThd"))

    def test_invalid_upload_does_not_consume_quota(self):
        with patch("soramimic_score.web.QUOTA_PER_DAY", 1):
            bad = self.client.post("/api/jobs", files={"audio": ("song.wav", b"broken")})
            self.assertEqual(bad.status_code, 400)
            with patch.dict("os.environ", {"SORAMIMIC_SCORE_SHEETSAGE_MODEL": "a",
                                        "SORAMIMIC_SCORE_SHEETSAGE_BASE": "b"}):
                self.assertEqual(self.submit().status_code, 200)
                self.assertEqual(self.submit().status_code, 429)

    def test_private_job_requires_unguessable_id(self):
        self.assertEqual(self.client.get("/api/jobs/missing").status_code, 404)
        self.assertEqual(self.client.get("/api/jobs/" + "0" * 32 + "/audio").status_code, 404)


if __name__ == "__main__":
    unittest.main()
