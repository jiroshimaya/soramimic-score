"""Client for the versioned, loopback-only Soramimic audio inference worker."""

from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

import requests


class SharedInference:
    def __init__(self, url: str, priority: str = "dev") -> None:
        if priority not in {"public", "preview", "dev", "eval"}:
            raise ValueError("invalid shared inference priority")
        self.url = url.rstrip("/")
        self.priority = priority
        response = requests.get(f"{self.url}/healthz", timeout=(5, 5))
        response.raise_for_status()
        health = response.json()
        api = health.get("api")
        if (health.get("status") != "ok"
                or api != {"name": "soramimic-audio-inference", "version": 1}
                or not all(health.get("capabilities", {}).get(key)
                           for key in ("demucs", "whisper", "kana_whisper", "sheetsage2"))):
            raise RuntimeError("shared audio inference service is incompatible")

    def run(self, kind: str, audio: Path, parameters: dict,
            artifacts: dict[str, Path] | None = None):
        job_id = None
        try:
            with audio.open("rb") as stream:
                response = requests.post(
                    f"{self.url}/v1/jobs",
                    files={"audio": (audio.name, stream, "application/octet-stream")},
                    data={"kind": kind, "priority": self.priority,
                          "parameters": json.dumps(parameters, ensure_ascii=False)},
                    timeout=(5, 300),
                )
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                try:
                    detail = response.json().get("detail")
                except (ValueError, AttributeError):
                    detail = None
                if detail:
                    raise RuntimeError(f"shared {kind} request rejected: {detail}") from exc
                raise
            job_id = response.json()["id"]
            while True:
                response = requests.get(f"{self.url}/v1/jobs/{job_id}", timeout=(5, 30))
                response.raise_for_status()
                state = response.json()
                if state.get("status") == "done":
                    if artifacts:
                        self._download(job_id, artifacts)
                    return state.get("result")
                if state.get("status") in {"error", "cancelled"}:
                    raise RuntimeError(f"shared {kind} inference failed: {state.get('error')}")
                time.sleep(.5)
        finally:
            if job_id is not None:
                try:
                    requests.delete(f"{self.url}/v1/jobs/{job_id}", timeout=(5, 10))
                except requests.RequestException:
                    pass

    def _download(self, job_id: str, artifacts: dict[str, Path]) -> None:
        temporary = {}
        try:
            for name, target in artifacts.items():
                temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
                temporary[target] = temp
                with requests.get(f"{self.url}/v1/jobs/{job_id}/artifacts/{name}",
                                  stream=True, timeout=(5, 300)) as response:
                    response.raise_for_status()
                    with temp.open("xb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output.write(chunk)
            for target, temp in temporary.items():
                temp.replace(target)
        finally:
            for temp in temporary.values():
                temp.unlink(missing_ok=True)
