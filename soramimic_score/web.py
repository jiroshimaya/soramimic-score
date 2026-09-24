"""Private-by-default job API for Soramimic Score."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import logging
import os
import secrets
import shutil
import sqlite3
import time
import wave

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from .audio import AudioPipelineError
from .document import load
from .exports import EXPORTS
from .models import ModelConfig
from .ir import has_usable_timing


MAX_WAV_BYTES = 100 * 1024 * 1024
MAX_DURATION_SEC = 15 * 60
QUOTA_PER_DAY = 100
RETENTION_SEC = 24 * 3600
logger = logging.getLogger(__name__)


def _now_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _validate_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getnchannels() not in (1, 2) or audio.getframerate() < 8000:
                raise ValueError("対応形式は8 kHz以上のモノラル・ステレオWAVです")
            if audio.getnframes() / audio.getframerate() > MAX_DURATION_SEC:
                raise ValueError("音源は15分以内にしてください")
            if audio.getnframes() == 0:
                raise ValueError("空の音源です")
    except (wave.Error, EOFError) as exc:
        raise ValueError("PCM WAVを指定してください") from exc


def _job_id(value: str) -> str:
    if len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
        raise HTTPException(404)
    return value


def _request_ip(request: Request) -> str:
    peer = request.client.host if request.client else ""
    # Cloudflare Tunnel terminates at loopback. Never trust headers from other peers.
    if peer in ("127.0.0.1", "::1"):
        forwarded = request.headers.get("cf-connecting-ip", "")
        if forwarded:
            from ipaddress import ip_address
            try:
                return str(ip_address(forwarded))
            except ValueError:
                pass
    return peer


def _prune(root: Path, db: Path) -> None:
    cutoff = datetime.fromtimestamp(time.time() - RETENTION_SEC, timezone.utc).isoformat()
    with sqlite3.connect(db) as conn:
        old = conn.execute("SELECT id FROM jobs WHERE created<? AND state IN ('done','failed')",
                           (cutoff,)).fetchall()
        conn.executemany("DELETE FROM jobs WHERE id=?", old)
        conn.execute("DELETE FROM quota WHERE day<?", (_now_day(),))
    for (job,) in old:
        shutil.rmtree(root / job, ignore_errors=True)


def create_app(*, data_root: Path | None = None, analyzer=None, public: bool | None = None) -> FastAPI:
    """Create a single-worker app. Keep its data directory outside the Git checkout."""
    root = data_root or Path(os.environ.get("SORAMIMIC_SCORE_DATA", "work/web"))
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    db = root / "jobs.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, state TEXT NOT NULL, "
                     "error TEXT, created TEXT NOT NULL, ip TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS quota (day TEXT NOT NULL, ip TEXT NOT NULL, "
                     "used INTEGER NOT NULL, PRIMARY KEY(day, ip))")
        conn.execute("UPDATE jobs SET state='failed', error='処理が中断されました' "
                     "WHERE state IN ('queued', 'running')")
    db.chmod(0o600)
    is_public = public if public is not None else os.environ.get("SORAMIMIC_SCORE_PUBLIC") == "1"
    pool = ThreadPoolExecutor(max_workers=1)
    app = FastAPI(title="Soramimic Score", docs_url=None, redoc_url=None, openapi_url=None)

    def analyze(job: str, supplied: bool) -> None:
        from .audio import analyze_audio
        from .document import dump
        job_dir = root / job
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE jobs SET state='running' WHERE id=?", (job,))
        try:
            config = ModelConfig(
                sheetsage_model=Path(os.environ["SORAMIMIC_SCORE_SHEETSAGE_MODEL"]),
                sheetsage_base=Path(os.environ["SORAMIMIC_SCORE_SHEETSAGE_BASE"]),
                device=os.environ.get("SORAMIMIC_SCORE_DEVICE", "cpu"),
                local_files_only=os.environ.get("SORAMIMIC_SCORE_LOCAL_ONLY") == "1",
            )
            lyrics = tuple(line for line in (job_dir / "lyrics.txt").read_text(
                encoding="utf-8").splitlines() if line.strip()) if supplied else None
            result = (analyzer or analyze_audio)(job_dir / "input.wav", model_config=config,
                                                 lyrics=lyrics)
            dump(result, job_dir / "score.json")
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE jobs SET state='done' WHERE id=?", (job,))
        except Exception as exc:
            logger.exception("score analysis failed for job %s", job)
            message = "解析に失敗しました。音源と設定を確認してください"
            if isinstance(exc, AudioPipelineError):
                if exc.stage == "melody" and "no melody notes were produced" in str(exc):
                    message = "歌唱の音符を検出できませんでした。別の歌唱音源をお試しください"
                elif exc.stage == "melody" and "At least two decoded beats" in str(exc):
                    message = "音符の推定に必要な長さが足りません。長めの歌唱音源をお試しください"
                elif exc.stage == "lyrics" and "no lyric lines were produced" in str(exc):
                    message = "歌詞を認識できませんでした。歌声が聞こえる音源をお試しください"
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE jobs SET state='failed', error=? WHERE id=?",
                             (message, job))

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def home():
        return FileResponse(Path(__file__).with_name("score.html"), media_type="text/html")

    @app.post("/api/jobs")
    async def submit(request: Request, audio: UploadFile = File(...), lyrics: str = Form("")):
        _prune(root, db)
        if len(lyrics.encode("utf-8")) > 200_000:
            raise HTTPException(413, "歌詞が大きすぎます")
        if not audio.filename or not audio.filename.lower().endswith(".wav"):
            raise HTTPException(400, "WAVファイルを選んでください")
        job = secrets.token_hex(16)
        job_dir = root / job
        job_dir.mkdir(mode=0o700)
        path = job_dir / "input.wav"
        size = 0
        try:
            with path.open("wb") as output:
                while chunk := await audio.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_WAV_BYTES:
                        raise HTTPException(413, "WAVは100 MB以下にしてください")
                    output.write(chunk)
            try:
                _validate_wav(path)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            if lyrics.strip():
                (job_dir / "lyrics.txt").write_text(lyrics, encoding="utf-8")
            ip = _request_ip(request)
            with sqlite3.connect(db, timeout=30, isolation_level=None) as conn:
                conn.execute("BEGIN IMMEDIATE")
                if is_public:
                    count = conn.execute("SELECT used FROM quota WHERE day=? AND ip=?",
                                         (_now_day(), ip)).fetchone()
                    if count and count[0] >= QUOTA_PER_DAY:
                        raise HTTPException(429, "本日の解析回数は100回に達しました")
                    conn.execute("INSERT INTO quota(day, ip, used) VALUES(?, ?, 1) "
                                 "ON CONFLICT(day, ip) DO UPDATE SET used=used+1",
                                 (_now_day(), ip))
                conn.execute("INSERT INTO jobs(id,state,created,ip) VALUES(?,?,?,?)",
                             (job, "queued", datetime.now(timezone.utc).isoformat(), ip))
                conn.commit()
            pool.submit(analyze, job, bool(lyrics.strip()))
        except Exception:
            if not path.exists() or not _known_job(db, job):
                for item in job_dir.iterdir():
                    item.unlink()
                job_dir.rmdir()
            raise
        return {"id": job, "state": "queued"}

    @app.get("/api/jobs/{job}")
    def status(job: str):
        job = _job_id(job)
        with sqlite3.connect(db) as conn:
            row = conn.execute("SELECT state,error FROM jobs WHERE id=?", (job,)).fetchone()
        if not row:
            raise HTTPException(404)
        return {"id": job, "state": row[0], "error": row[1]}

    @app.get("/api/jobs/{job}/score")
    def score(job: str):
        document = _completed(root, db, _job_id(job))
        slots = document.score.synthesis_plan
        units = {x.singing_unit_id: x for x in document.score.performed}
        observed = {x.id: x for x in document.observations.singing_units}
        moras = {x.id: x.text for x in document.observations.moras}
        by_unit = {}
        for slot in slots:
            interval = by_unit.setdefault(slot.singing_unit_id,
                                          [slot.start_sec, slot.end_sec, slot.utterance_id])
            interval[0] = min(interval[0], slot.start_sec)
            interval[1] = max(interval[1], slot.end_sec)
        timeline = []
        for unit_id, (start, end, line) in by_unit.items():
            ids = units[unit_id].mora_ids
            observation = observed.get(unit_id)
            aligned = (observation is not None
                       and has_usable_timing(observation.consonant_start)
                       and has_usable_timing(observation.end)
                       and observation.consonant_start.time_sec < observation.end.time_sec)
            if aligned:
                start = observation.consonant_start.time_sec
                end = observation.end.time_sec
            duration = (end - start) / len(ids)
            for index, mora_id in enumerate(ids):
                timeline.append({"id": mora_id, "line": line,
                                 "text": moras[mora_id],
                                 "start": start + duration * index,
                                 "end": start + duration * (index + 1),
                                 "source": "aligned" if aligned else "estimated"})
        timeline.sort(key=lambda x: (x["start"], x["end"]))
        return {"lines": [{"id": line.utterance_id, "text": line.text,
                           "kana": line.kana} for line in document.score.canonical],
                "notes": [{"start": s.start_sec, "end": s.end_sec, "pitch": s.midi_pitch,
                           "line": s.utterance_id, "kana": s.kana} for s in slots],
                "moras": timeline}

    @app.get("/api/jobs/{job}/audio")
    def audio(job: str):
        _completed(root, db, _job_id(job))
        return FileResponse(root / job / "input.wav", media_type="audio/wav")

    @app.get("/api/jobs/{job}/download/{format}")
    def download(job: str, format: str):
        document = _completed(root, db, _job_id(job))
        if format not in EXPORTS:
            raise HTTPException(404)
        media_type, exporter = EXPORTS[format]
        return Response(exporter(document), media_type=media_type,
                        headers={"Content-Disposition": f'attachment; filename="score.{format}"'})

    return app


def _known_job(db: Path, job: str) -> bool:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT 1 FROM jobs WHERE id=?", (job,)).fetchone() is not None


def _completed(root: Path, db: Path, job: str):
    with sqlite3.connect(db) as conn:
        state = conn.execute("SELECT state FROM jobs WHERE id=?", (job,)).fetchone()
    if not state:
        raise HTTPException(404)
    if state[0] != "done":
        raise HTTPException(409, "解析が完了していません")
    return load(root / job / "score.json")


app = create_app()
