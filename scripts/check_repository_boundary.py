#!/usr/bin/env python3
"""Fail when Git tracks private research data or oversized artifacts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


FORBIDDEN_PARTS = {"local-data", "work", "artifacts"}
FORBIDDEN_NAMES = {".env", ".netrc", ".npmrc", ".pypirc"}
FORBIDDEN_SUFFIXES = {
    ".aac",
    ".aif",
    ".aiff",
    ".bin",
    ".bz2",
    ".ckpt",
    ".flac",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".joblib",
    ".kar",
    ".key",
    ".m4a",
    ".mid",
    ".midi",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".npy",
    ".npz",
    ".ogg",
    ".onnx",
    ".opus",
    ".pb",
    ".pdf",
    ".pem",
    ".pkl",
    ".pickle",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tgz",
    ".th",
    ".tflite",
    ".token",
    ".wav",
    ".wave",
    ".webm",
    ".webp",
    ".wma",
    ".xz",
    ".zip",
}
MAX_TRACKED_BYTES = 10 * 1024 * 1024


def tracked_paths() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"])
    return [Path(item.decode()) for item in raw.split(b"\0") if item]


def main() -> int:
    failures: list[str] = []
    paths = tracked_paths()

    for path in paths:
        if FORBIDDEN_PARTS.intersection(path.parts):
            failures.append(f"private path is tracked: {path}")
        if path.name in FORBIDDEN_NAMES or (
            path.name.startswith(".env.") and path.name != ".env.example"
        ):
            failures.append(f"private configuration is tracked: {path}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden artifact type is tracked: {path}")
        if path.is_symlink():
            target = os.readlink(path)
            if os.path.isabs(target):
                failures.append(f"absolute symlink is tracked: {path} -> {target}")
            continue
        if path.exists() and path.stat().st_size > MAX_TRACKED_BYTES:
            failures.append(f"tracked file exceeds 10 MiB: {path}")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1

    print(f"repository boundary OK: {len(paths)} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
