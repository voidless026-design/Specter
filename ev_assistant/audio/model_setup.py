"""Downloads and caches the small offline Vosk speech model E.V. uses for
both wake-word spotting and command transcription. Runs once - after that,
`is_model_present` short-circuits.
"""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODEL_DIR_NAME = "vosk-model-small-en-us-0.15"

ProgressFn = Callable[[int, int], None]


def is_model_present(model_dir: Path) -> bool:
    return (model_dir / "conf" / "model.conf").exists() or (model_dir / "am" / "final.mdl").exists()


def download_model(model_dir: Path, progress: ProgressFn | None = None) -> None:
    """Download and extract the Vosk small English model into `model_dir`."""
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = model_dir.parent / "vosk-model.zip.part"

    with httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with zip_path.open("wb") as f:
            for chunk in response.iter_bytes(chunk_size=1 << 16):
                f.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(downloaded, total)

    extract_dir = model_dir.parent / "_vosk_extract_tmp"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    finally:
        zip_path.unlink(missing_ok=True)

    extracted_root = extract_dir / MODEL_DIR_NAME
    if not extracted_root.exists():
        candidates = [p for p in extract_dir.iterdir() if p.is_dir()]
        if len(candidates) == 1:
            extracted_root = candidates[0]
        else:
            shutil.rmtree(extract_dir, ignore_errors=True)
            raise RuntimeError(f"Unexpected Vosk model archive layout under {extract_dir}")

    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.move(str(extracted_root), str(model_dir))
    shutil.rmtree(extract_dir, ignore_errors=True)


def ensure_model(model_dir: Path, progress: ProgressFn | None = None) -> None:
    if not is_model_present(model_dir):
        download_model(model_dir, progress=progress)
