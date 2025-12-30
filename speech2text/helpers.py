import io
import shutil
import pathlib
from typing import List

from pydub import AudioSegment
from pydub.silence import split_on_silence

from config import Settings, settings


# === ffmpeg helper with OS-aware suggestions and optional prompting installer ===
def ensure_ffmpeg() -> None:
    """Raise if ffmpeg is not on PATH."""
    if not (shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")):
        raise RuntimeError("ffmpeg not found – install it and add to PATH.")


# === Audio processing utilities (in-memory) ===
def load_audio(path: pathlib.Path) -> AudioSegment:
    """Read any audio file and return an AudioSegment."""
    try:
        return AudioSegment.from_file(str(path))
    except Exception as exc:
        raise RuntimeError(f"Unable to load {path}: {exc}") from exc


def compute_silence_thresh(audio: AudioSegment, cfg: Settings = settings) -> float:
    """Silence threshold = audio dBFS + offset."""
    return audio.dBFS + cfg.silence_offset_dbfs


def split_into_chunks(
    audio: AudioSegment, cfg: Settings = settings
) -> List[AudioSegment]:
    """Return a list of AudioSegments split on silence."""
    thresh = compute_silence_thresh(audio, cfg)

    try:
        return split_on_silence(
            audio,
            min_silence_len=cfg.min_silence_len_ms,
            silence_thresh=thresh,
            keep_silence=cfg.keep_silence_ms,
        )
    except Exception as exc:
        raise RuntimeError("Silence-based splitting failed") from exc


def audiosegment_to_bytes(seg: AudioSegment, cfg: Settings = settings) -> bytes:
    """Convert to bytes once – cheap compared to model inference"""
    buffer = io.BytesIO()
    seg.export(buffer, format=cfg.audio_format)
    return buffer.getvalue()
