import pathlib
import tempfile
from typing import List

from pydub import AudioSegment

import helpers
from config import Settings, settings
from classes.Transcriber import Transcriber
from classes.BaseImpl import TranscriptionResult


def transcribe_chunks_sync(
    chunks: List[AudioSegment], transcriber: Transcriber, cfg: Settings = settings
) -> str:
    """
    Synchronous orchestration using executors inside Transcriber implementations.
    Preserves chunk order: submit (idx, bytes) tasks, collect (idx, text) and join in order.
    """
    print("Transcribing chunks...")
    chunk_bytes = [helpers.audiosegment_to_bytes(chunk, cfg) for chunk in chunks]
    print(f"Submitting {len(chunk_bytes)} chunks for transcription...")
    results: list[TranscriptionResult] = transcriber.batch_transcribe(chunk_bytes)
    return ". ".join(res.text for res in results if res.text)


def run_sync(input_path: pathlib.Path, cfg: Settings = settings) -> str:
    helpers.ensure_ffmpeg()
    audio = helpers.load_audio(input_path)
    chunks = helpers.split_into_chunks(audio, cfg)

    if not chunks:
        chunks = [audio]

    dump_dir = None

    if cfg.debug_dump_chunks:
        dump_dir = pathlib.Path(tempfile.mkdtemp(prefix="chunks_"))

        for idx, seg in enumerate(chunks, start=1):
            seg.export(
                dump_dir / f"chunk_{idx}.{cfg.audio_format}", format=cfg.audio_format
            )
    print(f"Total chunks to transcribe: {len(chunks)}")
    transcriber = Transcriber(cfg)

    try:
        return transcribe_chunks_sync(chunks, transcriber, cfg)
    finally:
        if dump_dir:
            pass  # keep dump for debugging; cleanup logic can be added

        transcriber.shutdown()
