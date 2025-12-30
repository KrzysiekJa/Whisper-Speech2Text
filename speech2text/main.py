"""
Podcast transcription using in-memory chunks and Whisper/backends.
- Splits audio by silence using pydub (silence_thresh default -45 dBFS)
- Preserves chunk order while transcribing chunks in parallel.
- Supports CPU 'whisper' and GPU 'faster-whisper'. Configure via settings.
- Defaults: silence_thresh = AudioSegment.dBFS - 16 dBFS, min_silence_len=500ms, keep_silence=100ms.

Usage:
    PYTHONPATH=. python speech2text/main.py <input-audio-file>
"""

import argparse
import logging
import pathlib
from typing import Tuple

from config import Settings, settings
from speech2text import run_sync


# TODO: def logger confFile !!!
logger = logging.getLogger(__name__)


def parse_cli() -> Tuple[argparse.Namespace, pathlib.Path]:
    parser = argparse.ArgumentParser(description="Transcribe podcast using Whisper")
    parser.add_argument("input", help="Input audio file (mp3/wav/etc)")
    parser.add_argument("--whisper-model-size", default=None)
    parser.add_argument("--faster-whisper-model-size", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--silence-offset", type=int, default=None)
    parser.add_argument("--min-silence-ms", type=int, default=None)
    parser.add_argument("--keep-silence-ms", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--debug-dump", type=bool, default=None)
    parser.add_argument("--audio-format", type=str, default=None)
    parser.add_argument("--language", type=str, default=None)

    args = parser.parse_args()
    return args, pathlib.Path(args.input)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parse_args, input_path = parse_cli()

    defaults = settings.model_dump()
    defaults.update(
        {key: value for key, value in vars(parse_args).items() if value is not None}
    )
    cfg = Settings(**defaults)
    print(cfg.model_dump())

    try:
        transcript = run_sync(input_path, cfg)
        print("\n--- TRANSCRIPT START ---\n")
        print(transcript)
        print("\n--- TRANSCRIPT END ---\n")
    except Exception as exc:
        logging.exception(f"Transcription failed: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
