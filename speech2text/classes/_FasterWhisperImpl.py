import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from config import Settings, settings
from classes.BaseImpl import BaseImpl, TranscriptionResult
from classes._impl_registry import register_impl


_thread_local = threading.local()


class _FasterWhisperImpl(BaseImpl):
    def __init__(self, cfg: Settings = settings):
        super().__init__(cfg)

        # conservative default: 1 worker per GPU unless user sets otherwise
        max_workers = max(1, cfg.max_workers)

        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, initializer=self._init_model
        )

    def _init_model(self):
        if getattr(_thread_local, "model", None) is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("Faster-Whisper not installed") from exc

            compute_type = "float16" if self.cfg.device.startswith("cuda") else "int8"

            _thread_local.model = WhisperModel(
                self.cfg.faster_whisper_model_size,
                device=self.cfg.device,
                compute_type=compute_type,
            )

    def _do_transcribe(
        self, bytes_io: bytes, idx: int, cfg: Settings
    ) -> TranscriptionResult:
        model = _thread_local.model

        try:
            import soundfile as sf
        except ImportError:
            raise RuntimeError("Soundfile not installed")

        with io.BytesIO(bytes_io) as bs_io:
            data, sr = sf.read(bs_io)

        segments, _ = model.transcribe(
            data, beam_size=5, language=cfg.language, vad_filter=False
        )
        text = " ".join(seg.text for seg in segments).strip()

        return TranscriptionResult(idx=idx, text=text)

    def batch_transcribe(
        self, audio_bytes_list: List[bytes]
    ) -> List[TranscriptionResult]:
        futures = []

        for idx, audio_bytes in enumerate(audio_bytes_list):
            futures.append(
                self.executor.submit(self._do_transcribe, audio_bytes, idx, self.cfg)
            )

        results: List[TranscriptionResult] = []

        for future in as_completed(futures):
            results.append(future.result())

        # return results in the original order
        results.sort(key=lambda res: res.idx)
        return results

    def shutdown(self) -> None:
        try:
            self.executor.shutdown(wait=True)
        except Exception:
            pass


# register this implementation for `cuda` device key
register_impl("cuda", _FasterWhisperImpl)
