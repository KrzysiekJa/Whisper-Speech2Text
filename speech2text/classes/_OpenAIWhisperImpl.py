import io
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from config import Settings, settings
from classes.BaseImpl import BaseImpl, TranscriptionResult
from classes._impl_registry import register_impl


# module-level variable to hold per-process model
_process_model = None


def _proc_transcribe(bytes_io: bytes, idx: int, language: str) -> TranscriptionResult:
    """Top-level worker function executed inside worker processes.

    Kept at module level so it's picklable and does not carry a reference
    to the parent instance (which contains locks/semaphores).
    """
    global _process_model

    if _process_model is None:
        raise RuntimeError("Worker model not initialized")

    try:
        import soundfile as sf
        import numpy as np
        import librosa
    except ImportError:
        raise RuntimeError("Required packages missing in worker process")

    with io.BytesIO(bytes_io) as bs_io:
        data, sr = sf.read(bs_io)

    if 1 < data.ndim:
        data = np.mean(data, axis=1)

    target_sr = 16000
    data = data.astype("float32")

    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    result = _process_model.transcribe(data, language=language, task="transcribe")

    return TranscriptionResult(idx=idx, text=result.get("text", "").strip())


class _OpenAIWhisperImpl(BaseImpl):
    def __init__(self, cfg: Settings = settings):
        super().__init__(cfg)

        cpu_count = multiprocessing.cpu_count()
        workers = cfg.max_workers or max(1, cpu_count - 1)
        # ProcessPoolExecutor tends to give good core utilization.
        self.executor = ProcessPoolExecutor(
            max_workers=workers, initializer=self._init_process, initargs=(cfg,)
        )

    @classmethod
    def _init_process(cls, cfg: Settings):
        """Initializing the Whisper model in worker processes"""
        global _process_model

        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError("OpenAI-Whisper not installed") from exc
        # load model into process memory once
        _process_model = whisper.load_model(cfg.whisper_model_size, device="cpu")

    def batch_transcribe(
        self, audio_bytes_list: List[bytes]
    ) -> List[TranscriptionResult]:
        futures = []

        for idx, audio_bytes in enumerate(audio_bytes_list):
            # submit a top-level worker function so only simple values are pickled
            futures.append(
                self.executor.submit(
                    _proc_transcribe, audio_bytes, idx, self.cfg.language
                )
            )

        results: List[TranscriptionResult] = []

        for future in as_completed(futures):
            results.append(future.result())

        results.sort(key=lambda r: r.idx)
        return results

    def shutdown(self) -> None:
        try:
            self.executor.shutdown(wait=True)
        except Exception:
            pass


# register this implementation for `cpu` device key
register_impl("cpu", _OpenAIWhisperImpl)
