from typing import List, Optional

from config import Settings, settings
from classes.BaseImpl import BaseImpl, TranscriptionResult
from classes._impl_registry import get_impl_class


class Transcriber:
    """
    High-level transcriber that picks appropriate implementation and exposes
    a synchronous batch_transcribe API that returns list of (idx, text).
    """

    def __init__(self, cfg: Settings = settings, impl: Optional[BaseImpl] = None):
        self.cfg = cfg

        # allow dependency injection for testing or custom implementations
        if impl is not None:
            self._impl = impl
        else:
            impl_cls = get_impl_class(cfg)
            self._impl = impl_cls(cfg)

    def batch_transcribe(
        self, audio_bytes_list: List[bytes]
    ) -> List[TranscriptionResult]:
        return self._impl.batch_transcribe(audio_bytes_list)

    def shutdown(self) -> None:
        """Release any resources used by the implementation."""
        try:
            self._impl.shutdown()
        except Exception:
            pass
