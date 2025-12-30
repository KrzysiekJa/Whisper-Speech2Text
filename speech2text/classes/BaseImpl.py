from typing import List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import Settings, settings


@dataclass
class TranscriptionResult:
    idx: int
    text: str
    confidence: Optional[float] = None


class BaseImpl(ABC):
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg

    def start(self) -> None:
        """Optional lifecycle hook to initialise resources (executors, models)."""
        pass

    def shutdown(self) -> None:
        """Optional lifecycle hook to release resources (executors, models)."""
        pass

    @abstractmethod
    def batch_transcribe(
        self, audio_bytes_list: List[bytes]
    ) -> List[TranscriptionResult]: ...
