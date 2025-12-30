from typing import Dict, Type, Callable, List
import importlib

from config import Settings
from classes.BaseImpl import BaseImpl


_registry: Dict[str, Type[BaseImpl]] = {}


def register_impl(key: str, cls: Type[BaseImpl]) -> None:
    """Register an implementation class for a given key (e.g., device).

    Keys are arbitrary strings; `Transcriber` uses `cfg.device` by default.
    """
    _registry[key] = cls


def impl_register(key: str) -> Callable[[Type[BaseImpl]], Type[BaseImpl]]:
    """Decorator to register an implementation class."""

    def _decorator(cls: Type[BaseImpl]) -> Type[BaseImpl]:
        register_impl(key, cls)
        return cls

    return _decorator


DEVICE_MODULE_CANDIDATES: Dict[str, List[str]] = {
    "cpu": [
        "speech2text.classes._OpenAIWhisperImpl",
        "classes._OpenAIWhisperImpl",
    ],
    "cuda": [
        "speech2text.classes._FasterWhisperImpl",
        "classes._FasterWhisperImpl",
    ],
}


def _import_candidates(device: str) -> None:
    """Attempt to import candidate modules for a device key to trigger
    registration side-effects.
    """
    modules = DEVICE_MODULE_CANDIDATES.get(device, [])

    for module in modules:
        try:
            importlib.import_module(module)
        except Exception:
            print(f"Failed to import module {module} for device {device}")


def get_impl_class(cfg: Settings) -> Type[BaseImpl]:
    impl = _registry.get(cfg.device)

    if impl is not None:
        return impl

    _import_candidates(cfg.device)

    impl = _registry.get(cfg.device)

    if impl is None:
        raise ValueError(
            f"No implementation registered for device: {cfg.device}."
            " Ensure the implementation module is imported or register it"
        )
    return impl
