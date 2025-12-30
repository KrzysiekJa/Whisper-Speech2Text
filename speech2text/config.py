import pathlib
from pydantic_settings import BaseSettings, SettingsConfigDict


DOTENV = pathlib.Path(__file__).resolve().parent / ".env"


class Settings(BaseSettings):
    whisper_model_size: str
    faster_whisper_model_size: str
    device: str
    silence_offset_dbfs: float
    min_silence_len_ms: int
    keep_silence_ms: int
    max_workers: int
    debug_dump_chunks: bool
    audio_format: str
    language: str

    model_config = SettingsConfigDict(
        env_file=DOTENV,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
