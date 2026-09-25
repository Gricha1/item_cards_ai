from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Environment-only configuration. No credential has a source-code default."""

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str = Field(min_length=10)
    telegram_proxy_url: str | None = None
    model_device: str = "cuda"
    qwen_enabled: bool = False
    qwen_min_vram_gb: int = 20
    qwen_remote_url: str | None = None
    output_dir: Path = BASE_DIR / "data" / "outputs"
    temp_dir: Path = BASE_DIR / "data" / "tmp"
    input_dir: Path = BASE_DIR / "data" / "inputs"
    fashn_weights_dir: Path = BASE_DIR / "weights" / "fashn-vton-1.5"
    fashn_category: str = "tops"
    # FASHN normally duplicates the denoising batch for classifier-free guidance.
    # On 8 GB cards this can exceed VRAM; low-memory mode uses the conditional
    # pass only, trading a little visual quality for a working local result.
    fashn_low_memory: bool = False
    fashn_num_timesteps: int = 50
    flux_model_id: str = "black-forest-labs/FLUX.1-schnell"
    log_level: str = "INFO"

    @field_validator("output_dir", "temp_dir", "input_dir", "fashn_weights_dir", mode="before")
    @classmethod
    def make_project_paths_absolute(cls, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else BASE_DIR / path

    def create_directories(self) -> None:
        for directory in (self.output_dir, self.temp_dir, self.input_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
