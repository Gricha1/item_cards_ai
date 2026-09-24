from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.services.fashn_vton import FashnVtonService
from app.services.flux_fallback import FluxFallbackService
from app.services.qwen_edit import QwenEditService, QwenUnavailableError
from app.utils.temp_files import unique_path


class TemplateNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratedVariant:
    label: str
    path: Path


class GenerationPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.fashn = FashnVtonService(
            settings.fashn_weights_dir,
            settings.fashn_category,
            low_memory=settings.fashn_low_memory,
            num_timesteps=settings.fashn_num_timesteps,
        )
        self.qwen = QwenEditService(settings.qwen_enabled, settings.qwen_min_vram_gb)
        self.flux = FluxFallbackService(settings.flux_model_id)

    def template_for(self, gender: str, framing: str) -> Path:
        frame = "full" if framing == "full" else "waist"
        path = Path(__file__).resolve().parent.parent / "assets" / "templates" / f"{gender}_{frame}.png"
        if not path.exists():
            raise TemplateNotFoundError(
                f"Не найден шаблон {path.name}. Добавьте собственное изображение модели в app/assets/templates/."
            )
        return path

    def generate(self, garment_path: Path, gender: str, framing: str) -> list[GeneratedVariant]:
        person_path = self.template_for(gender, framing)
        first_path = unique_path(self.settings.output_dir)
        second_path = unique_path(self.settings.output_dir)
        first = self.fashn.generate(person_path, garment_path, first_path)

        try:
            second = self.qwen.generate(person_path, garment_path, second_path, gender, framing)
            return [GeneratedVariant("Вариант 1 — FASHN VTON", first), GeneratedVariant("Вариант 2 — Qwen Image Edit", second)]
        except QwenUnavailableError:
            second = self.flux.generate(person_path, second_path, gender, framing)
            return [GeneratedVariant("Вариант 1 — FASHN VTON", first), GeneratedVariant("Вариант 2 — FLUX fallback (точность одежды ниже)", second)]
