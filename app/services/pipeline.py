from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.services.fashn_vton import FashnVtonService
from app.services.flux_fallback import FluxFallbackError, FluxFallbackService
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

    def template_for(self, gender: str, framing: str, age_range: str | None = None) -> Path:
        frame = "full" if framing == "full" else "waist"
        templates_dir = Path(__file__).resolve().parent.parent / "assets" / "templates"
        age_specific = templates_dir / f"{gender}_{frame}_{age_range}.png" if age_range else None
        requested = templates_dir / f"{gender}_{frame}.png"
        alternate_frame = "waist" if frame == "full" else "full"
        fallback = templates_dir / f"{gender}_{alternate_frame}.png"

        for candidate in (age_specific, requested, fallback):
            if candidate and candidate.exists():
                return candidate

        raise TemplateNotFoundError(
            f"Не найден шаблон {requested.name}. Добавьте собственное изображение модели в app/assets/templates/."
        )

    def generate(
        self,
        garment_path: Path,
        gender: str,
        framing: str,
        progress_callback: Callable[[str, int, int], None] | None = None,
        *,
        age_range: str | None = None,
        seed: int = 42,
    ) -> list[GeneratedVariant]:
        person_path = self.template_for(gender, framing, age_range)
        first_path = unique_path(self.settings.output_dir)
        second_path = unique_path(self.settings.output_dir)
        first = self.fashn.generate(
            person_path, garment_path, first_path, progress_callback, variant="first", seed=seed
        )

        try:
            second = self.qwen.generate(person_path, garment_path, second_path, gender, framing)
            return [GeneratedVariant("Вариант 1 — FASHN VTON", first), GeneratedVariant("Вариант 2 — Qwen Image Edit", second)]
        except QwenUnavailableError:
            try:
                second = self.flux.generate(person_path, second_path, gender, framing)
                return [
                    GeneratedVariant("Вариант 1 — FASHN VTON", first),
                    GeneratedVariant("Вариант 2 — FLUX fallback (точность одежды ниже)", second),
                ]
            except FluxFallbackError:
                # FLUX.1-schnell is gated on Hugging Face. Keep the promised
                # two useful images available without an API key by sampling a
                # second FASHN variation with a different reproducible seed.
                second = self.fashn.generate(
                    person_path,
                    garment_path,
                    second_path,
                    progress_callback,
                    variant="second",
                    seed=seed + 1,
                )
                return [
                    GeneratedVariant("Вариант 1 — FASHN VTON", first),
                    GeneratedVariant("Вариант 2 — FASHN VTON (альтернативная вариация)", second),
                ]
