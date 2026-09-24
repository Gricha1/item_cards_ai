from __future__ import annotations

from pathlib import Path

from PIL import Image


class FashnVtonError(RuntimeError):
    pass


class FashnVtonService:
    """Lazy local FASHN VTON v1.5 inference; model loads only after user input."""

    def __init__(self, weights_dir: Path, category: str) -> None:
        self.weights_dir = weights_dir
        self.category = category
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if not (self.weights_dir / "model.safetensors").exists():
            raise FashnVtonError("Веса FASHN VTON не найдены. На сервере выполните scripts/setup_server.sh.")
        try:
            from fashn_vton import TryOnPipeline

            self._pipeline = TryOnPipeline(weights_dir=str(self.weights_dir))
            return self._pipeline
        except Exception as error:
            raise FashnVtonError("Не удалось загрузить FASHN VTON. Проверьте CUDA, веса и свободную VRAM.") from error

    def generate(self, person_path: Path, garment_path: Path, output_path: Path) -> Path:
        try:
            pipeline = self._get_pipeline()
            result = pipeline(
                person_image=Image.open(person_path).convert("RGB"),
                garment_image=Image.open(garment_path).convert("RGB"),
                category=self.category,
            )
            result.images[0].save(output_path, "PNG")
            return output_path
        except FashnVtonError:
            raise
        except Exception as error:
            raise FashnVtonError("FASHN VTON не смог обработать эту фотографию.") from error
