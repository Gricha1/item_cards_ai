from __future__ import annotations

from pathlib import Path

from PIL import Image


class QwenUnavailableError(RuntimeError):
    pass


class QwenEditService:
    """Optional Qwen editor. It is deliberately blocked on undersized GPUs."""

    MODEL_ID = "Qwen/Qwen-Image-Edit"

    def __init__(self, enabled: bool, min_vram_gb: int) -> None:
        self.enabled = enabled
        self.min_vram_gb = min_vram_gb
        self._pipeline = None

    def _assert_available(self) -> None:
        if not self.enabled:
            raise QwenUnavailableError("Qwen-Image-Edit отключён в QWEN_ENABLED.")
        try:
            import torch

            if not torch.cuda.is_available():
                raise QwenUnavailableError("Для Qwen требуется доступная CUDA GPU.")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if memory_gb < self.min_vram_gb:
                raise QwenUnavailableError(f"Qwen требует минимум {self.min_vram_gb} ГБ VRAM; доступно {memory_gb:.1f} ГБ.")
        except QwenUnavailableError:
            raise
        except Exception as error:
            raise QwenUnavailableError("Не удалось проверить GPU для Qwen.") from error

    def _get_pipeline(self):
        self._assert_available()
        if self._pipeline is None:
            try:
                import torch
                from diffusers import QwenImageEditPipeline

                self._pipeline = QwenImageEditPipeline.from_pretrained(self.MODEL_ID, torch_dtype=torch.bfloat16)
                self._pipeline.enable_model_cpu_offload()
            except Exception as error:
                raise QwenUnavailableError("Не удалось загрузить Qwen-Image-Edit.") from error
        return self._pipeline

    def generate(self, person_path: Path, garment_path: Path, output_path: Path, gender: str, framing: str) -> Path:
        prompt = (
            f"Generate a realistic ecommerce-style image of an adult {gender} model, {framing}, "
            "wearing the garment from the second reference image. Preserve garment color, silhouette, "
            "zipper details, collar shape, and visible appearance. Plain light studio background, "
            "marketplace catalog style, no text, no watermark."
        )
        try:
            image = self._get_pipeline()(image=[Image.open(person_path).convert("RGB"), Image.open(garment_path).convert("RGB")], prompt=prompt).images[0]
            image.save(output_path, "PNG")
            return output_path
        except QwenUnavailableError:
            raise
        except Exception as error:
            raise QwenUnavailableError("Qwen-Image-Edit не смог создать изображение.") from error
