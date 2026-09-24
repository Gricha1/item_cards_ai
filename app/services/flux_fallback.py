from __future__ import annotations

from pathlib import Path

from PIL import Image


class FluxFallbackError(RuntimeError):
    pass


class FluxFallbackService:
    """Best-effort image-to-image fallback, intentionally labelled as lower fidelity."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                import torch
                from diffusers import FluxImg2ImgPipeline

                self._pipeline = FluxImg2ImgPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
                self._pipeline.enable_model_cpu_offload()
            except Exception as error:
                raise FluxFallbackError("Не удалось загрузить FLUX.1-schnell. Проверьте место на диске и CUDA.") from error
        return self._pipeline

    def generate(self, person_path: Path, output_path: Path, gender: str, framing: str) -> Path:
        prompt = (
            f"Photorealistic ecommerce catalog photo of an adult {gender} model, {framing}, "
            "wearing a neutral upper garment, light plain studio background, natural proportions, "
            "product-focused composition, no words, no logo, no watermark."
        )
        try:
            source = Image.open(person_path).convert("RGB")
            image = self._get_pipeline()(
                prompt=prompt,
                image=source,
                strength=0.55,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256,
            ).images[0]
            image.save(output_path, "PNG")
            return output_path
        except FluxFallbackError:
            raise
        except Exception as error:
            raise FluxFallbackError("FLUX fallback не смог создать изображение.") from error
