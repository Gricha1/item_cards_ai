from __future__ import annotations

from pathlib import Path

from PIL import Image


class FashnVtonError(RuntimeError):
    pass


class FashnVtonService:
    """Lazy local FASHN VTON v1.5 inference; model loads only after user input."""

    def __init__(
        self,
        weights_dir: Path,
        category: str,
        *,
        low_memory: bool = False,
        num_timesteps: int = 20,
    ) -> None:
        self.weights_dir = weights_dir
        self.category = category
        self.low_memory = low_memory
        self.num_timesteps = num_timesteps
        self._pipeline = None

    @staticmethod
    def _enable_low_memory_denoising(pipeline) -> None:
        """Avoid FASHN's 2x classifier-free-guidance batch on small GPUs.

        The upstream pipeline invokes ``forward_for_cfg`` for conditional and
        unconditional predictions in one concatenated batch.  Its fixed
        864x576 resolution makes that peak too large for an 8 GB RTX 2070.
        Returning the conditional prediction for both paths preserves the
        upstream scheduler while needing just one forward pass.
        """
        import torch

        model = pipeline.tryon_model

        def forward_without_cfg(noisy_images, timesteps, **kwargs):
            kwargs["mask"] = torch.ones(
                noisy_images.shape[0], device=noisy_images.device, dtype=torch.bool
            )
            prediction = model.forward(noisy_images, timesteps, **kwargs)["x"]
            return {"v_c": prediction, "v_u": prediction}

        model.forward_for_cfg = forward_without_cfg

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if not (self.weights_dir / "model.safetensors").exists():
            raise FashnVtonError("Веса FASHN VTON не найдены. На сервере выполните scripts/setup_server.sh.")
        try:
            from fashn_vton import TryOnPipeline

            self._pipeline = TryOnPipeline(weights_dir=str(self.weights_dir))
            if self.low_memory:
                self._enable_low_memory_denoising(self._pipeline)
            return self._pipeline
        except Exception as error:
            raise FashnVtonError("Не удалось загрузить FASHN VTON. Проверьте CUDA, веса и свободную VRAM.") from error

    def generate(self, person_path: Path, garment_path: Path, output_path: Path) -> Path:
        try:
            pipeline = self._get_pipeline()
            # Release cache held by a previous request before the fixed-size
            # FASHN denoising pass starts.
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            result = pipeline(
                person_image=Image.open(person_path).convert("RGB"),
                garment_image=Image.open(garment_path).convert("RGB"),
                category=self.category,
                num_timesteps=self.num_timesteps,
            )
            result.images[0].save(output_path, "PNG")
            return output_path
        except FashnVtonError:
            raise
        except Exception as error:
            raise FashnVtonError("FASHN VTON не смог обработать эту фотографию.") from error
