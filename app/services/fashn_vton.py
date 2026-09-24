from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import MethodType

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

        # RTX 20xx uses PyTorch's math SDPA fallback for this model's long
        # image-token sequence.  It materializes an attention matrix large
        # enough to OOM even after CFG batching is removed.  Split queries;
        # keys and values stay shared and the result is mathematically the
        # same attention operation.
        import fashn_vton.tryon_mmdit as tryon_mmdit

        if not getattr(tryon_mmdit, "_item_cards_chunked_attention", False):
            attention_chunk_size = 128

            def chunked_attention(query, key, value):
                if query.shape[-2] <= attention_chunk_size:
                    return torch.nn.functional.scaled_dot_product_attention(query, key, value)
                chunks = []
                for start in range(0, query.shape[-2], attention_chunk_size):
                    chunk = query[..., start : start + attention_chunk_size, :]
                    chunks.append(torch.nn.functional.scaled_dot_product_attention(chunk, key, value))
                return torch.cat(chunks, dim=-2)

            tryon_mmdit._attn_processor = chunked_attention
            tryon_mmdit._item_cards_chunked_attention = True

        def forward_without_cfg(noisy_images, timesteps, **kwargs):
            kwargs["mask"] = torch.ones(
                noisy_images.shape[0], device=noisy_images.device, dtype=torch.bool
            )
            prediction = model.forward(noisy_images, timesteps, **kwargs)["x"]
            return {"v_c": prediction, "v_u": prediction}

        model.forward_for_cfg = forward_without_cfg

    @staticmethod
    def _install_progress_sampler(
        pipeline,
        progress_callback: Callable[[str, int, int], None] | None,
        variant: str,
    ) -> None:
        """Expose FASHN's denoising step count without changing its output."""
        import torch
        from fashn_vton.pipeline import get_rf_schedule, tensor_to_pil

        @torch.inference_mode()
        def sample_with_progress(
            _pipeline,
            *,
            ca_images,
            garment_images,
            person_poses,
            garment_poses,
            garment_categories,
            num_timesteps: int = 30,
            time_shift_mu: float = 1.5,
            guidance_scale: float = 1.5,
            skip_cfg_last_n_steps: int = 1,
            use_tqdm: bool = True,
        ):
            del use_tqdm
            device, dtype = ca_images.device, ca_images.dtype
            batch_size = ca_images.shape[0]
            channels, height, width = _pipeline.tryon_model.channels_in, *_pipeline.tryon_model.input_shape
            images = torch.randn((batch_size, channels, height, width), dtype=dtype, device=device)
            timesteps = get_rf_schedule(num_steps=num_timesteps, mu=time_shift_mu)
            model_kwargs = {
                "person_poses": person_poses,
                "garment_poses": garment_poses,
                "ca_images": ca_images,
                "garment_images": garment_images,
                "garment_categories": garment_categories,
            }

            for step_index, (current_time, previous_time) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                delta = previous_time - current_time
                time_vector = torch.full((batch_size,), current_time, dtype=dtype, device=device)
                prediction = _pipeline.tryon_model.forward_for_cfg(images, time_vector, **model_kwargs)
                conditional, unconditional = prediction["v_c"], prediction["v_u"]
                if skip_cfg_last_n_steps > 0 and step_index >= num_timesteps - skip_cfg_last_n_steps:
                    guided = conditional
                else:
                    guided = unconditional + guidance_scale * (conditional - unconditional)
                images = images + delta * guided
                if progress_callback is not None:
                    progress_callback(variant, step_index + 1, num_timesteps)

            images = images.to(dtype=torch.float).clamp_(-1.0, 1.0)
            return [tensor_to_pil(image, unnormalize=True) for image in images]

        pipeline._sample = MethodType(sample_with_progress, pipeline)

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

    def generate(
        self,
        person_path: Path,
        garment_path: Path,
        output_path: Path,
        progress_callback: Callable[[str, int, int], None] | None = None,
        *,
        variant: str = "first",
        seed: int = 42,
    ) -> Path:
        try:
            pipeline = self._get_pipeline()
            self._install_progress_sampler(pipeline, progress_callback, variant)
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
                seed=seed,
            )
            result.images[0].save(output_path, "PNG")
            return output_path
        except FashnVtonError:
            raise
        except Exception as error:
            raise FashnVtonError("FASHN VTON не смог обработать эту фотографию.") from error
