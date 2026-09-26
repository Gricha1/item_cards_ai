from __future__ import annotations

import glob
import traceback
from io import BytesIO
from pathlib import Path
from threading import Lock

import torch


def _disable_optional_compile(function, *args, **kwargs):
    """Flex Attention compilation is unsupported/very slow on TITAN RTX."""
    return function


# DiffSynth imports Flex Attention at module import time and wraps it in
# torch.compile even though this worker uses the standard attention backend.
# On Turing that creates dozens of compiler workers before any image work.
torch.compile = _disable_optional_compile

# PyTorch 2.6 may dispatch scaled-dot-product attention to a cuDNN backend
# that is not compatible with the TITAN RTX (Turing).  Qwen's visual encoder
# reaches that path before diffusion starts and the driver reports an illegal
# instruction.  The math backend is slower but portable and uses FP16 here.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

# Existing model snapshot mounted by Docker; changing engines does not require
# another 54 GB download.
MODEL_DIR = Path("/models/hub/models--Qwen--Qwen-Image-Edit/snapshots/ac7f9318f633fc4b5778c59367c8128225f1e3de")
app = FastAPI(title="ItemCards AI Qwen worker")
_pipeline: QwenImagePipeline | None = None
_lock = Lock()


def pipeline() -> QwenImagePipeline:
    global _pipeline
    if _pipeline is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside the Qwen worker")
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            # TITAN RTX (Turing, compute capability 7.5) has neither native
            # FP8 nor BF16 instructions.  FP16 is the compatible low-VRAM
            # transport/computation format on this GPU generation.
            "onload_dtype": torch.float16,
            "onload_device": "cpu",
            "preparing_dtype": torch.float16,
            "preparing_device": "cuda",
            "computation_dtype": torch.float16,
            "computation_device": "cuda",
        }
        _pipeline = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.float16,
            device="cuda",
            model_configs=[
                ModelConfig(
                    path=glob.glob(str(MODEL_DIR / "transformer/diffusion_pytorch_model*.safetensors")),
                    **vram_config,
                ),
                ModelConfig(path=glob.glob(str(MODEL_DIR / "text_encoder/model*.safetensors")), **vram_config),
                ModelConfig(path=str(MODEL_DIR / "vae/diffusion_pytorch_model.safetensors"), **vram_config),
            ],
            processor_config=ModelConfig(path=str(MODEL_DIR / "processor")),
            vram_limit=torch.cuda.mem_get_info("cuda")[1] / 1024**3 - 0.5,
        )
    return _pipeline


def reference_sheet(person: Image.Image, garment: Image.Image) -> Image.Image:
    """Provide the single-image Qwen Edit checkpoint a clear two-panel reference."""
    canvas = Image.new("RGB", (1024, 768), "white")
    for source, x in ((person, 0), (garment, 512)):
        panel = source.copy()
        panel.thumbnail((496, 744))
        canvas.paste(panel, (x + (512 - panel.width) // 2, (768 - panel.height) // 2))
    return canvas


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": torch.cuda.is_available(), "loaded": _pipeline is not None}


@app.post("/generate")
async def generate(
    person: UploadFile = File(...),
    garment: UploadFile = File(...),
    gender: str = Form(...),
    framing: str = Form(...),
) -> Response:
    prompt = (
        "This is a two-panel reference. LEFT: an adult "
        f"{gender} model; RIGHT: a garment. Create one realistic ecommerce photo of the left "
        "model wearing the exact right garment. Preserve garment color, material, silhouette, zipper and collar. "
        f"{framing} framing, clean light studio background, catalog photography, no text, no watermark."
    )
    try:
        person_image = Image.open(BytesIO(await person.read())).convert("RGB")
        garment_image = Image.open(BytesIO(await garment.read())).convert("RGB")
        with _lock:
            image = pipeline()(
                prompt=prompt,
                edit_image=reference_sheet(person_image, garment_image),
                edit_image_auto_resize=True,
                seed=42,
                num_inference_steps=30,
                width=576,
                height=768,
            )
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")
    except Exception as error:
        # Keep the complete CUDA location in container logs.  The HTTP client
        # only needs a concise 503 response, but unsupported Turing kernels
        # otherwise lose their actionable traceback here.
        print(traceback.format_exc(), flush=True)
        raise HTTPException(status_code=503, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
