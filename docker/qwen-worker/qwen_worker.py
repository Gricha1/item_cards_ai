from __future__ import annotations

from io import BytesIO
from threading import Lock

import torch
from diffusers import BitsAndBytesConfig, QwenImageEditPipeline, QwenImageTransformer2DModel
from diffusers.quantizers import PipelineQuantizationConfig
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

MODEL_ID = "Qwen/Qwen-Image-Edit"
app = FastAPI(title="ItemCards AI Qwen worker")
_pipeline: QwenImageEditPipeline | None = None
_lock = Lock()


def pipeline() -> QwenImageEditPipeline:
    global _pipeline
    if _pipeline is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside the Qwen worker")
        quant_kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        }
        transformer_quant_config = BitsAndBytesConfig(**quant_kwargs)

        # Diffusers only balances whole *pipeline* components, which leaves
        # Qwen's transformer too large for one 24 GB GPU.  Loading that
        # component separately lets Accelerate split its blocks across both
        # visible GPUs before the pipeline is assembled.
        transformer = QwenImageTransformer2DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            dtype=torch.bfloat16,
            quantization_config=transformer_quant_config,
            device_map="auto",
            # Leave headroom for the text encoder, VAE and the denoising
            # activations on GPU 0.  Ten GiB per card forces the transformer
            # blocks to be placed on both visible GPUs.
            max_memory={0: "10GiB", 1: "10GiB"},
        )
        text_quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs=quant_kwargs,
            components_to_quantize=["text_encoder"],
        )
        _pipeline = QwenImageEditPipeline.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            transformer=transformer,
            quantization_config=text_quant_config,
            device_map="cuda",
        )
    return _pipeline


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
        f"Create a realistic ecommerce image of an adult {gender} model, {framing}, "
        "wearing the garment from the second reference image. Preserve the garment's color, "
        "silhouette, zipper details, collar shape and material. Plain light studio background, "
        "marketplace catalog photography, no text, no watermark."
    )
    try:
        person_image = Image.open(BytesIO(await person.read())).convert("RGB")
        garment_image = Image.open(BytesIO(await garment.read())).convert("RGB")
        with _lock:
            image = pipeline()(
                image=[person_image, garment_image],
                prompt=prompt,
                negative_prompt="distorted anatomy, duplicate sleeves, duplicate arms, extra limbs, watermark, text",
                true_cfg_scale=3.5,
                num_inference_steps=30,
                width=576,
                height=768,
            ).images[0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
