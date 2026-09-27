from __future__ import annotations

from io import BytesIO
from threading import Lock

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from fashn_vton import TryOnPipeline

app = FastAPI(title="ItemCards AI FASHN worker")
lock = Lock()
pipeline: TryOnPipeline | None = None


def get_pipeline() -> TryOnPipeline:
    global pipeline
    if pipeline is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside the FASHN worker")
        pipeline = TryOnPipeline(weights_dir="/weights")
    return pipeline


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": torch.cuda.is_available(), "loaded": pipeline is not None}


@app.post("/generate")
async def generate(
    person: UploadFile = File(...),
    garment: UploadFile = File(...),
    seed: int = Form(42),
    garment_photo_type: str = Form("flat-lay"),
) -> Response:
    try:
        if garment_photo_type not in {"flat-lay", "model"}:
            raise ValueError("garment_photo_type must be 'flat-lay' or 'model'")
        person_image = Image.open(BytesIO(await person.read())).convert("RGB")
        garment_image = Image.open(BytesIO(await garment.read())).convert("RGB")
        with lock:
            result = get_pipeline()(
                person_image=person_image,
                garment_image=garment_image,
                category="tops",
                # A product photo (hanger, mannequin or flat lay) and a
                # garment already worn by a person need distinct conditioning
                # modes. The bot asks the user explicitly instead of guessing
                # from pixels and silently degrading either case.
                garment_photo_type=garment_photo_type,
                # Upstream describes 50 steps as the quality setting; 30 is
                # the balanced/faster preset.  GPU 2 has enough headroom for
                # the higher-fidelity mode.
                num_timesteps=50,
                seed=seed,
            )
        buffer = BytesIO()
        result.images[0].save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
