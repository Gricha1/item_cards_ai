FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/IDM-VTON

# This research model is deliberately isolated from the production FASHN
# worker. Its licence is non-commercial and it is used here only for a
# side-by-side quality evaluation explicitly requested by the project owner.
# The CUDA 11.7 PyTorch base uses Python 3.8; 1.15.1 is the newest compatible
# ONNX Runtime GPU wheel in the package index.
RUN git clone --depth 1 https://github.com/yisol/IDM-VTON.git . \
    && pip install --no-cache-dir \
        accelerate==0.25.0 torchmetrics==1.2.1 tqdm==4.66.1 \
        transformers==4.36.2 diffusers==0.25.0 einops==0.7.0 \
        bitsandbytes==0.39.0 scipy==1.11.1 opencv-python \
        gradio==4.24.0 fvcore cloudpickle omegaconf pycocotools basicsr av \
        onnxruntime-gpu==1.15.1 \
    && pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# diffusers 0.25 imports cached_download, removed by later Hub releases.
RUN pip install --no-cache-dir huggingface-hub==0.20.3

# IDM-VTON's Gradio 4.24 demo uses Starlette's pre-0.28 template API.
# Newer FastAPI/Starlette releases are installed transitively by pip and make
# the demo fail during its localhost readiness check before inference starts.
RUN pip install --no-cache-dir \
    fastapi==0.103.2 starlette==0.27.0 \
    pydantic==2.4.2 pydantic-core==2.10.1

# The upstream demo defaults to loopback. The worker is intentionally bound
# only to ML4 localhost by docker's port publishing below.
RUN sed -i 's/image_blocks.launch()/image_blocks.launch(server_name="0.0.0.0")/' gradio_demo/app.py

EXPOSE 7860
CMD ["python", "gradio_demo/app.py"]
