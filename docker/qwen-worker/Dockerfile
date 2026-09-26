FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

USER root
# DiffSynth reads the mounted safetensors directly.  Do not install the old
# Diffusers worker stack here: its Hub 1.x requirement conflicts with the
# Transformers 4.x API required by DiffSynth's Qwen2.5-VL integration.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade \
    "transformers==4.54.0" "accelerate>=1.10" \
    "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart==0.0.20" \
    "git+https://github.com/modelscope/DiffSynth-Studio.git"

WORKDIR /opt/qwen-worker
COPY docker/qwen-worker/qwen_worker.py /opt/qwen-worker/qwen_worker.py
ENTRYPOINT ["python3", "/opt/qwen-worker/qwen_worker.py"]
