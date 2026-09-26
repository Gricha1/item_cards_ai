FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade \
        "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart==0.0.20" \
        "onnxruntime-gpu==1.20.2" \
        "git+https://github.com/fashn-AI/fashn-vton-1.5.git"

WORKDIR /opt/fashn-worker
COPY docker/fashn-worker/fashn_worker.py /opt/fashn-worker/fashn_worker.py
ENTRYPOINT ["python", "/opt/fashn-worker/fashn_worker.py"]
