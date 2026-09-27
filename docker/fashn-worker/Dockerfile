FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# ONNX Runtime does not automatically discover PyTorch's bundled CUDA 12.1
# libraries. Expose them so DWPose runs on the GPU instead of silently using CPU.
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade \
        "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart==0.0.20" \
        "onnxruntime-gpu==1.20.2" \
        "git+https://github.com/fashn-AI/fashn-vton-1.5.git"

WORKDIR /opt/fashn-worker
COPY docker/fashn-worker/fashn_worker.py /opt/fashn-worker/fashn_worker.py
ENTRYPOINT ["python", "/opt/fashn-worker/fashn_worker.py"]
