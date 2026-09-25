FROM vllm/vllm-openai:v0.8.5

USER root
# The vLLM base already provides CUDA 12.4, PyTorch, Transformers, Accelerate,
# BitsAndBytes, FastAPI and Uvicorn. Avoid resolving/reinstalling PyTorch here:
# it can take minutes and pull incompatible CUDA wheels.
RUN pip install --no-cache-dir --no-deps \
    "diffusers==0.39.0" "huggingface-hub==0.36.0" "hf-xet==1.6.0" "python-multipart==0.0.20"

WORKDIR /opt/qwen-worker
COPY docker/qwen-worker/qwen_worker.py /opt/qwen-worker/qwen_worker.py
ENTRYPOINT ["python3", "/opt/qwen-worker/qwen_worker.py"]
