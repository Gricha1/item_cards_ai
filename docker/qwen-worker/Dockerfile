FROM vllm/vllm-openai:v0.8.5

USER root
RUN pip install --no-cache-dir \
    "diffusers>=0.40" "transformers>=4.57" "accelerate>=1.10" \
    "bitsandbytes>=0.46" "fastapi>=0.115" "uvicorn[standard]>=0.34" \
    "python-multipart>=0.0.20" "pillow>=10"

WORKDIR /opt/qwen-worker
COPY qwen_worker.py /opt/qwen-worker/qwen_worker.py
ENTRYPOINT ["python3", "/opt/qwen-worker/qwen_worker.py"]
