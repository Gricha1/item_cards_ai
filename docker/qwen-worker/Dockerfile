FROM vllm/vllm-openai:v0.8.5

USER root
# Keep CUDA/PyTorch from the vLLM base but update the model stack together.
# Qwen Image Edit needs current Diffusers, Transformers and Hugging Face Hub.
RUN pip install --no-cache-dir --upgrade \
    "diffusers==0.40.0" "transformers==5.17.0" "accelerate>=1.10" \
    "huggingface-hub==1.33.0" "hf-xet==1.6.0" "python-multipart==0.0.20"

WORKDIR /opt/qwen-worker
COPY docker/qwen-worker/qwen_worker.py /opt/qwen-worker/qwen_worker.py
ENTRYPOINT ["python3", "/opt/qwen-worker/qwen_worker.py"]
