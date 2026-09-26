FROM vllm/vllm-openai:v0.8.5

USER root
# DiffSynth reads the mounted safetensors directly.  Do not install the old
# Diffusers worker stack here: its Hub 1.x requirement conflicts with the
# Transformers 4.x API required by DiffSynth's Qwen2.5-VL integration.
RUN pip install --no-cache-dir --upgrade \
    "transformers==4.51.3" "accelerate>=1.10" \
    "python-multipart==0.0.20" \
    "git+https://github.com/modelscope/DiffSynth-Studio.git"

WORKDIR /opt/qwen-worker
COPY docker/qwen-worker/qwen_worker.py /opt/qwen-worker/qwen_worker.py
ENTRYPOINT ["python3", "/opt/qwen-worker/qwen_worker.py"]
