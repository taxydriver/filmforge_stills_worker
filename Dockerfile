FROM runpod/worker-comfyui:5.7.1-base-cuda12.8.1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV COMFY_MODELS_DIR=/comfyui/models

# Keep worker-comfyui's startup flow so ComfyUI starts before RunPod handler.
# start.sh invokes `python -u /handler.py`, so we overwrite /handler.py
# with FilmForge's still-worker handler.
RUN cp /app/handler.py /handler.py

COPY start_filmforge.sh /start_filmforge.sh
RUN chmod +x /start_filmforge.sh

CMD ["/start_filmforge.sh"]
