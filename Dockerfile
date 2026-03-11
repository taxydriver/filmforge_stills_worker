FROM runpod/worker-comfyui:5.7.1-base-cuda12.8.1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
