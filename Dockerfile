FROM python:3.11-slim

WORKDIR /app

# System libs needed by the .eim ELF binary and edge-impulse-linux SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    libatlas-base-dev \
    libopenblas-dev \
    libasound2-dev \
    libportaudio2 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY model.eim .
RUN chmod +x model.eim

# Render injects $PORT at runtime (default 10000).
# We read it in the CMD so the correct port is always used.
EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
