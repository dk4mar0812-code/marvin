FROM python:3.11-slim
 
WORKDIR /app
 
# System libs needed by the .eim ELF binary and edge-impulse-linux SDK
# NOTE: python:3.11-slim uses Debian Bookworm where libasound2 was renamed
# to libasound2t64. libatlas/libopenblas are not needed; numpy bundles its own BLAS.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    libasound2t64 \
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
