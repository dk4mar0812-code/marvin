FROM python:3.11-slim
 
WORKDIR /app
 
# gcc is needed to compile pyaudio from source.
# libasound2t64 is the correct package name on Debian Trixie (python:3.11-slim base).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
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
 
EXPOSE 10000
 
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
 
