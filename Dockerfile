FROM python:3.11-slim
 
WORKDIR /app
 
# Runtime libs needed by the .eim binary at inference time
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    libasound2t64 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*
 
# Build a pure-Python stub for pyaudio so edge-impulse-linux installs without
# needing gcc or portaudio headers. Our main.py only imports from
# edge_impulse_linux.runner directly, so real pyaudio is never called.
RUN mkdir -p /tmp/fake_pyaudio/pyaudio && \
    echo "" > /tmp/fake_pyaudio/pyaudio/__init__.py && \
    printf '[metadata]\nname = pyaudio\nversion = 0.2.14\n' \
        > /tmp/fake_pyaudio/setup.cfg && \
    printf 'from setuptools import setup\nsetup()\n' \
        > /tmp/fake_pyaudio/setup.py && \
    pip wheel /tmp/fake_pyaudio --no-deps -w /tmp/fake_wheels -q && \
    pip install /tmp/fake_wheels/pyaudio-0.2.14-py3-none-any.whl
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY main.py .
COPY model.eim .
RUN chmod +x model.eim
 
EXPOSE 10000
 
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
