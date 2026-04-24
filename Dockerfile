FROM python:3.11-slim
 
WORKDIR /app
 
# libcurl4t64 is required by the .eim ELF binary (it links libcurl.so.4).
# On Debian Trixie (python:3.11-slim base) the package is named libcurl4t64.
# Installing it pulls in all its transitive deps (ssl, crypto, ssh, etc.) automatically.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libcurl4t64 \
    && rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY main.py .
COPY model.eim .
RUN chmod +x model.eim
 
EXPOSE 10000
 
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
 
