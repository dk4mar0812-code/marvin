FROM python:3.11-slim
 
WORKDIR /app
 
# Only runtime libs needed by the .eim ELF binary itself
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY main.py .
COPY model.eim .
RUN chmod +x model.eim
 
EXPOSE 10000
 
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
