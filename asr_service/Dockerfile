FROM python:3.11-slim

WORKDIR /app

# ffmpeg is required for audio format conversion (webm → wav)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gcc \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pre-install whisper dependencies
RUN pip install --no-cache-dir numpy torch tqdm more-itertools tiktoken

# Copy requirements and install without build isolation
COPY requirements.txt .
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

COPY . .

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]