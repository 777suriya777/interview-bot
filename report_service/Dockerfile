FROM python:3.11-slim

WORKDIR /app

# WeasyPrint needs Cairo, Pango, and other system libraries for PDF rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8005
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
