FROM python:3.10-slim

# SYSTEM LEVEL OVERRIDE: Forces the system to clean caches and ignore secure certificate time checks
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update -o Acquire::Check-Valid-Until=false || true

# Run a secure, verified system installation framework loop
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Pre-fetch clean lightweight CPU packages for deep learning pipelines
RUN pip install --no-cache-dir --extra-index-url https://pytorch.org torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

# Mirror our local folder structure inside the container storage natively
COPY . .

EXPOSE 8000

# Fire up the application pointing straight to your sub-module path
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
