FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Fix 1: Added space and a dot at the end
COPY requirements.txt .

# Fix 2: Updated to the correct lightweight CPU URL to save memory
RUN pip install --no-cache-dir --extra-index-url https://pytorch.org torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

# Fix 3: Corrected the copy syntax to copy all local folders
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
