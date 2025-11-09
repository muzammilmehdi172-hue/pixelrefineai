# Use Python 3.10 for compatibility with GFPGAN + RealESRGAN
FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Upgrade pip and basic tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy before torch to fix numpy runtime issue
RUN pip install --no-cache-dir numpy==1.26.4

# ✅ Install the correct compatible PyTorch + torchvision versions
# Basicsr & GFPGAN expect torch==1.12.1 + torchvision==0.13.1
RUN pip install --no-cache-dir \
    torch==1.12.1+cpu \
    torchvision==0.13.1+cpu \
    torchaudio==0.12.1+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install the rest of the dependencies
RUN pip install --no-cache-dir flask opencv-python-headless basicsr==1.4.2 gfpgan==1.3.8 realesrgan==0.3.0

# Create folders
RUN mkdir -p uploads results models

EXPOSE 8000

CMD ["python", "app.py"]
