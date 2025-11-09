# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU versions of PyTorch, torchvision, torchaudio
RUN pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install other Python dependencies (if you have requirements.txt)
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# Default command
CMD ["python", "app.py"]
