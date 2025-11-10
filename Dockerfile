# Use Python 3.11 (stable with PyTorch 2.1.2)
FROM python:3.11-slim

# Prevent root pip warnings
ENV PIP_ROOT_USER_ACTION=ignore

# Install dependencies for OpenCV and basic libs
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 wget curl && \
    rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ⚡ FIX 1: Install a compatible NumPy version FIRST (before torch)
RUN pip install --no-cache-dir numpy==1.26.4

# ⚡ FIX 2: Install PyTorch CPU + torchvision manually (compatible versions)
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    torchaudio==2.1.2+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install other dependencies (Flask, OpenCV, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Create folders (safety)
RUN mkdir -p uploads results models

# Expose port for Render
EXPOSE 8000

# Run Flask
CMD ["python", "app.py"]
