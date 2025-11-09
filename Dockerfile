# Use Python 3.11 (compatible with torch 2.5.0)
FROM python:3.11-slim

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your files
COPY . .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU + torchvision
RUN pip install --no-cache-dir torch==2.5.0+cpu torchvision==0.20.1+cpu torchaudio==2.5.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install your other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create folders (in case not in Git)
RUN mkdir -p uploads results models

# Expose port
EXPOSE 8000

# Run Flask
CMD ["python", "app.py"]
