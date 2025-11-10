FROM python:3.11-slim

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy everything (models/ included)
COPY . .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU + torchvision
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    torchaudio==2.1.2+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install your app deps
RUN pip install --no-cache-dir -r requirements.txt

# Create upload/results dirs
RUN mkdir -p uploads results

EXPOSE 8000
CMD ["python", "app.py"]
