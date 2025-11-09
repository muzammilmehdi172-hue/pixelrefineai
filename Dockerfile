FROM python:3.11-slim

WORKDIR /app

COPY . .

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch CPU wheels compatible with Python 3.11
RUN pip install torch==2.5.0+cpu torchvision==0.20.1+cpu torchaudio==2.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install other Python dependencies
RUN pip install -r requirements.txt

# Expose port if using a web service
EXPOSE 8000

CMD ["python", "app.py"]
