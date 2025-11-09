# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies for OpenCV and general requirements
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . .

# Upgrade pip and install all Python dependencies in one layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.1.2+cpu \
        torchvision==0.16.2+cpu \
        torchaudio==2.1.2+cpu \
        -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Create folders if not in repository
RUN mkdir -p uploads results models

# Expose Flask port (ensure your app runs on this port)
EXPOSE 8000

# Run Flask (for production, you can switch to gunicorn later)
CMD ["python", "app.py"]
