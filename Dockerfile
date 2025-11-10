# Use Python 3.9 (required for basicsr + torch 1.13)
FROM python:3.9-slim

# Install system packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary folders
RUN mkdir -p uploads results models

EXPOSE 8000

CMD ["python", "app.py"]
