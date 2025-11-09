# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for opencv and numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 wget curl && \
    rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt requirements.txt
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Render expects
EXPOSE 10000

# Start the Flask app
CMD ["python", "app.py"]
