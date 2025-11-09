FROM python:3.11-slim

WORKDIR /app

# --- System dependencies (fixes "Numpy is not available") ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    wget \
    curl && \
    rm -rf /var/lib/apt/lists/*

# --- Copy and install Python packages ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy project files ---
COPY . .

# --- Expose and run Flask ---
EXPOSE 10000
CMD ["python", "app.py"]
