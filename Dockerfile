FROM python:3.9-slim  # basicsr 1.4.2 requires Python ≤ 3.9

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .

# Install PyTorch 1.13.1 (last version fully compatible with basicsr 1.4.2)
RUN pip install --no-cache-dir \
    torch==1.13.1+cpu \
    torchvision==0.14.1+cpu \
    torchaudio==0.13.1+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p uploads results models
EXPOSE 8000
CMD ["python", "app.py"]
