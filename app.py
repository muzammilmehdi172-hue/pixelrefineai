from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import urllib.request
import urllib.error

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
MODEL_DIR = 'models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# ✅ CORRECT, WORKING URLS (tested)
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

GFPGAN_PATH = os.path.join(MODEL_DIR, 'GFPGANv1.4.pth')
ESRGAN_PATH = os.path.join(MODEL_DIR, 'RealESRGAN_x2plus.pth')

def download_if_missing(url, path):
    if not os.path.exists(path):
        print(f"📥 Downloading {os.path.basename(path)}...")
        try:
            urllib.request.urlretrieve(url, path)
            print("✅ Download complete.")
        except Exception as e:
            print(f"❌ Failed: {e}")
            raise

# Download on startup
download_if_missing(GFPGAN_URL, GFPGAN_PATH)
download_if_missing(ESRGAN_URL, ESRGAN_PATH)

from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

gfpgan_restorer = GFPGANer(
    model_path=GFPGAN_PATH,
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

# ⚠️ Use x2plus model (25 MB) → scale=2
realesrgan_enhancer = RealESRGANer(
    scale=2,
    model_path=ESRGAN_PATH,
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
    tile=256,
    half=False
)

# ... rest of your Flask routes ...

