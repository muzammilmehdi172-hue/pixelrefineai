from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import urllib.request

# === SETUP FOLDERS ===
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
MODEL_DIR = 'models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit (Render-friendly)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# === MODEL PATHS (HIGH-QUALITY) ===
GFPGAN_PATH = os.path.join(MODEL_DIR, 'GFPGANv1.4.pth')
ESRGAN_PATH = os.path.join(MODEL_DIR, 'RealESRGAN_x4plus.pth')

# === DOWNLOAD MODELS ONCE (IF MISSING) ===
def download_if_missing(url, path, name):
    if not os.path.exists(path):
        print(f"⬇️ Downloading {name} (this takes 1-2 min)...")
        urllib.request.urlretrieve(url, path)
        print(f"✅ {name} ready.")

download_if_missing(
    'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    GFPGAN_PATH,
    "GFPGANv1.4"
)
download_if_missing(
    'https://github.com/xinnt/Real-ESRGAN/releases/download/v1.0.0/RealESRGAN_x4plus.pth',
    ESRGAN_PATH,
    "RealESRGAN_x4plus"
)

# === LOAD MODELS (CPU MODE) ===
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# GFPGAN (face restoration)
gfpgan_restorer = GFPGANer(
    model_path=GFPGAN_PATH,
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None  # We'll upscale background separately with RealESRGAN
    half=False
)

# RealESRGAN x4 (HIGH QUALITY) – with memory-safe settings
realesrgan_enhancer = RealESRGANer(
    scale=4,
    model_path=ESRGAN_PATH,
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    tile=256,          # ⚠️ Critical: limits RAM usage (256–400 works on free tier)
    tile_pad=10,
    pre_pad=0,
    half=False         # CPU doesn't support FP16 → disable
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_to_fit(img, max_size=1200):
    """Optional: prevent huge images from crashing"""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file selected"}), 400
    file = request.files['file']
    if not file or file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['RESULT_FOLDER'], f"enhanced_{filename}")

    file.save(input_path)

    try:
        img = cv2.imread(input_path)
        if img is None:
            return jsonify({"error": "Could not read image"}), 500

        # 🔍 Optional: resize very large images to avoid OOM
        img = resize_to_fit(img, max_size=1200)

        # STEP 1: GFPGAN face restoration
        _, _, restored_img = gfpgan_restorer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )

        # STEP 2: RealESRGAN x4 upscaling (HIGH QUALITY)
        upscaled, _ = realesrgan_enhancer.enhance(restored_img)

        cv2.imwrite(output_path, upscaled)

        return jsonify({
            "success": True,
            "before_url": f"/uploads/{filename}",
            "after_url": f"/results/enhanced_{filename}"
        })

    except Exception as e:
        return jsonify({"error": f"Enhancement failed: {str(e)}"}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

# RENDER uses dynamic PORT
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)