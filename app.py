from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import urllib.request, urllib.error

# === FOLDERS ===
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
MODEL_DIR = 'models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# === MODEL URLs ===
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
GFPGAN_PATH = os.path.join(MODEL_DIR, 'GFPGANv1.4.pth')

REAL_ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
REAL_ESRGAN_PATH = os.path.join(MODEL_DIR, 'RealESRGAN_x4plus.pth')

# === SAFE DOWNLOAD ===
def download_if_missing(url, path):
    if not os.path.exists(path):
        print(f"📥 Downloading {os.path.basename(path)}...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"✅ {os.path.basename(path)} ready.")
        except Exception as e:
            print(f"❌ Failed to download {os.path.basename(path)}: {e}")
            raise

# Download both models if missing
try:
    download_if_missing(GFPGAN_URL, GFPGAN_PATH)
    download_if_missing(REAL_ESRGAN_URL, REAL_ESRGAN_PATH)
except Exception as e:
    print(f"⚠️ Model download error: {e}")

# === LOAD MODELS ===
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

gfpgan_restorer = GFPGANer(
    model_path=GFPGAN_PATH,
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None,
    half=False
)

realesrgan_enhancer = RealESRGANer(
    scale=4,
    model_path=REAL_ESRGAN_PATH,
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    tile=256,
    tile_pad=10,
    pre_pad=0,
    half=False
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

        # 1️⃣ Face Restoration (GFPGAN)
        _, _, restored_img = gfpgan_restorer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )

        # 2️⃣ 4x Upscaling (Real-ESRGAN)
        upscaled, _ = realesrgan_enhancer.enhance(restored_img, outscale=4)

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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
