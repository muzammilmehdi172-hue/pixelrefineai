from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import urllib.request
import urllib.error
import sqlite3
from contextlib import closing
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image

# === INIT ===
app = Flask(__name__)
app.secret_key = 'pixelrefine-secret-key-2025'

# === FOLDERS ===
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

# === DATABASE ===
def get_db():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        email TEXT UNIQUE,
        password_hash TEXT
    )''')
    return conn

# === AUTH ROUTES ===
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].lower()
        password = request.form['password']
        conn = get_db()
        user = conn.execute('SELECT id, password_hash FROM users WHERE email = ?', (email,)).fetchone()
        if user and check_password_hash(user[1], password):
            return jsonify({"success": True})
        return jsonify({"error": "Invalid credentials"}), 401
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    email = request.form['email'].lower()
    password = request.form['password']
    try:
        conn = get_db()
        conn.execute('INSERT INTO users (email, password_hash) VALUES (?, ?)',
                    (email, generate_password_hash(password)))
        conn.commit()
        return jsonify({"success": True})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email exists"}), 400

@app.route('/logout')
def logout():
    return jsonify({"success": True})

@app.route('/api/user')
def user_api():
    return jsonify({"is_authenticated": False})

# === MODEL DOWNLOAD ===
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
ESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
GFPGAN_PATH = "models/GFPGANv1.4.pth"
ESRGAN_PATH = "models/RealESRGAN_x2plus.pth"

def download_if_missing(url, path):
    if not os.path.exists(path):
        print(f"📥 Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        print("✅ Download complete.")

download_if_missing(GFPGAN_URL, GFPGAN_PATH)
download_if_missing(ESRGAN_URL, ESRGAN_PATH)

# === LOAD MODELS ===
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

gfpgan_restorer = GFPGANer(model_path=GFPGAN_PATH, upscale=2, bg_upsampler=None)
realesrgan_enhancer = RealESRGANer(
    scale=2,
    model_path=ESRGAN_PATH,
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
    tile=256,
    half=False
)

# === AI ENHANCEMENT ===
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    input_path = f"uploads/{filename}"
    output_path = f"results/enhanced_{filename}"
    
    # Resize large images to prevent OOM
    img_pil = Image.open(file.stream)
    if max(img_pil.size) > 1200:
        img_pil.thumbnail((1200, 1200), Image.LANCZOS)
    img_pil.save(input_path)
    
    # Process with AI
    img = cv2.imread(input_path)
    _, _, restored = gfpgan_restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    upscaled, _ = realesrgan_enhancer.enhance(restored)
    cv2.imwrite(output_path, upscaled)
    
    return jsonify({
        "success": True,
        "before_url": f"/uploads/{filename}",
        "after_url": f"/results/enhanced_{filename}"
    })

# === STATIC FILES ===
@app.route('/uploads/<path>')
def uploads(path):
    return send_from_directory('uploads', path)

@app.route('/results/<path>')
def results(path):
    return send_from_directory('results', path)

# === MAIN PAGE ===
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
