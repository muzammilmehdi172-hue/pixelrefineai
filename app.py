from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)

# Allow uploads up to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def home():
    return render_template('index.html')  # ← Serves templates/index.html

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Load and process image
        img = Image.open(file.stream).convert('RGB')
        img_np = np.array(img)

        # 🔧 YOUR AI MODEL HERE (replace this placeholder)
        enhanced = cv2.detailEnhance(img_np, sigma_s=10, sigma_r=0.15)
        enhanced_img = Image.fromarray(enhanced)

        # Save to static folder so it can be served
        os.makedirs('static', exist_ok=True)
        output_path = "static/output.jpg"
        enhanced_img.save(output_path)

        return jsonify({"result": "/static/output.jpg"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve static files (like output.jpg)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)