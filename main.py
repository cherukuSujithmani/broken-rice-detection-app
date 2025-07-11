import os
import pathlib
from flask import Flask, render_template, request
from PIL import Image
import torch
import numpy as np

# Fix PosixPath error on Windows (important)
pathlib.PosixPath = pathlib.WindowsPath

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.conf = 0.25  # confidence threshold

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files.get('image')
    if not image_file:
        return 'No file uploaded!', 400

    # Paths
    input_path = os.path.join(UPLOAD_FOLDER, 'input.jpg')
    result_path = os.path.join(RESULT_FOLDER, 'image0.jpg')

    # Clear previous input/output
    if os.path.exists(input_path):
        os.remove(input_path)
    if os.path.exists(result_path):
        os.remove(result_path)

    # Save uploaded file
    image_file.save(input_path)

    # Run detection
    results = model(input_path)
    results.render()  # Draw boxes

    # Save result image
    output_img = Image.fromarray(results.ims[0])
    output_img.convert("RGB").save(result_path)

    return render_template('result.html', image_path='/static/results/image0.jpg')

if __name__ == '__main__':
    app.run(debug=True) 