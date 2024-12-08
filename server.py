from flask import Flask, request, jsonify
from PIL import Image
import torch
import os

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can use yolov5m, yolov5l for better accuracy

# Initialize Flask
app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_objects():
    # Receive image from FlutterFlow
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    image_file = request.files['image']
    image = Image.open(image_file.stream)

    # Perform inference
    results = model(image)

    # Format results as JSON
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    return jsonify(detections)

if __name__ == '_main_':
    app.run(debug=True, host='0.0.0.0', port=5000)
