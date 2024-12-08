from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = Flask(_name_)

# Load the YOLOv8 pre-trained model
model = YOLO("yolov8m.pt")  # Use 'yolov8s.pt', 'yolov8m.pt', etc., for larger models

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Check if an image is included in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        # Load the image from the request
        image_file = request.files['image']
        image = Image.open(image_file.stream)

        # Convert the image to a NumPy array for YOLO inference
        results = model(image)

        # Format results into a JSON response
        detections = []
        for box in results[0].boxes:
            detections.append({
                "label": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "x_min": int(box.xyxy[0][0]),
                "y_min": int(box.xyxy[0][1]),
                "x_max": int(box.xyxy[0][2]),
                "y_max": int(box.xyxy[0][3]),
            })

        return jsonify(detections)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if _name_ == '_main_':
    app.run(debug=True, host="0.0.0.0", port=5000)
