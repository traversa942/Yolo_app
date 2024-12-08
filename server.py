from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import os

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace with yolov8s.pt, yolov8m.pt, etc., for different sizes

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Check if an image file is in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        # Load the image
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')

        # Perform inference
        results = model(image)

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image)
        for box in results[0].boxes:
            xyxy = box.xyxy[0]  # Bounding box coordinates
            cls = int(box.cls)  # Class index
            conf = box.conf     # Confidence score

            # Draw the bounding box and label
            draw.rectangle(xyxy.tolist(), outline="red", width=3)
            label = f"{model.names[cls]}: {conf:.2f}"
            draw.text((xyxy[0], xyxy[1] - 10), label, fill="red")

        # Save the image to a BytesIO object
        img_io = io.BytesIO()
        image.save(img_io, format="JPEG")
        img_io.seek(0)

        # Send the image back to the client
        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
