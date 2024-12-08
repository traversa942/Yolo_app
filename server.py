from flask import Flask, request, jsonify
from PIL import Image
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

app = Flask(__name__)

# Load the YOLO model
device = select_device('')  # Automatically select CPU or GPU
model = DetectMultiBackend('yolov5s.pt', device=device)
stride, names, pt = model.stride, model.names, model.pt
img_size = check_img_size(640, s=stride)  # Ensure image size is valid

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Check if image file is in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        # Load the image
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')

        # Preprocess the image for YOLO
        img = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img = img.to(device)

        # Perform inference
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

        # Process results
        results = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.size).round()
                for *xyxy, conf, cls in det:
                    results.append({
                        "label": names[int(cls)],
                        "confidence": float(conf),
                        "x_min": int(xyxy[0]),
                        "y_min": int(xyxy[1]),
                        "x_max": int(xyxy[2]),
                        "y_max": int(xyxy[3]),
                    })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
