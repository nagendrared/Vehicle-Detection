"""
Vehicle Detection Flask Backend
YOLO (Ultralytics) + best.pt model
"""

import os
import base64
import logging
import time

import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Flask Setup
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_PATH = "yolov11_model.pt"
INPUT_SIZE = 640
CONF_THRESH = 0.25

# 🔥 CHANGED FOLDERS
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Bounding Box Styling
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 6
TEXT_THICKNESS = 2
FONT_SCALE = 0.8

# Create folders if not exist
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model = YOLO(MODEL_PATH)

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_detection(image_path, output_path, conf_thresh, max_results, selected_classes):
    start_time = time.time()

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image file")

    orig_h, orig_w = image.shape[:2]

    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    results = model.predict(source=resized, conf=conf_thresh, verbose=False)
    result = results[0]

    sx = orig_w / INPUT_SIZE
    sy = orig_h / INPUT_SIZE

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]

            if selected_classes and cls_name not in selected_classes:
                continue

            x1r, y1r, x2r, y2r = box.xyxy[0].tolist()

            # Scale boxes back
            x1 = int(x1r * sx)
            y1 = int(y1r * sy)
            x2 = int(x2r * sx)
            y2 = int(y2r * sy)

            detections.append({
                "label": cls_name,
                "score": round(conf, 3),
                "box": [x1, y1, x2, y2],
                "dimensions": {
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "area": (x2 - x1) * (y2 - y1)
                }
            })

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

            label = f"{cls_name} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                TEXT_THICKNESS
            )

            cv2.rectangle(
                image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                BOX_COLOR,
                -1
            )

            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (0, 0, 0),
                TEXT_THICKNESS
            )

    detections = detections[:max_results]

    # 🔥 SAVE OUTPUT IMAGE
    cv2.imwrite(output_path, image)

    # Encode base64 (optional, same as before)
    _, buffer = cv2.imencode(".jpg", image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "success": True,
        "count": len(detections),
        "detections": detections,
        "image_base64": image_base64,
        "output_path": output_path,   # 🔥 Added
        "processing_time": round(time.time() - start_time, 3)
    }


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)

    # 🔥 INPUT PATH
    input_path = os.path.join(INPUT_FOLDER, filename)

    # 🔥 OUTPUT PATH
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    # Save original image
    file.save(input_path)

    # Params
    conf = float(request.form.get("min_confidence", CONF_THRESH))
    max_results = int(request.form.get("max_results", 10))
    classes = request.form.get("classes", "")
    selected_classes = classes.split(",") if classes else []

    try:
        result = run_detection(
            input_path,
            output_path,
            conf,
            max_results,
            selected_classes
        )
        return jsonify(result)

    except Exception as e:
        logger.error(e)
        return jsonify({"success": False, "error": str(e)}), 500


# --------------------------------------------------
# Run App
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)