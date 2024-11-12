from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import os
from PIL import Image
import io
import uuid

app = Flask(__name__)

model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def preprocess_image(image):
    # Resize the image to YOLO's expected input size
    processed_image = image.resize(
        (640, 640)
    )  # Resize to 640x640, or the input size expected by YOLO

    # Convert image to RGB if it’s in another mode, as YOLO expects RGB images
    if processed_image.mode != "RGB":
        processed_image = processed_image.convert("RGB")

    return processed_image


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Generate a unique filename and save the original image
        unique_filename = str(uuid.uuid4()) + ".jpg"
        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)

        # Open and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        preprocessed_image = preprocess_image(image)
        preprocessed_image.save(image_path)

        # Run the YOLO model for prediction on the preprocessed image
        results = model.predict(image_path, save=True)

        # Extract predictions
        predictions = []
        for box in results[0].boxes:
            prediction = {
                "confidence": box.conf[0].item(),
                "label": int(box.cls[0].item()),
                "coordinates": box.xywh[0].tolist(),
            }
            predictions.append(prediction)

        return (
            jsonify(
                {
                    "predictions": predictions,
                    "image_url": f"/runs/detect/predict/{unique_filename}",
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/")
def home():
    return "Fabric Defect Detection API"


if __name__ == "__main__":
    app.run(debug=True)
