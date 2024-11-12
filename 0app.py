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


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:

        unique_filename = str(uuid.uuid4()) + ".jpg"
        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)

        image = Image.open(io.BytesIO(file.read()))
        image.save(image_path)

        results = model.predict(image_path, save=True)

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
