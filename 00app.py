from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import os
from PIL import Image
import io
import uuid
import shutil

app = Flask(__name__)

# Load the YOLO model
model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the uploaded image
        unique_filename = str(uuid.uuid4()) + ".jpg"
        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)

        # Read and save the image
        image = Image.open(io.BytesIO(file.read()))
        image.save(image_path)

        # Run prediction and save results
        results = model.predict(image_path, save=True)

        # Locate the processed image path
        processed_image_path = os.path.join(results[0].save_dir, unique_filename)
        output_image_path = os.path.join(OUTPUT_FOLDER, unique_filename)

        # Move the processed image to OUTPUT_FOLDER
        shutil.move(str(processed_image_path), output_image_path)

        # Send back the processed image
        return send_file(output_image_path, mimetype="image/jpeg"), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "Fabric Defect Detection API"


if __name__ == "__main__":
    app.run(debug=True)
