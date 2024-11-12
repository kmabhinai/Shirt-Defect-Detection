from flask import Flask, request, jsonify, send_file, render_template_string
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

# HTML template for the UI
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fabric Defect Detection</title>
    <style>
        body { display: flex; justify-content: center; align-items: center; min-height: 100vh; background-color: #f2f2f2; font-family: Arial, sans-serif; }
        .container { text-align: center; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); max-width: 400px; }
        h2 { color: #333; margin-bottom: 20px; }
        .button { padding: 10px 20px; color: white; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        .upload-btn { background-color: #4caf50; }
        .submit-btn { background-color: #007bff; }
        .output-image { margin-top: 20px; max-width: 100%; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Fabric Defect Detection</h2>
        <input type="file" id="file-input" accept="image/*" onchange="previewImage(event)">
        <br><br>
        <button class="button submit-btn" onclick="submitImage()">Submit</button>
        <div id="output" class="output-image" style="display: none;">
            <h3>Processed Image:</h3>
            <img id="output-image" src="" alt="Output Image" style="max-width: 100%; height: auto; border-radius: 8px;">
        </div>
    </div>
    
    <script>
        let selectedFile;
        
        function previewImage(event) {
            selectedFile = event.target.files[0];
        }

        async function submitImage() {
            if (!selectedFile) {
                alert("Please select an image to upload.");
                return;
            }

            const formData = new FormData();
            formData.append("file", selectedFile);

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });
                
                if (!response.ok) throw new Error("Image processing failed.");

                const blob = await response.blob();
                const outputImageUrl = URL.createObjectURL(blob);
                
                document.getElementById("output-image").src = outputImageUrl;
                document.getElementById("output").style.display = "block";
            } catch (error) {
                console.error("Error:", error);
                alert("An error occurred while processing the image.");
            }
        }
    </script>
</body>
</html>
"""


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

        # Save the image from the request
        image = Image.open(io.BytesIO(file.read()))
        image.save(image_path)

        # Run the YOLO model for prediction
        results = model.predict(image_path, save=True)

        # Locate and move the processed image
        processed_image_path = os.path.join(results[0].save_dir, unique_filename)
        output_image_path = os.path.join(OUTPUT_FOLDER, unique_filename)
        shutil.move(str(processed_image_path), output_image_path)

        # Send back the processed image
        return send_file(output_image_path, mimetype="image/jpeg"), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return render_template_string(html_template)


if __name__ == "__main__":
    app.run(debug=True)
