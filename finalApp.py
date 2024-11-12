from flask import Flask, request, jsonify, send_file, render_template_string
from ultralytics import YOLO
import torch
import torch.nn as nn
import cv2
import io
import os
import uuid
import shutil
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__)

# Load the YOLO model
yolo_model = YOLO("best.pt")


# Load the RCNN model
class SimpleRCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleRCNN, self).__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)


def load_rcnn_model(weight_path, num_classes):
    model = SimpleRCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img


def check_defect_rcnn(model, image_tensor):
    with torch.no_grad():
        output = torch.sigmoid(model(image_tensor))
        defect_present = output > 0.5  # Threshold at 0.5
        return defect_present.any()


# Load RCNN model weights
rcnn_model = load_rcnn_model("simple_rcnn_weights.pth", num_classes=3)

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
        <label for="model-select">Select Model:</label>
        <select id="model-select">
            <option value="yolo">YOLO</option>
            <option value="rcnn">RCNN</option>
        </select>
        <br><br>
        <button class="button submit-btn" onclick="submitImage()">Submit</button>
        <div id="output" class="output-image" style="display: none;">
            <h3 id="output-text"></h3>
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
            formData.append("model", document.getElementById("model-select").value);

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });
                
                if (!response.ok) throw new Error("Image processing failed.");

                const data = await response.json();
                if (data.model === "yolo") {
                    document.getElementById("output-image").src = data.image_url;
                    document.getElementById("output-text").innerText = "Processed Image:";
                    document.getElementById("output-image").style.display = "block";
                } else if (data.model === "rcnn") {
                    document.getElementById("output-text").innerText = data.message;
                    document.getElementById("output-image").style.display = "none";
                }
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

    model_type = request.form.get("model", "yolo")

    try:
        unique_filename = str(uuid.uuid4()) + ".jpg"
        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)

        # Save the image from the request
        image = Image.open(io.BytesIO(file.read()))
        image.save(image_path)

        if model_type == "yolo":
            # Run YOLO model for prediction
            results = yolo_model.predict(image_path, save=True)
            processed_image_path = os.path.join(results[0].save_dir, unique_filename)
            output_image_path = os.path.join(OUTPUT_FOLDER, unique_filename)
            shutil.move(str(processed_image_path), output_image_path)
            return jsonify({"model": "yolo", "image_url": "/output/" + unique_filename})

        elif model_type == "rcnn":
            # Run RCNN model for defect detection
            image_tensor = preprocess_image(image_path)
            defect_present = check_defect_rcnn(rcnn_model, image_tensor)
            message = (
                "Defect detected in the image."
                if defect_present
                else "No defect detected in the image."
            )
            return jsonify({"model": "rcnn", "message": message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/output/<filename>")
def output_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype="image/jpeg")


@app.route("/")
def home():
    return render_template_string(html_template)


if __name__ == "__main__":
    app.run(debug=True)
