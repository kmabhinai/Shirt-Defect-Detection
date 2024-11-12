import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import matplotlib.pyplot as plt

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleRCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleRCNN, self).__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)


def load_model(weight_path, num_classes):
    """
    Load the trained model weights.
    """
    model = SimpleRCNN(num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    # model.to(device)
    model.eval()
    return model


def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box on the image.
    - bbox: (x, y, w, h) coordinates of the bounding box
    - color: color of the bounding box (default green)
    - thickness: thickness of the bounding box lines
    """
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image


def process_image_for_defect(model, image_path, output_path="output_folder"):
    """
    Processes the input image, detects defects, and returns the image with bounding box drawn.
    """
    # Load the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Preprocess the image
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(img_rgb).unsqueeze(0).to(device)  # Add batch dimension

    # Predict the defect
    with torch.no_grad():
        output = torch.sigmoid(model(img_tensor)).cpu().numpy().flatten()

    # If defect is detected (assuming binary classification with defect in index 1)
    if output[1] > 0.5:  # Adjust threshold as needed
        print("Defect detected")

        # Dummy bounding box coordinates, replace with model's actual bounding box prediction if available
        bbox = (50, 50, 100, 100)  # (x, y, width, height)

        # Draw the bounding box on the image
        img_with_bbox = draw_bounding_box(img_rgb, bbox)

        # Convert back to BGR for saving
        img_with_bbox = cv2.cvtColor(img_with_bbox, cv2.COLOR_RGB2BGR)

        # Save the output image
        os.makedirs(output_path, exist_ok=True)
        output_image_path = os.path.join(output_path, os.path.basename(image_path))
        cv2.imwrite(output_image_path, img_with_bbox)

        print(f"Image with bounding box saved to {output_image_path}")

    else:
        print("No defect detected")


def main():
    # Define paths
    weight_path = (
        "./simple_rcnn_weights.pth"  # Replace with actual path to the weights file
    )
    image_path = "./20231106_212651_jpg.rf.66bdad807bb58a08792751623c915408.jpg"  # Replace with the actual path to the input image

    # Load model
    model = load_model(weight_path, num_classes=3)  # Ensure the correct class count

    # Process image and save result
    process_image_for_defect(model, image_path)


if __name__ == "__main__":
    main()
