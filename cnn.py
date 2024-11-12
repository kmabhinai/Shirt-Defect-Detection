import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms


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


def check_defect(model, image_tensor):
    with torch.no_grad():
        output = torch.sigmoid(model(image_tensor))
        defect_present = output > 0.5  # Threshold at 0.5
        return defect_present


def main():
    # Paths and parameters
    weight_path = "simple_rcnn_weights.pth"
    image_path = "./20231106_212651_jpg.rf.66bdad807bb58a08792751623c915408.jpg"
    num_classes = 3  # Use the correct number of classes as per your trained model

    # Load model and image
    model = load_model(weight_path, num_classes)
    image_tensor = preprocess_image(image_path)

    # Check for defects
    defect_present = check_defect(model, image_tensor)
    if defect_present.any():
        print("Defect detected in the image.")
    else:
        print("No defect detected in the image.")


if __name__ == "__main__":
    main()
