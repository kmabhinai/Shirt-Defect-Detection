import torch
import torch.nn as nn
from torchvision import models


class SimpleRCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleRCNN, self).__init__()

        # Using ResNet-50 as the backbone
        self.backbone = models.resnet50(pretrained=True)

        # Remove the final fully connected layer
        self.backbone.fc = (
            nn.Identity()
        )  # Use Identity so that output is just the features

        # Add a new fully connected layer for classification
        self.fc = nn.Linear(
            2048, num_classes
        )  # 2048 is the output size of ResNet-50's feature extractor

    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone(x)

        # Pass the features through the fully connected layer
        return self.fc(features)
