import torch
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image

# Load the pretrained model and custom weights
model = fasterrcnn_resnet50_fpn(
    pretrained=False, num_classes=2
)  # Set num_classes based on your model
model_weights_path = "simple_rcnn_weights.pth"
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device("cpu")))
model.eval()  # Set model to evaluation mode

# Define image transformation
transform = T.Compose(
    [
        T.ToTensor(),  # Convert image to tensor
    ]
)


# Function to draw bounding boxes
def draw_bounding_box(image, boxes, labels=None, scores=None, threshold=0.5):
    # Convert image to BGR for OpenCV
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < threshold:
            continue  # Skip low-confidence boxes
        x_min, y_min, x_max, y_max = box.int()  # Convert box to integer
        color = (255, 0, 0)  # Color for bounding box (Red)

        # Draw the bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=2)

        # Add label and score if provided
        if labels:
            label_text = f"{labels[i]}: {scores[i]:.2f}" if scores else f"{labels[i]}"
            cv2.putText(
                img,
                label_text,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    # Convert back to RGB for plotting
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# Function to make predictions and visualize results
def predict_and_visualize(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        predictions = model(img_tensor)

    # Extract bounding boxes, labels, and scores
    boxes = predictions[0]["boxes"]
    labels = predictions[0]["labels"]
    scores = predictions[0]["scores"]

    # Draw bounding boxes on the image
    draw_bounding_box(img, boxes, labels, scores)


# Run prediction and visualization on an input image
image_path = "./test_image.jpg"  # Replace with your image path
predict_and_visualize(image_path)
