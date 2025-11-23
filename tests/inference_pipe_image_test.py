import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from humanoid_vision.inference_pipeline import InferencePipeline
from humanoid_vision.training_pipeline import TrainingPipeline
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset


# Simple CNN for basic object detection (binary classification: object present or not)
class SimpleObjectDetector(nn.Module):
    def __init__(self):
        super(SimpleObjectDetector, self).__init__()
        # Input: 28x28 grayscale images (flattened to 784)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 28x28x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 28x28x32
        self.pool = nn.MaxPool2d(2, 2)  # Reduces to 14x14
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 14 * 14)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


print("=== Basic Object Detection Test ===\n")

# Create synthetic image data (28x28 images)
# Simulating images with and without objects
num_samples = 200
image_size = 28

# Generate random "images" - in reality these would be actual image data
X = torch.randn(num_samples, 1, image_size, image_size)

# Generate labels: 1 = object detected, 0 = no object
# For simplicity, we'll create a pattern: images with high mean intensity have objects
y = (X.mean(dim=(1, 2, 3)) > 0).float().unsqueeze(1)

print(f"Generated {num_samples} synthetic {image_size}x{image_size} images")
print(f"Objects detected in {y.sum().item():.0f} images")
print(f"No objects in {(num_samples - y.sum()).item():.0f} images\n")

# Create dataset
image_data = TensorDataset(X, y)

# Initialize model
detector_model = SimpleObjectDetector()

print("Training object detection model...")
training_pipeline = TrainingPipeline(
    model=detector_model,
    data=image_data,
    batch_size=16,
    lr=0.001,
    epochs=20,
    device="cpu",
)
training_pipeline.training()

# Save the trained model
model_path = "object_detector.pth"
training_pipeline.save(model_path)
print(f"\nModel saved to {model_path}")

# Create a new model instance for inference
inference_detector = SimpleObjectDetector()

print("\n=== Testing Inference Pipeline ===")
inference_pipeline = InferencePipeline(
    model=inference_detector,
    data=image_data,
    batch_size=16,
    device="cpu",
    model_path=model_path,
)

# Evaluate the model
print("\nEvaluating model on test data...")
eval_loss = inference_pipeline.evaluate()
print(f"Evaluation loss: {eval_loss:.4f}")

# Test prediction on individual images
print("\n=== Testing Individual Image Predictions ===")
test_images = X[:5]  # Take first 5 images
test_labels = y[:5]

predictions = inference_pipeline.predict(test_images)

print("\nPredictions vs Ground Truth:")
for i, (pred, label) in enumerate(zip(predictions, test_labels)):
    pred_class = "OBJECT DETECTED" if pred.item() > 0.5 else "NO OBJECT"
    true_class = "OBJECT DETECTED" if label.item() > 0.5 else "NO OBJECT"
    confidence = pred.item() if pred.item() > 0.5 else (1 - pred.item())
    print(f"Image {i+1}: Predicted: {pred_class} (confidence: {confidence:.2%}) | True: {true_class}")

# Test batch predictions
print("\n=== Testing Batch Predictions ===")
all_predictions = inference_pipeline.predict_batch()
print(f"Generated predictions for {len(all_predictions)} batches")

# Calculate accuracy
total_correct = 0
total_samples = 0
for batch_preds, (batch_images, batch_labels) in zip(all_predictions, inference_pipeline.data_loader):
    batch_preds_binary = (batch_preds > 0.5).float()
    total_correct += (batch_preds_binary == batch_labels).sum().item()
    total_samples += batch_labels.size(0)

accuracy = total_correct / total_samples
print(f"Overall Accuracy: {accuracy:.2%}")

print("\n=== Object Detection Test Complete ===")
