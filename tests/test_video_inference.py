import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from humanoid_vision.inference_pipeline import InferencePipeline
from humanoid_vision.training_pipeline import TrainingPipeline
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

# --- Create Inference Model ---
# Create model for testing
model = nn.Sequential(
    nn.Linear(3, 1),
    nn.Sigmoid()
)

# Create test data
X = torch.randn(100, 3)
y = torch.randint(0, 2, (100, 1)).float()

data = TensorDataset(X, y)

# Train the model first
print("Training model...")
training_pipeline = TrainingPipeline(model=model, batch_size=10, data=data, device="cpu", epochs=10)
training_pipeline.training()

# Save trained model
model_path = "test_model.pth"
training_pipeline.save(model_path)
print(f"\nModel saved to {model_path}")

# Create a new model for inference
inference_model = nn.Sequential(
    nn.Linear(3, 1),
    nn.Sigmoid()
)


# --- Test video function ---

def mock_capture_with_n_frames(n=1):
    """Mock object that simulates a cv2.VideoCapture with n frames."""
    mock_cap = MagicMock()

    # Simulate isOpened() always successful
    mock_cap.isOpened.return_value = True # always set to true for successful test

    # Prepare N fake frames as numpy arrays BGR
    frames = [
        np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(n)
    ]

    # Simulate fake frames: read() returns (True, frame) N times, then (False, None)
    mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]

    mock_cap.release.return_value = None
    return mock_cap

# unit test that video function works
def test_predict_video():
    pipeline = InferencePipeline(model=inference_model, data=data, batch_size=10, device="cpu", model_path=model_path)
    
    pipeline.predict = MagicMock(return_value=torch.tensor([5.0])) # fake model

    with patch("humanoid_vision.inference_pipeline.cv2.VideoCapture") as mock_capture:
        mock_capture.return_value = mock_capture_with_n_frames(4)
        preds = pipeline.predict_video("fake_video.mp4") # prediction array of tensors stored in preds

    # --- Tests ---
    # Should return 4 predictions
    assert len(preds) == 4

    # Each prediction is a CPU tensor
    assert torch.equal(preds[0], torch.tensor([5.0]))

    # pipeline.predict() should have been called 4 times
    assert pipeline.predict.call_count == 4
