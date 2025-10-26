from humanoid_vision.training_pipeline import TrainingPipeline
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

def test_training_pipeline():
    # Create a simple model for testing
    model = nn.Sequential(
        nn.Linear(3,1),
        nn.Sigmoid()
    )

    # Create test data
    X = torch.randn(100,3)
    y = torch.randint(0, 2, (100, 1)).float()
    test_data = TensorDataset(X,y)

    # Initialize and run the pipeline
    pipeline = TrainingPipeline(model=model, batch_size=10, data=test_data, device="cpu")
    pipeline.training()

    # Add assertions to verify the training worked
    assert isinstance(pipeline.model, nn.Sequential), "Model should be a Sequential model"
    assert len(pipeline.data) == 100, "Dataset should have 100 samples"