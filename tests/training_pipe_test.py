from humanoid_vision.training_pipeline import TrainingPipeline
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

model = nn.Sequential(
    nn.Linear(3,1),
    nn.Sigmoid()
)

# Create test data
X = torch.randn(100,3)
y = torch.randint(0, 2, (100, 1)).float()
test_data = TensorDataset(X,y)

data = TensorDataset(X,y)

pipeline = TrainingPipeline(model=model, batch_size=10, data=data, device="cpu")
pipeline.training()
