from humanoid_vision.training_pipeline import TrainingPipeline
#from training_pipeline import TrainingPipeline
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

model = nn.Sequential(
    nn.Linear(3,1),
    nn.Sigmoid()
)

X = torch.randn(100,3)
print(X)
y = torch.randint(0, 2, (100, 1)).float()

data = TensorDataset(X,y)

pipeline = TrainingPipeline(model=model, batch_size=10, data=data, device="cpu")
pipeline.training()