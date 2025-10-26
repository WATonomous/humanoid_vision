from src.humanoid_vision.training_pipeline import TrainingPipeline
#from training_pipeline import TrainingPipeline
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

testModel = nn.Sequential(
    nn.Linear(3,1),
    nn.Sigmoid()
)

X = torch.randn(100,3)
print(X)
y = torch.randint(0, 2, (100, 1)).float()

testData = TensorDataset(X,y)

pipeline = TrainingPipeline(model = testModel,batch_size = 10,data = testData, device = "cpu")
pipeline.training()