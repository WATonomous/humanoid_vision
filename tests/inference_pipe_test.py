from humanoid_vision.inference_pipeline import InferencePipeline
from humanoid_vision.training_pipeline import TrainingPipeline
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

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

# Test inference pipeline with saved model
print("\n--- Testing Inference Pipeline ---")
inference_pipeline = InferencePipeline(model=inference_model, data=data, batch_size=10, device="cpu", model_path=model_path)

# Test evaluate method
print("\nEvaluating model on test data...")
eval_loss = inference_pipeline.evaluate()
print(f"Evaluation loss: {eval_loss}")

# Test predict_batch method
print("\nMaking batch predictions...")
predictions = inference_pipeline.predict_batch()
print(f"Number of prediction batches: {len(predictions)}")
print(f"First batch shape: {predictions[0].shape}")

# Test predict method on single batch
print("\nMaking prediction on single input...")
single_input = X[:5]  # Take first 5 samples
single_prediction = inference_pipeline.predict(single_input)
print(f"Single prediction shape: {single_prediction.shape}")
print(f"Predictions: {single_prediction.flatten()}")

print("\n--- Inference Pipeline Test Complete ---")
