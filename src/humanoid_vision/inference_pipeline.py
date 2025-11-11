import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class InferencePipeline():
    def __init__(self, model, data=None, batch_size=32, criterion=nn.BCELoss, device=None, model_path=None):
        '''
        Initialize the Inference Pipeline
        Args:
            :param model: PASS IN MODEL CLASS.
            :param data: pass into data in tensor form (optional for inference).
            :param batch_size: batch size for inference.
            :param device: The device model and data will be passed into (ie cpu or cuda).
            :param criterion: PASS IN LOSS FUNCTION CLASS (optional, for evaluation).
            :param model_path: Path to load trained model weights from (optional).
        '''
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.criterion = criterion()
        
        # Load model weights if path is provided
        if model_path:
            self.load_model(model_path)
        
        # Create DataLoader if data is provided
        if data is not None:
            self.data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        else:
            self.data_loader = None
        
        # Set model to evaluation mode
        self.model.eval()
    
    def load_model(self, path):
        '''
        Load model weights from a file.
        Args:
            :param path: Path to the model weights file
        '''
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model weights loaded successfully from {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {path}: {e}")
    
    def predict(self, input_data):
        '''
        Make predictions on input data
        Args:
            :param input_data: Input tensor
        Returns:
            torch.Tensor: Model predictions
        '''
        self.model.eval()
        input_data = input_data.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(input_data)
        
        return predictions
    
    def predict_batch(self):
        '''
        Make predictions on all data in the data loader
        Returns:
            list: List of prediction tensors
        '''
        if self.data_loader is None:
            raise ValueError("No data provided. Initialize with data or use predict() method.")
        
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for xb, yb in self.data_loader:
                xb = xb.to(self.device)
                preds = self.model(xb)
                all_predictions.append(preds.cpu())
        
        return all_predictions
    
    def evaluate(self):
        '''
        Evaluates the model using the provided data
        Returns:
            float: average loss for the data
        '''
        if self.data_loader is None:
            raise ValueError("No data provided for evaluation.")
        
        self.model.eval()
        total_loss = 0
        
        with torch.inference_mode():
            for xb, yb in self.data_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                total_loss += loss.item()
        
        return total_loss / len(self.data_loader)