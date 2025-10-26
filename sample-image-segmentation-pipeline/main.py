#!/usr/bin/env python3
"""
Image Segmentation Training Pipeline
====================================

A comprehensive pipeline for training semantic segmentation models on PASCAL VOC dataset.
This pipeline includes data handling, model architecture, training, validation, and testing.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import random
import math
import copy
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as tF
import torchvision.models as models

# Import custom VOC dataset
from lib.voc import VOCSegmentation

# Import evaluation metrics
try:
    from chainercv.evaluations import eval_semantic_segmentation
except ImportError:
    print("Warning: chainercv not available. Some evaluation metrics may not work.")
    eval_semantic_segmentation = None


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for the training pipeline."""
    
    def __init__(self):
        # Dataset configuration
        self.dataset_path = 'datasets'
        self.num_classes = 21  # VOC has 20 classes + background
        self.ignore_index = 255  # Void class
        
        # Model configuration
        self.backbone = 'resnet18'  # or 'resnet34', 'resnet50'
        self.pretrained = True
        
        # Training configuration
        self.batch_size = 4
        self.num_epochs = 50
        self.learning_rate = 0.006
        self.momentum = 0.9
        self.weight_decay = 1e-4
        
        # Data augmentation
        self.image_size = (375, 500)
        self.crop_size = 224
        self.min_scale = 0.9
        self.max_scale = 1.1
        
        # Hardware configuration
        self.use_gpu = torch.cuda.is_available()
        self.num_workers = 0 if os.name == 'nt' else 4  # Windows compatibility
        
        # Training phases
        self.sanity_check_epochs = 5
        self.overfit_epochs = 40
        
        # Output configuration
        self.output_dir = 'outputs'
        self.save_interval = 10
        self.log_interval = 10
        
        # Normalization values (ImageNet)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


# ============================================================================
# DATA TRANSFORMS
# ============================================================================

class JointToTensor:
    """Convert PIL images to tensors."""
    def __call__(self, img, target):
        return tF.to_tensor(img), torch.from_numpy(np.array(target.convert('P'), dtype=np.int32)).long()


class JointCenterCrop:
    """Center crop for both image and target."""
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img, target):
        return (tF.center_crop(img, self.size), 
                tF.center_crop(target, self.size))


class RandomFlip:
    """Random horizontal flip."""
    def __call__(self, img, target):
        if torch.rand(1) > 0.5:
            return (tF.hflip(img), tF.hflip(target))
        else:
            return (img, target)


class RandomResizeCrop:
    """Random resize and crop for data augmentation."""
    def __init__(self, min_scale=0.9, max_scale=1.1, size=224):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.size = size
        
    def __call__(self, img, target):
        scale = torch.rand(1) * (self.max_scale - self.min_scale) + self.min_scale
        w, h = img.size
        new_size = (int(h * scale), int(w * scale))
        
        resized_img = tF.resize(img, new_size)
        resized_target = tF.resize(target, new_size)
        
        crop_size = min(self.size, new_size[0], new_size[1])
        x_crop = random.randint(0, new_size[1] - crop_size)
        y_crop = random.randint(0, new_size[0] - crop_size)
        
        cropped_img = tF.crop(resized_img, y_crop, x_crop, crop_size, crop_size)
        cropped_target = tF.crop(resized_target, y_crop, x_crop, crop_size, crop_size)
        
        final_img = tF.resize(cropped_img, (self.size, self.size))
        final_target = tF.resize(cropped_target, (self.size, self.size))
        
        return (final_img, final_target)


class Normalize:
    """Normalize image with mean and std."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor, target):
        return (tF.normalize(tensor, self.mean, self.std), target)


class JointCompose:
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        assert img.size == target.size
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SegmentationNet(nn.Module):
    """
    Encoder-Decoder segmentation network with skip connections.
    
    Architecture:
    - Encoder: Pre-trained ResNet backbone
    - Decoder: Transpose convolutions with skip connections
    - Skip connection from early encoder features
    """
    
    def __init__(self, num_classes=21, backbone='resnet18', pretrained=True, criterion=None):
        super(SegmentationNet, self).__init__()
        
        self.num_classes = num_classes
        self.criterion = criterion
        
        # Load backbone
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Extract encoder layers
        self.encoder_early = nn.Sequential(*list(resnet.children())[:5])  # First 4 layers
        self.encoder_late = nn.Sequential(*list(resnet.children())[5:8])   # Last 3 layers
        
        # Get feature dimensions
        if backbone == 'resnet18':
            early_features = 64
            late_features = 512
        elif backbone == 'resnet34':
            early_features = 64
            late_features = 512
        elif backbone == 'resnet50':
            early_features = 256
            late_features = 2048
        
        # Decoder layers
        self.decoder_conv1 = nn.ConvTranspose2d(early_features + late_features, 400, 5, 1)
        self.decoder_conv2 = nn.ConvTranspose2d(400, 512, 3, 1)
        self.decoder_conv3 = nn.ConvTranspose2d(512, 256, 3, 1)
        self.decoder_conv4 = nn.Conv2d(256, num_classes, 1, 1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)
        
    def forward(self, x, gts=None):
        # Encoder forward pass
        early_features = self.encoder_early(x)
        late_features = self.encoder_late(early_features)
        
        # Skip connection: upsample late features to match early features
        upsampled_late = F.interpolate(late_features, size=early_features.shape[-2:], 
                                    mode='bilinear', align_corners=False)
        
        # Concatenate skip connection
        combined_features = torch.cat([upsampled_late, early_features], dim=1)
        
        # Decoder forward pass
        x = F.relu(self.decoder_conv1(combined_features))
        x = F.relu(self.bn2(self.decoder_conv2(x)))
        x = F.relu(self.decoder_conv3(x))
        x = self.decoder_conv4(x)
        
        # Final upsampling to input size
        output = F.interpolate(x, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        if self.training and gts is not None:
            # Return loss during training
            return self.criterion(output, gts)
        else:
            # Return predictions during evaluation
            return output


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class CustomCrossEntropyLoss:
    """Custom implementation of Cross Entropy Loss for debugging."""
    
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index
        
    def __call__(self, logits, targets):
        # Use PyTorch's implementation for numerical stability
        return F.cross_entropy(logits, targets, ignore_index=self.ignore_index)


# ============================================================================
# DATASET HANDLING
# ============================================================================

class DatasetManager:
    """Manages dataset creation and data loaders."""
    
    def __init__(self, config: Config):
        self.config = config
        self.setup_transforms()
        
    def setup_transforms(self):
        """Setup data transforms for different phases."""
        
        # Sanity check transforms (minimal for overfitting)
        self.sanity_transforms = JointCompose([
            JointToTensor(),
            Normalize(self.config.mean, self.config.std),
        ])
        
        # Training transforms (with augmentation)
        self.train_transforms = JointCompose([
            RandomResizeCrop(self.config.min_scale, self.config.max_scale, self.config.crop_size),
            JointCenterCrop(self.config.crop_size),
            RandomFlip(),
            JointToTensor(),
            Normalize(self.config.mean, self.config.std),
        ])
        
        # Validation transforms (no augmentation)
        self.val_transforms = JointCompose([
            JointCenterCrop(self.config.crop_size),
            JointToTensor(),
            Normalize(self.config.mean, self.config.std),
        ])
    
    def create_datasets(self):
        """Create training, validation, and sanity check datasets."""
        
        # Sanity check dataset (single image for debugging)
        self.sanity_dataset = VOCSegmentation(
            self.config.dataset_path,
            image_set='train',
            transforms=self.sanity_transforms,
            sanity_check=200,  # Use image index 200
            download=True
        )
        
        # Training dataset
        self.train_dataset = VOCSegmentation(
            self.config.dataset_path,
            image_set='train',
            transforms=self.train_transforms,
            download=True
        )
        
        # Validation dataset
        self.val_dataset = VOCSegmentation(
            self.config.dataset_path,
            image_set='val',
            transforms=self.val_transforms,
            download=True
        )
        
        print(f"Created datasets:")
        print(f"  Sanity: {len(self.sanity_dataset)} samples")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples")
    
    def create_dataloaders(self):
        """Create data loaders."""
        
        self.sanity_loader = DataLoader(
            self.sanity_dataset,
            batch_size=1,
            num_workers=self.config.num_workers,
            shuffle=False
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.config.num_workers,
            shuffle=False
        )


# ============================================================================
# TRAINING AND VALIDATION
# ============================================================================

class Trainer:
    """Handles model training and validation."""
    
    def __init__(self, config: Config, model: nn.Module, dataloaders: Dict):
        self.config = config
        self.model = model
        self.dataloaders = dataloaders
        
        # Setup device
        self.device = torch.device('cuda' if config.use_gpu else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
        self.model.criterion = self.criterion
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.dataloaders['train'], desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            loss = self.model(inputs, gts=targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log interval
            if batch_idx % self.config.log_interval == 0:
                logging.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        mious = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.dataloaders['val'], desc='Validating'):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculate mIoU
                if eval_semantic_segmentation is not None:
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    gts = torch.where(targets == 255, -1, targets).cpu().numpy()
                    
                    for pred, gt in zip(preds, gts):
                        conf = eval_semantic_segmentation([pred], [gt])
                        mious.append(conf['miou'])
        
        avg_loss = total_loss / len(self.dataloaders['val'])
        avg_miou = np.mean(mious) if mious else 0.0
        
        self.val_losses.append(avg_loss)
        self.val_mious.append(avg_miou)
        
        return avg_loss, avg_miou
    
    def train(self):
        """Main training loop."""
        logging.info("Starting training...")
        
        best_miou = 0.0
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_miou = self.validate()
            
            # Logging
            logging.info(f'Epoch {epoch+1}/{self.config.num_epochs}:')
            logging.info(f'  Train Loss: {train_loss:.4f}')
            logging.info(f'  Val Loss: {val_loss:.4f}')
            logging.info(f'  Val mIoU: {val_miou:.4f}')
            
            # Save best model
            if val_miou > best_miou:
                best_miou = val_miou
                self.save_model(f'best_model_epoch_{epoch+1}.pth')
                logging.info(f'New best model saved with mIoU: {best_miou:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
        
        logging.info(f'Training completed. Best mIoU: {best_miou:.4f}')
    
    def save_model(self, filename: str):
        """Save model checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        filepath = os.path.join(self.config.output_dir, filename)
        
        torch.save({
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mious': self.val_mious,
            'config': self.config.__dict__
        }, filepath)
        
        logging.info(f'Model saved to {filepath}')


# ============================================================================
# TESTING AND EVALUATION
# ============================================================================

class Evaluator:
    """Handles model evaluation and testing."""
    
    def __init__(self, config: Config, model: nn.Module):
        self.config = config
        self.model = model
        self.device = torch.device('cuda' if config.use_gpu else 'cpu')
        self.model.to(self.device)
        
        # Color palette for visualization
        self.palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
                       128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
                       64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
    
    def colorize_mask(self, mask):
        """Convert segmentation mask to colored image."""
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(self.palette)
        return new_mask
    
    def evaluate_model(self, dataloader, model_name="Model"):
        """Evaluate model on a dataset."""
        self.model.eval()
        total_loss = 0.0
        mious = []
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc=f'Evaluating {model_name}'):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                if eval_semantic_segmentation is not None:
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    gts = torch.where(targets == 255, -1, targets).cpu().numpy()
                    
                    for pred, gt in zip(preds, gts):
                        conf = eval_semantic_segmentation([pred], [gt])
                        mious.append(conf['miou'])
        
        avg_loss = total_loss / len(dataloader)
        avg_miou = np.mean(mious) if mious else 0.0
        
        logging.info(f'{model_name} Evaluation:')
        logging.info(f'  Loss: {avg_loss:.4f}')
        logging.info(f'  mIoU: {avg_miou:.4f}')
        
        return avg_loss, avg_miou
    
    def visualize_predictions(self, dataloader, num_samples=5, save_path=None):
        """Visualize model predictions."""
        self.model.eval()
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                if i >= num_samples:
                    break
                
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()[0]
                
                # Convert to PIL images for visualization
                input_img = inputs[0].cpu().permute(1, 2, 0).numpy()
                input_img = (input_img * np.array(self.config.std) + np.array(self.config.mean))
                input_img = np.clip(input_img, 0, 1)
                
                target_img = targets[0].numpy()
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(input_img)
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                
                axes[1].imshow(self.colorize_mask(target_img))
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                axes[2].imshow(self.colorize_mask(preds))
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(os.path.join(save_path, f'prediction_{i}.png'))
                
                plt.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def setup_logging(config: Config):
    """Setup logging configuration."""
    os.makedirs(config.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.output_dir, 'training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main training pipeline."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Image Segmentation Training Pipeline')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'sanity'], 
                       default='train', help='Training mode')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Override config with command line arguments
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.gpu:
        config.use_gpu = True
    
    # Setup logging
    setup_logging(config)
    
    logging.info("Starting Image Segmentation Pipeline")
    logging.info(f"Configuration: {config.__dict__}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        # ========================================================================
        # DATASET SETUP
        # ========================================================================
        logging.info("Setting up datasets...")
        dataset_manager = DatasetManager(config)
        dataset_manager.create_datasets()
        dataset_manager.create_dataloaders()
        
        # ========================================================================
        # MODEL SETUP
        # ========================================================================
        logging.info("Creating model...")
        model = SegmentationNet(
            num_classes=config.num_classes,
            backbone=config.backbone,
            pretrained=config.pretrained
        )
        
        logging.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # ========================================================================
        # TRAINING PHASES
        # ========================================================================
        
        if args.mode == 'sanity':
            # Sanity check training (overfit on single image)
            logging.info("Starting sanity check training...")
            
            trainer = Trainer(config, model, {
                'train': dataset_manager.sanity_loader,
                'val': dataset_manager.sanity_loader
            })
            
            # Train for fewer epochs on single image
            original_epochs = config.num_epochs
            config.num_epochs = config.sanity_check_epochs
            
            trainer.train()
            
            # Evaluate sanity model
            evaluator = Evaluator(config, model)
            evaluator.evaluate_model(dataset_manager.sanity_loader, "Sanity Model")
            evaluator.visualize_predictions(dataset_manager.sanity_loader, save_path=config.output_dir)
            
        elif args.mode == 'train':
            # Full training
            logging.info("Starting full training...")
            
            trainer = Trainer(config, model, {
                'train': dataset_manager.train_loader,
                'val': dataset_manager.val_loader
            })
            
            trainer.train()
            
            # Final evaluation
            evaluator = Evaluator(config, model)
            evaluator.evaluate_model(dataset_manager.val_loader, "Final Model")
            evaluator.visualize_predictions(dataset_manager.val_loader, save_path=config.output_dir)
            
        elif args.mode == 'test':
            # Testing mode (load pre-trained model)
            logging.info("Testing mode - loading pre-trained model...")
            
            # Load model checkpoint
            checkpoint_path = os.path.join(config.output_dir, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"Loaded model from {checkpoint_path}")
            else:
                logging.warning("No pre-trained model found. Using untrained model.")
            
            # Evaluate model
            evaluator = Evaluator(config, model)
            evaluator.evaluate_model(dataset_manager.val_loader, "Test Model")
            evaluator.visualize_predictions(dataset_manager.val_loader, save_path=config.output_dir)
        
        logging.info("Pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
