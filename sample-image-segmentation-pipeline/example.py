#!/usr/bin/env python3
"""
Example usage of the Image Segmentation Training Pipeline
=========================================================

This script demonstrates how to use the main pipeline for different tasks.
"""

import os
import sys
import logging
from main import Config, DatasetManager, SegmentationNet, Trainer, Evaluator

def example_sanity_check():
    """Example: Sanity check training on single image."""
    print("=" * 60)
    print("SANITY CHECK TRAINING EXAMPLE")
    print("=" * 60)
    
    # Setup configuration
    config = Config()
    config.num_epochs = 5
    config.use_gpu = False  # Use CPU for this example
    config.output_dir = 'example_outputs'
    
    # Setup logging
    os.makedirs(config.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create datasets
        dataset_manager = DatasetManager(config)
        dataset_manager.create_datasets()
        dataset_manager.create_dataloaders()
        
        # Create model
        model = SegmentationNet(
            num_classes=config.num_classes,
            backbone=config.backbone,
            pretrained=config.pretrained
        )
        
        # Create trainer
        trainer = Trainer(config, model, {
            'train': dataset_manager.sanity_loader,
            'val': dataset_manager.sanity_loader
        })
        
        # Train for sanity check
        print("Starting sanity check training...")
        trainer.train()
        
        # Evaluate
        evaluator = Evaluator(config, model)
        loss, miou = evaluator.evaluate_model(dataset_manager.sanity_loader, "Sanity Model")
        
        print(f"Sanity check completed!")
        print(f"Final Loss: {loss:.4f}")
        print(f"Final mIoU: {miou:.4f}")
        
    except Exception as e:
        print(f"Error in sanity check: {e}")


def example_model_architecture():
    """Example: Demonstrate model architecture."""
    print("=" * 60)
    print("MODEL ARCHITECTURE EXAMPLE")
    print("=" * 60)
    
    config = Config()
    
    # Create model
    model = SegmentationNet(
        num_classes=config.num_classes,
        backbone='resnet18',
        pretrained=True
    )
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Architecture: {config.backbone}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Number of Classes: {config.num_classes}")
    
    # Print model structure
    print("\nModel Structure:")
    print(model)


def example_data_transforms():
    """Example: Demonstrate data transforms."""
    print("=" * 60)
    print("DATA TRANSFORMS EXAMPLE")
    print("=" * 60)
    
    config = Config()
    dataset_manager = DatasetManager(config)
    
    # Create a sample dataset
    dataset_manager.create_datasets()
    
    # Get a sample
    sample_img, sample_target = dataset_manager.sanity_dataset[0]
    
    print(f"Original image size: {sample_img.size}")
    print(f"Original target size: {sample_target.size}")
    
    # Apply transforms
    transformed_img, transformed_target = dataset_manager.sanity_transforms(sample_img, sample_target)
    
    print(f"Transformed image shape: {transformed_img.shape}")
    print(f"Transformed target shape: {transformed_target.shape}")
    print(f"Image value range: [{transformed_img.min():.3f}, {transformed_img.max():.3f}]")


def example_evaluation():
    """Example: Demonstrate evaluation without training."""
    print("=" * 60)
    print("EVALUATION EXAMPLE")
    print("=" * 60)
    
    config = Config()
    config.use_gpu = False
    
    try:
        # Create datasets
        dataset_manager = DatasetManager(config)
        dataset_manager.create_datasets()
        dataset_manager.create_dataloaders()
        
        # Create untrained model
        model = SegmentationNet(
            num_classes=config.num_classes,
            backbone=config.backbone,
            pretrained=config.pretrained
        )
        
        # Evaluate untrained model
        evaluator = Evaluator(config, model)
        loss, miou = evaluator.evaluate_model(dataset_manager.val_loader, "Untrained Model")
        
        print(f"Untrained Model Results:")
        print(f"  Loss: {loss:.4f}")
        print(f"  mIoU: {miou:.4f}")
        
    except Exception as e:
        print(f"Error in evaluation: {e}")


def main():
    """Run all examples."""
    print("Image Segmentation Pipeline Examples")
    print("====================================")
    
    # Check if we can import required modules
    try:
        import torch
        import torchvision
        import numpy as np
        import matplotlib.pyplot as plt
        print("✓ All required modules imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install required dependencies: pip install -r requirements.txt")
        return
    
    # Run examples
    try:
        example_model_architecture()
        print()
        
        example_data_transforms()
        print()
        
        example_evaluation()
        print()
        
        # Uncomment to run sanity check (requires dataset download)
        # example_sanity_check()
        
        print("=" * 60)
        print("EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nTo run full training:")
        print("  python main.py --mode train --epochs 50")
        print("\nTo run sanity check:")
        print("  python main.py --mode sanity --epochs 5")
        print("\nTo test pre-trained model:")
        print("  python main.py --mode test")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("This might be due to missing dataset or dependencies.")


if __name__ == "__main__":
    main()
