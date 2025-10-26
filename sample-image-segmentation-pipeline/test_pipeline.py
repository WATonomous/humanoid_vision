#!/usr/bin/env python3
"""
Test script for the Image Segmentation Pipeline
===============================================

This script performs basic tests to ensure the pipeline components work correctly.
"""

import os
import sys
import torch
import numpy as np
from main import Config, SegmentationNet, JointToTensor, Normalize, JointCompose

def test_config():
    """Test configuration setup."""
    print("Testing configuration...")
    config = Config()
    
    assert config.num_classes == 21
    assert config.batch_size > 0
    assert config.num_epochs > 0
    assert len(config.mean) == 3
    assert len(config.std) == 3
    
    print("✓ Configuration test passed")


def test_transforms():
    """Test data transforms."""
    print("Testing data transforms...")
    
    # Create dummy data
    from PIL import Image
    dummy_img = Image.new('RGB', (100, 100), color='red')
    dummy_target = Image.new('L', (100, 100), color=0)
    
    # Test JointToTensor
    transform = JointToTensor()
    img_tensor, target_tensor = transform(dummy_img, dummy_target)
    
    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(target_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 100, 100)
    assert target_tensor.shape == (100, 100)
    
    # Test Normalize
    normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    norm_img, norm_target = normalize(img_tensor, target_tensor)
    
    assert norm_img.shape == img_tensor.shape
    assert norm_target.shape == target_tensor.shape
    
    # Test JointCompose
    compose = JointCompose([JointToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    comp_img, comp_target = compose(dummy_img, dummy_target)
    
    assert comp_img.shape == (3, 100, 100)
    assert comp_target.shape == (100, 100)
    
    print("✓ Transforms test passed")


def test_model():
    """Test model creation and forward pass."""
    print("Testing model...")
    
    config = Config()
    model = SegmentationNet(
        num_classes=config.num_classes,
        backbone='resnet18',
        pretrained=False  # Don't download weights for test
    )
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (1, config.num_classes, 224, 224)
    
    # Test training mode
    model.train()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    model.criterion = criterion
    
    dummy_target = torch.randint(0, config.num_classes, (1, 224, 224))
    
    loss = model(dummy_input, gts=dummy_target)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    
    print("✓ Model test passed")


def test_device_detection():
    """Test GPU/CPU device detection."""
    print("Testing device detection...")
    
    config = Config()
    
    if torch.cuda.is_available():
        assert config.use_gpu == True
        print("✓ GPU available and detected")
    else:
        print("✓ CPU mode (GPU not available)")
    
    print("✓ Device detection test passed")


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        import torchvision
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        from tqdm import tqdm
        print("✓ All core modules imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    try:
        from lib.voc import VOCSegmentation
        print("✓ Custom VOC dataset imported successfully")
    except ImportError as e:
        print(f"✗ Custom module import error: {e}")
        return False
    
    try:
        from chainercv.evaluations import eval_semantic_segmentation
        print("✓ ChainerCV evaluation imported successfully")
    except ImportError:
        print("⚠ ChainerCV not available (optional)")
    
    print("✓ Imports test passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("Running Image Segmentation Pipeline Tests")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 5
    
    try:
        if test_imports():
            tests_passed += 1
        
        test_config()
        tests_passed += 1
        
        test_transforms()
        tests_passed += 1
        
        test_model()
        tests_passed += 1
        
        test_device_detection()
        tests_passed += 1
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
    
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Pipeline is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
