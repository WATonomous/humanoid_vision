# Image Segmentation Pipeline Configuration
# ========================================

# Dataset Configuration
dataset_path = 'datasets'
num_classes = 21  # VOC has 20 classes + background
ignore_index = 255  # Void class

# Model Configuration
backbone = 'resnet18'  # Options: 'resnet18', 'resnet34', 'resnet50'
pretrained = True

# Training Configuration
batch_size = 4
num_epochs = 50
learning_rate = 0.006
momentum = 0.9
weight_decay = 1e-4

# Data Augmentation
image_size = (375, 500)
crop_size = 224
min_scale = 0.9
max_scale = 1.1

# Hardware Configuration
use_gpu = True  # Will be overridden by torch.cuda.is_available()
num_workers = 0  # Set to 0 on Windows, 4+ on Linux/Mac

# Training Phases
sanity_check_epochs = 5
overfit_epochs = 40

# Output Configuration
output_dir = 'outputs'
save_interval = 10
log_interval = 10

# Normalization values (ImageNet)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Advanced Options
mixed_precision = False  # Use automatic mixed precision
gradient_clipping = None  # Gradient clipping value (e.g., 1.0)
scheduler = None  # Learning rate scheduler ('cosine', 'step', None)

# Evaluation Options
eval_interval = 1  # Evaluate every N epochs
save_predictions = True  # Save prediction visualizations
compute_per_class_metrics = True  # Compute per-class IoU
