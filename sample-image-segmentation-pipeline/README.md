# Image Segmentation Training Pipeline

A comprehensive pipeline for training semantic segmentation models on the PASCAL VOC dataset. This implementation includes data handling, model architecture, training, validation, and testing components.

## Features

### 🎯 **Core Components**
- **Dataset Management**: Automated PASCAL VOC dataset handling with custom transforms
- **Model Architecture**: Encoder-decoder network with skip connections based on ResNet backbone
- **Training Pipeline**: Complete training loop with validation and checkpointing
- **Evaluation**: Comprehensive testing with mIoU metrics and visualization
- **Configuration**: Flexible configuration system for easy experimentation

### 🔧 **Advanced Features**
- **Data Augmentation**: Random resize, crop, flip, and normalization
- **Multi-phase Training**: Sanity check, overfitting, and full training modes
- **GPU Support**: Automatic GPU detection and utilization
- **Logging**: Comprehensive logging with file and console output
- **Visualization**: Automatic prediction visualization and saving
- **Model Checkpointing**: Automatic best model saving and resuming

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sample-image-segmentation-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

#### 1. Sanity Check Training (Overfit on Single Image)
```bash
python main.py --mode sanity --epochs 5
```

#### 2. Full Training
```bash
python main.py --mode train --epochs 50 --batch_size 4 --lr 0.006
```

#### 3. Testing Pre-trained Model
```bash
python main.py --mode test
```

### Advanced Usage

#### Custom Configuration
```bash
python main.py --mode train \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.01 \
    --gpu
```

## Pipeline Structure

### 📁 **Dataset Section**
- **VOCSegmentation**: Custom PASCAL VOC dataset loader
- **Data Transforms**: Joint transforms for image and mask augmentation
- **Data Loaders**: Efficient data loading with multiprocessing support

### 🧠 **Model Section**
- **SegmentationNet**: Encoder-decoder architecture with skip connections
- **Backbone Support**: ResNet18/34/50 with pre-trained weights
- **Skip Connections**: Feature fusion from early encoder layers

### 🏋️ **Training Section**
- **Trainer Class**: Complete training loop with validation
- **Optimizer**: SGD with momentum and weight decay
- **Loss Function**: Cross-entropy loss with ignore index
- **Checkpointing**: Automatic model saving and resuming

### 🧪 **Testing Section**
- **Evaluator Class**: Model evaluation with mIoU metrics
- **Visualization**: Prediction visualization and comparison
- **Metrics**: Comprehensive performance evaluation

## Configuration Options

### Model Configuration
```python
config.backbone = 'resnet18'  # or 'resnet34', 'resnet50'
config.pretrained = True
config.num_classes = 21
```

### Training Configuration
```python
config.batch_size = 4
config.num_epochs = 50
config.learning_rate = 0.006
config.momentum = 0.9
config.weight_decay = 1e-4
```

### Data Configuration
```python
config.image_size = (375, 500)
config.crop_size = 224
config.min_scale = 0.9
config.max_scale = 1.1
```

## Model Architecture

The segmentation network follows an encoder-decoder architecture:

```
Input Image (3, H, W)
    ↓
Encoder (ResNet Backbone)
    ↓
Skip Connection
    ↓
Decoder (Transpose Convolutions)
    ↓
Final Segmentation (21, H, W)
```

### Key Components:
- **Encoder**: Pre-trained ResNet backbone for feature extraction
- **Skip Connection**: Feature fusion from early encoder layers
- **Decoder**: Transpose convolutions for upsampling
- **Final Layer**: 1x1 convolution for class prediction

## Training Phases

### 1. Sanity Check Phase
- Train on single image to verify pipeline
- Quick debugging and hyperparameter tuning
- Expected to overfit on training image

### 2. Full Training Phase
- Train on complete training dataset
- Regular validation and checkpointing
- Best model selection based on validation mIoU

### 3. Testing Phase
- Load best model and evaluate on test set
- Generate visualizations and metrics
- Final performance assessment

## Output Structure

```
outputs/
├── training.log          # Training logs
├── best_model.pth        # Best model checkpoint
├── checkpoint_epoch_*.pth # Regular checkpoints
└── prediction_*.png      # Visualization outputs
```

## Performance Metrics

- **mIoU (mean Intersection over Union)**: Primary segmentation metric
- **Pixel Accuracy**: Overall classification accuracy
- **Per-class IoU**: Individual class performance
- **Training Loss**: Cross-entropy loss during training

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Dataset Download Issues**: Check internet connection and disk space
3. **Import Errors**: Ensure all dependencies are installed
4. **Windows Multiprocessing**: Set `num_workers=0` on Windows

### Performance Tips

1. **GPU Usage**: Enable GPU for faster training
2. **Batch Size**: Increase batch size if GPU memory allows
3. **Data Loading**: Use multiple workers for faster data loading
4. **Mixed Precision**: Consider using automatic mixed precision

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PASCAL VOC dataset creators
- PyTorch and TorchVision teams
- ChainerCV for evaluation metrics
- Original assignment design by Towaki Takikawa and Olga Veksler
