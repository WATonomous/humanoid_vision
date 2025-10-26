# Image Segmentation Pipeline Makefile
# ===================================

.PHONY: help install test train sanity test-model clean

# Default target
help:
	@echo "Image Segmentation Training Pipeline"
	@echo "===================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run pipeline tests"
	@echo "  make sanity      - Run sanity check training"
	@echo "  make train       - Run full training"
	@echo "  make test-model  - Test pre-trained model"
	@echo "  make example     - Run example script"
	@echo "  make clean       - Clean output files"
	@echo ""
	@echo "Advanced usage:"
	@echo "  make train EPOCHS=100 BATCH_SIZE=8 LR=0.01"
	@echo "  make sanity EPOCHS=10"

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	python test_pipeline.py

# Run example script
example:
	python example.py

# Sanity check training (overfit on single image)
sanity:
	python main.py --mode sanity --epochs $(or $(EPOCHS),5)

# Full training
train:
	python main.py --mode train \
		--epochs $(or $(EPOCHS),50) \
		--batch_size $(or $(BATCH_SIZE),4) \
		--lr $(or $(LR),0.006)

# Test pre-trained model
test-model:
	python main.py --mode test

# Clean output files
clean:
	rm -rf outputs/
	rm -rf datasets/
	rm -rf __pycache__/
	rm -rf lib/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Quick training (fewer epochs for testing)
quick-train:
	python main.py --mode train --epochs 10 --batch_size 2

# GPU training
gpu-train:
	python main.py --mode train --epochs 50 --gpu

# Debug mode (CPU only)
debug:
	python main.py --mode sanity --epochs 3
	python main.py --mode train --epochs 2 --batch_size 1
