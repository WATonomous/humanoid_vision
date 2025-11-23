# WATonomous + UWRL Humanoid Project - Vision
Vision Prototyping For The Humanoid Project

## Setup Instructions

### 1. Create and activate a Python virtual environment:
```
python -m venv .venv
```
Windows PowerShell Activation
```
.\.venv\Scripts\Activate.ps1
```
---

### 2. Install the humanoid package:
```
pip install -e '.[dev]'
```
---

### 3. Run tests:
```
pytest -q
```