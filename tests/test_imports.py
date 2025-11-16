import importlib


def test_import_camera_and_pipeline():
    """Ensure camera and pipeline modules import without syntax errors."""
    importlib.import_module("humanoid_vision.camera")
    importlib.import_module("humanoid_vision.pipeline")
    importlib.import_module("humanoid_vision.main")
