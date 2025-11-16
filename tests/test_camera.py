import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from humanoid_vision.camera import Camera, available


@pytest.fixture
def mock_capture():
    """Create a mock cv2.VideoCapture instance."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_cap.get.return_value = 10
    return mock_cap


def test_init_valid_camera(mock_capture):
    with patch("humanoid_vision.camera.cv2.VideoCapture", return_value=mock_capture):
        # Assume available camera 0
        available.clear()
        available.append(0)
        cam = Camera(0)
        assert cam.cam == mock_capture
        assert cam.overlays == []


def test_init_invalid_camera():
    available.clear()
    with pytest.raises(ValueError):
        Camera(99)


def test_set_brightness(mock_capture):
    with patch("humanoid_vision.camera.cv2.VideoCapture", return_value=mock_capture):
        available[:] = [0]
        cam = Camera(0)
        cam.set_brightness(0.5)
        mock_capture.set.assert_called_with(10, 0.5)  # 10 = cv2.CAP_PROP_BRIGHTNESS


def test_getFrame_success(mock_capture):
    with patch("humanoid_vision.camera.cv2.VideoCapture", return_value=mock_capture):
        available[:] = [0]
        cam = Camera(0)
        frame = cam.getFrame()
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8


def test_getFrame_failure(mock_capture):
    with patch("humanoid_vision.camera.cv2.VideoCapture", return_value=mock_capture):
        mock_capture.read.return_value = (False, None)
        available[:] = [0]
        cam = Camera(0)
        with pytest.raises(RuntimeError):
            cam.getFrame()


def test_getFrames_duration(mock_capture):
    """Should return multiple frames within the duration."""
    with patch("humanoid_vision.camera.cv2.VideoCapture", return_value=mock_capture):
        available[:] = [0]
        cam = Camera(0)
        frames = cam.getFrames(duration=0.2)
        assert isinstance(frames, list)
        assert all(isinstance(f, np.ndarray) for f in frames)
        assert len(frames) >= 1


def test_getFrameStream_yields_frames(mock_capture):
    with patch("humanoid_vision.camera.cv2.VideoCapture", return_value=mock_capture):
        available[:] = [0]
        cam = Camera(0)
        gen = cam.getFrameStream()
        frame = next(gen)
        assert isinstance(frame, np.ndarray)


def test_addOverlay_stores_bboxes(mock_capture):
    with patch("humanoid_vision.camera.cv2.VideoCapture", return_value=mock_capture):
        available[:] = [0]
        cam = Camera(0)
        boxes = [((0, 0), (100, 100)), ((50, 50), (150, 150))]
        cam.addOverlay(boxes)
        assert cam.overlays == boxes
