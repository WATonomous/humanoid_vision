import pytest
import cv2
from camera import Camera, available

@pytest.mark.skipif(not available, reason="No available camera devices detected")
def test_camera_basic():
    cap = Camera(available[0])

    before = cap.cam.get(cv2.CAP_PROP_BRIGHTNESS)
    cap.set_brightness(0.05)
    after = cap.cam.get(cv2.CAP_PROP_BRIGHTNESS)

    assert after is not None
    assert cap.cam.isOpened()

    ret, frame = cap.cam.read()
    assert ret is True
    assert frame is not None

    cap.cam.release()
