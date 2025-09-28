"""Placeholder camera module.

This file intentionally contains no implementations. Add Camera classes and helpers here.
"""

from typing import Any
import cv2
from cv2_enumerate_cameras import enumerate_cameras
import torch
import torch.nn as nn

def placeholder_camera() -> Any:
    """Placeholder function to indicate where camera API will live."""
    raise NotImplementedError("Camera module not yet implemented")

MAX_IDX = 10
#cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

available = []
for i in range(MAX_IDX):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available.append(i)
        cap.release()

print(available)

class Camera():
    def __init__(self, idx: int): #could also be a string
        """
        initialize instance to a cv2 index
        """
        if idx not in available:
            raise ValueError(f"id isn't an available device, select from {available}")
        
        self.cam = cv2.VideoCapture(idx)
            # Show the frame in a window

    def set_brightness(self, val:float):
        self.cam.set(cv2.CAP_PROP_BRIGHTNESS, val)

def test():
    cap = Camera(0)
    print("Brightness before:", cap.cam.get(cv2.CAP_PROP_BRIGHTNESS))
    cap.set_brightness(0.05)
    print("Brightness after:", cap.cam.get(cv2.CAP_PROP_BRIGHTNESS))
    ret, frame = cap.cam.read()
    cv2.imshow("One Frame", frame)
    cv2.waitKey(0)   # wait until any key is pressed
    cv2.destroyAllWindows()
if __name__ == "__main__":
    test()
