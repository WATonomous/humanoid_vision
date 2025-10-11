"""Placeholder camera module.

This file intentionally contains no implementations. Add Camera classes and helpers here.
"""

from typing import Any
import cv2


MAX_IDX = 10

available = []
for i in range(MAX_IDX):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available.append(i)
        cap.release()

print(available)

class Camera:
    def __init__(self, idx: int):
        """
        Initialize instance to a cv2 index
        """
        if idx not in available:
            raise ValueError(f"id isn't an available device, select from {available}")
        
        self.cam = cv2.VideoCapture(idx)


    def set_brightness(self, val:float):
        self.cam.set(cv2.CAP_PROP_BRIGHTNESS, val)

