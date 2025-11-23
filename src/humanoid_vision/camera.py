"""Placeholder camera module.

This file intentionally contains no implementations. Add Camera classes and helpers here.
"""

from typing import Any
import cv2
import numpy as np
import time


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
        self.overlays: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def set_brightness(self, val: float):
        self.cam.set(cv2.CAP_PROP_BRIGHTNESS, val)

    def getFrame(self) -> np.ndarray:
        """
        getFrame(): returns the current frame of the camera whenever the function is called
        """
        ret, frame = self.cam.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def getFrames(self, duration: int) -> list[np.ndarray]:
        """
        getFrames(int duration): returns a list of frames for however long the specified duration is
        """
        start = time.time()
        frames = []
        while time.time() - start <= duration:
            ret, frame = self.cam.read()
            if not ret:
                raise RuntimeError("Failed to read frame from camera")
            frames.append(frame)

        return frames

    def getFrameStream(self):
        """
        getFrameStream(): returns a stream of frames based on generators (and also log how many skipped frames between generations)
        """
        prev_frame_id = 0
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break  # stop if camera fails
            frame_id = int(self.cam.get(cv2.CAP_PROP_POS_FRAMES))
            skipped = frame_id - prev_frame_id - 1
            if skipped > 0:
                print(f"Skipped {skipped} frames")
            prev_frame_id = frame_id
            yield frame

    def viewStream(self):
        """
        # viewStream(): open a video stream from the current camera
        # press q to quit
        """
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break

            # draw overlays
            for top_left, bottom_right in getattr(self, "overlays", []):
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            cv2.imshow("Camera Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cam.release()
        cv2.destroyAllWindows()

    def addOverlay(self, boundingBoxList: list[tuple[tuple[int, int], tuple[int, int]]]):
        """
        addOverlay(boundingBoxList): utility function to add bounding boxes (using top left and bottom right coordinates), on a camera, that are displayed when we call viewStream(). (its not used when any of the getFrames() are called
        """

        self.overlays = boundingBoxList
