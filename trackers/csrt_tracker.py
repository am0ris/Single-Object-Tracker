import cv2
import numpy as np

from .base_tracker import BaseTracker, BBox


class CSRTTracker(BaseTracker):
    """CSRT implementation of the common tracker interface."""

    def __init__(self) -> None:
        self._tracker = cv2.TrackerCSRT_create()

    @property
    def name(self) -> str:
        return "CSRT"

    def initialize(
        self,
        frame: np.ndarray,
        bbox: BBox,
    ) -> None:
        self._tracker.init(frame, bbox)

    def update(
        self,
        frame: np.ndarray,
    ) -> tuple[bool, BBox]:

        success, bbox = self._tracker.update(frame)

        if not success:
            return False, (0, 0, 0, 0)

        x, y, width, height = map(int, bbox)

        return True, (x, y, width, height)