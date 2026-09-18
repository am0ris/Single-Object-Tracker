from pathlib import Path

import cv2
import numpy as np

from .base_tracker import BaseTracker, BBox


class VitTracker(BaseTracker):
    """VitTrack implementation using OpenCV's TrackerVit API."""

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:
        self._model_path = str(model_path)

        self._validate_model_path()

        params = cv2.TrackerVit_Params()

        params.net = self._model_path

        self._tracker = cv2.TrackerVit_create(params)

    def _validate_model_path(self) -> None:
        model = Path(self._model_path)

        if not model.is_file():
            raise FileNotFoundError(
                f"VitTrack model not found: {model}"
            )

    @property
    def name(self) -> str:
        return "TrackerVit"

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

    def get_score(self) -> float:
        return float(
            self._tracker.getTrackingScore()
        )