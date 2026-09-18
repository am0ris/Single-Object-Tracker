from pathlib import Path

import cv2
import numpy as np

from .base_tracker import BaseTracker, BBox


class NanoTracker(BaseTracker):
    """TrackerNano implementation."""

    def __init__(
        self,
        backbone_path: str | Path,
        neckhead_path: str | Path,
    ) -> None:
        self._backbone_path = str(backbone_path)
        self._neckhead_path = str(neckhead_path)

        self._validate_model_paths()

        params = cv2.TrackerNano_Params()

        params.backbone = self._backbone_path
        params.neckhead = self._neckhead_path

        self._tracker = cv2.TrackerNano_create(params)

    def _validate_model_paths(self) -> None:
        backbone = Path(self._backbone_path)
        neckhead = Path(self._neckhead_path)

        if not backbone.is_file():
            raise FileNotFoundError(
                f"Nano backbone model not found: {backbone}"
            )

        if not neckhead.is_file():
            raise FileNotFoundError(
                f"Nano neckhead model not found: {neckhead}"
            )

    @property
    def name(self) -> str:
        return "TrackerNano"

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
        return float(self._tracker.getTrackingScore())