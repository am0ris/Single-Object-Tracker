from abc import ABC, abstractmethod

import numpy as np


BBox = tuple[int, int, int, int]


class BaseTracker(ABC):
    """Common interface for all single-object trackers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tracker name."""
        raise NotImplementedError

    @abstractmethod
    def initialize(
        self,
        frame: np.ndarray,
        bbox: BBox,
    ) -> None:
        """Initialize the tracker using the first frame and target ROI."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        frame: np.ndarray,
    ) -> tuple[bool, BBox]:
        """
        Update the tracker using a new frame.

        Returns:
            success: Whether the target was successfully tracked.
            bbox: Estimated target bounding box.
        """
        raise NotImplementedError

    def get_score(self) -> float | None:
        """
        Return the tracker confidence/score when available.

        Trackers that do not expose a score return None.
        """
        return None