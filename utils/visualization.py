from typing import Optional

import cv2
import numpy as np

from trackers.base_tracker import BBox


def draw_tracking_result(
    frame: np.ndarray,
    bbox: BBox,
    tracker_name: str,
    success: bool,
    fps: float,
    score: Optional[float] = None,
) -> np.ndarray:
    """
    Draw tracking information on a frame.

    Args:
        frame: Current video frame.
        bbox: Current target bounding box.
        tracker_name: Name of the active tracker.
        success: Whether tracking succeeded.
        fps: Current FPS.
        score: Optional tracker confidence score.

    Returns:
        Annotated frame.
    """
    output = frame.copy()

    if success:
        x, y, width, height = bbox

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            f"{tracker_name} | Tracking",
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    else:
        cv2.putText(
            output,
            f"{tracker_name} | Target Lost",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    if score is not None:
        cv2.putText(
            output,
            f"Score: {score:.3f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

    return output