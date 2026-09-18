from pathlib import Path

from .base_tracker import BBox, BaseTracker
from .csrt_tracker import CSRTTracker
from .nano_tracker import NanoTracker
from .vit_tracker import VitTracker


def create_tracker(
    tracker_name: str,
    models_dir: Path,
) -> BaseTracker:
    """
    Create a tracker instance based on the requested tracker name.

    Args:
        tracker_name: Tracker identifier: csrt, nano, or vit.
        models_dir: Directory containing DNN model files.

    Returns:
        Initialized tracker object.

    Raises:
        ValueError: If the tracker name is unsupported.
    """
    name = tracker_name.strip().lower()

    if name == "csrt":
        return CSRTTracker()

    if name == "nano":
        return NanoTracker(
            backbone_path=(
                models_dir / "nanotrack_backbone_sim.onnx"
            ),
            neckhead_path=(
                models_dir / "nanotrack_head_sim.onnx"
            ),
        )

    if name == "vit":
        return VitTracker(
            model_path=(
                models_dir /
                "object_tracking_vittrack_2023sep.onnx"
            ),
        )

    raise ValueError(
        f"Unsupported tracker '{tracker_name}'. "
        "Available trackers: csrt, nano, vit."
    )


__all__ = [
    "BBox",
    "BaseTracker",
    "CSRTTracker",
    "NanoTracker",
    "VitTracker",
    "create_tracker",
]