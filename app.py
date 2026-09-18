import argparse
from pathlib import Path

import cv2

from trackers import create_tracker
from utils import FPSCounter, draw_tracking_result


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

WINDOW_NAME = "CV Object Tracker"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Single Object Tracker"
    )

    parser.add_argument(
        "--tracker",
        choices=["csrt", "nano", "vit"],
        default="csrt",
        help="Tracking algorithm to use.",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index.",
    )

    return parser.parse_args()


def select_target(frame):
    """Allow the user to select the target ROI."""
    bbox = cv2.selectROI(
        "Select Target",
        frame,
        fromCenter=False,
        showCrosshair=True,
    )

    cv2.destroyWindow("Select Target")

    x, y, width, height = map(int, bbox)

    if width <= 0 or height <= 0:
        return None

    return x, y, width, height


def main() -> None:
    args = parse_args()

    # Create selected tracker
    tracker = create_tracker(
        tracker_name=args.tracker,
        models_dir=MODELS_DIR,
    )

    print(f"Selected tracker: {tracker.name}")

    # Open webcam
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam with index {args.camera}."
        )

    # Capture first frame
    success, frame = cap.read()

    if not success:
        cap.release()
        raise RuntimeError(
            "Could not read the first frame from the webcam."
        )

    # Select target
    bbox = select_target(frame)

    if bbox is None:
        cap.release()
        cv2.destroyAllWindows()

        print("No valid target was selected.")
        return

    # Initialize tracker
    tracker.initialize(
        frame,
        bbox,
    )

    print(f"{tracker.name} tracker initialized.")
    print("Press 'q' to quit.")

    fps_counter = FPSCounter()

    while True:
        # Read frame
        success, frame = cap.read()

        if not success:
            print("Could not read frame from webcam.")
            break

        # Update tracker
        tracking_success, bbox = tracker.update(frame)

        # Update FPS
        fps = fps_counter.update()

        # Get optional tracker score
        score = tracker.get_score()

        # Draw result
        output = draw_tracking_result(
            frame=frame,
            bbox=bbox,
            tracker_name=tracker.name,
            success=tracking_success,
            fps=fps,
            score=score,
        )

        # Display
        cv2.imshow(
            WINDOW_NAME,
            output,
        )

        # Keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()