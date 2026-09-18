import time

import cv2

from trackers import CSRTTracker


WINDOW_NAME = "CV Object Tracker"
CAMERA_INDEX = 0


def main() -> None:
    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam with index {CAMERA_INDEX}."
        )

    # Read the first frame
    success, frame = cap.read()

    if not success:
        cap.release()
        raise RuntimeError(
            "Could not read the first frame from the webcam."
        )

    # Select target
    bbox = cv2.selectROI(
        "Select Target",
        frame,
        fromCenter=False,
        showCrosshair=True,
    )

    cv2.destroyWindow("Select Target")

    # Validate ROI
    x, y, width, height = map(int, bbox)

    if width <= 0 or height <= 0:
        cap.release()
        cv2.destroyAllWindows()

        print("No valid target was selected.")
        return

    # Create CSRT tracker through our abstraction
    tracker = CSRTTracker()

    tracker.initialize(
        frame,
        (x, y, width, height),
    )

    print(f"{tracker.name} tracker initialized.")
    print("Press 'q' to quit.")

    previous_time = time.perf_counter()

    while True:
        # Read next frame
        success, frame = cap.read()

        if not success:
            print("Could not read frame from webcam.")
            break

        # Update tracker
        tracking_success, bbox = tracker.update(frame)

        # Calculate FPS
        current_time = time.perf_counter()
        elapsed_time = current_time - previous_time

        fps = 0.0

        if elapsed_time > 0:
            fps = 1.0 / elapsed_time

        previous_time = current_time

        # Draw tracking result
        if tracking_success:
            x, y, width, height = bbox

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"{tracker.name} | Tracking",
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        else:
            cv2.putText(
                frame,
                f"{tracker.name} | Target Lost",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # Draw FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        # Display frame
        cv2.imshow(WINDOW_NAME, frame)

        # Keyboard control
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()