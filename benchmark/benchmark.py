import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from trackers import create_tracker


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "benchmark" / "results"


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark single-object trackers."
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to benchmark video.",
    )

    parser.add_argument(
        "--trackers",
        nargs="+",
        choices=["csrt", "nano", "vit"],
        default=["csrt", "nano", "vit"],
        help="Trackers to benchmark.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames. 0 means all frames.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "benchmark_results.csv",
        help="Output CSV path.",
    )

    return parser.parse_args()


def select_initial_roi(video_path: Path):
    """Read the first frame and manually select the target ROI."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open benchmark video: {video_path}"
        )

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            "Could not read the first frame of the benchmark video."
        )

    bbox = cv2.selectROI(
        "Select Benchmark Target",
        frame,
        fromCenter=False,
        showCrosshair=True,
    )

    cv2.destroyWindow("Select Benchmark Target")

    x, y, width, height = map(int, bbox)

    if width <= 0 or height <= 0:
        raise ValueError(
            "Invalid ROI. Width and height must be greater than zero."
        )

    return x, y, width, height


def benchmark_tracker(
    tracker_name: str,
    video_path: Path,
    initial_bbox: tuple[int, int, int, int],
    max_frames: int = 0,
) -> dict:
    """Benchmark one tracker on the same video and initial ROI."""

    tracker = create_tracker(
        tracker_name=tracker_name,
        models_dir=MODELS_DIR,
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video for tracker: {tracker_name}"
        )

    success, first_frame = cap.read()

    if not success:
        cap.release()
        raise RuntimeError(
            "Could not read the first video frame."
        )

    # Measure initialization time
    init_start = time.perf_counter()

    tracker.initialize(
        first_frame,
        initial_bbox,
    )

    initialization_ms = (
        time.perf_counter() - init_start
    ) * 1000.0

    total_frames = 0
    successful_frames = 0
    lost_frames = 0

    update_times_ms: list[float] = []
    scores: list[float] = []

    while True:

        if max_frames > 0 and total_frames >= max_frames:
            break

        success, frame = cap.read()

        if not success:
            break

        # Measure ONLY tracker.update()
        update_start = time.perf_counter()

        tracking_success, _ = tracker.update(frame)

        update_time_ms = (
            time.perf_counter() - update_start
        ) * 1000.0

        update_times_ms.append(update_time_ms)

        total_frames += 1

        if tracking_success:
            successful_frames += 1
        else:
            lost_frames += 1

        score = tracker.get_score()

        if score is not None and tracking_success:
            scores.append(float(score))

    cap.release()

    if not update_times_ms:
        raise RuntimeError(
            f"No frames were processed for tracker: {tracker_name}"
        )

    update_array = np.array(update_times_ms)

    average_update_ms = float(
        np.mean(update_array)
    )

    median_update_ms = float(
        np.median(update_array)
    )

    p95_update_ms = float(
        np.percentile(update_array, 95)
    )

    benchmark_fps = (
        1000.0 / average_update_ms
        if average_update_ms > 0
        else 0.0
    )

    success_rate = (
        successful_frames / total_frames
        if total_frames > 0
        else 0.0
    )

    average_score = (
        float(np.mean(scores))
        if scores
        else None
    )

    return {
        "tracker": tracker.name,
        "total_frames": total_frames,
        "successful_frames": successful_frames,
        "lost_frames": lost_frames,
        "success_rate": success_rate,
        "initialization_ms": initialization_ms,
        "avg_update_ms": average_update_ms,
        "median_update_ms": median_update_ms,
        "p95_update_ms": p95_update_ms,
        "benchmark_fps": benchmark_fps,
        "avg_tracking_score": average_score,
    }


def save_results(
    results: list[dict],
    output_path: Path,
) -> None:
    """Save benchmark results to CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "tracker",
        "total_frames",
        "successful_frames",
        "lost_frames",
        "success_rate",
        "initialization_ms",
        "avg_update_ms",
        "median_update_ms",
        "p95_update_ms",
        "benchmark_fps",
        "avg_tracking_score",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def print_results(results: list[dict]) -> None:
    """Print a readable benchmark summary."""

    print()
    print("=" * 95)
    print("TRACKER BENCHMARK RESULTS")
    print("=" * 95)

    header = (
        f"{'Tracker':<15}"
        f"{'FPS':>10}"
        f"{'Avg ms':>12}"
        f"{'P95 ms':>12}"
        f"{'Lost':>10}"
        f"{'Success %':>14}"
        f"{'Score':>12}"
    )

    print(header)
    print("-" * 95)

    for result in results:

        score = result["avg_tracking_score"]

        score_text = (
            f"{score:.3f}"
            if score is not None
            else "N/A"
        )

        print(
            f"{result['tracker']:<15}"
            f"{result['benchmark_fps']:>10.2f}"
            f"{result['avg_update_ms']:>12.3f}"
            f"{result['p95_update_ms']:>12.3f}"
            f"{result['lost_frames']:>10}"
            f"{result['success_rate'] * 100:>13.2f}%"
            f"{score_text:>12}"
        )

    print("=" * 95)
    print()


def main() -> None:
    args = parse_args()

    video_path = args.video.resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    print(f"Benchmark video: {video_path}")

    print()
    print(
        "Select the target ONCE. "
        "The same ROI will be used for every tracker."
    )

    initial_bbox = select_initial_roi(
        video_path
    )

    print(
        f"Initial ROI: {initial_bbox}"
    )

    results = []

    for tracker_name in args.trackers:

        print()
        print(
            f"Running benchmark for: {tracker_name}"
        )

        result = benchmark_tracker(
            tracker_name=tracker_name,
            video_path=video_path,
            initial_bbox=initial_bbox,
            max_frames=args.max_frames,
        )

        results.append(result)

        print(
            f"Completed {tracker_name}: "
            f"{result['benchmark_fps']:.2f} FPS"
        )

    save_results(
        results,
        args.output,
    )

    print_results(results)

    print(
        f"Results saved to: {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()