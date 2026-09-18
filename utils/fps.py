import time


class FPSCounter:
    """Measure and smooth real-time processing FPS."""

    def __init__(self, smoothing: float = 0.9) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(
                "smoothing must be in the range [0.0, 1.0)."
            )

        self._smoothing = smoothing
        self._previous_time = time.perf_counter()
        self._fps = 0.0

    def update(self) -> float:
        """Update the FPS measurement and return the smoothed FPS."""
        current_time = time.perf_counter()

        elapsed = current_time - self._previous_time

        self._previous_time = current_time

        if elapsed <= 0:
            return self._fps

        instant_fps = 1.0 / elapsed

        if self._fps == 0.0:
            self._fps = instant_fps
        else:
            self._fps = (
                self._smoothing * self._fps
                + (1.0 - self._smoothing) * instant_fps
            )

        return self._fps

    @property
    def value(self) -> float:
        """Return the current smoothed FPS."""
        return self._fps