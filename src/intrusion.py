import math
import time


class PerimeterIntrusionDetector:
    """
    Detects when a tracked object crosses a virtual perimeter line.
    """

    def __init__(self, line_start, line_end, cooldown_seconds=5.0):
        self.line_start = tuple(line_start)
        self.line_end = tuple(line_end)

        self.previous_side = {}
        self.last_event_time = {}

        self.cooldown_seconds = cooldown_seconds

    def _signed_side(self, point):
        """
        Determines which side of the line the point lies on.

        Returns:
            positive number -> one side
            negative number -> other side
            zero -> on line
        """

        px, py = point
        x1, y1 = self.line_start
        x2, y2 = self.line_end

        return (
            (x2 - x1) * (py - y1)
            - (y2 - y1) * (px - x1)
        )

    def _distance_from_line(self, point):
        px, py = point
        x1, y1 = self.line_start
        x2, y2 = self.line_end

        numerator = abs(
            (y2 - y1) * px
            - (x2 - x1) * py
            + x2 * y1
            - y2 * x1
        )

        denominator = math.sqrt(
            (y2 - y1) ** 2 +
            (x2 - x1) ** 2
        )

        if denominator == 0:
            return float("inf")

        return numerator / denominator

    def check_crossing(self, track_id, current_point):
        """
        Returns True only when the object crosses the line.
        """

        if track_id is None or track_id < 0:
            return False

        current_side = self._signed_side(current_point)

        if track_id not in self.previous_side:
            self.previous_side[track_id] = current_side
            return False

        previous_side = self.previous_side[track_id]
        self.previous_side[track_id] = current_side

        # Ignore points extremely close to line.
        if abs(current_side) < 1e-6 or abs(previous_side) < 1e-6:
            return False

        # Different signs = crossed the line.
        crossed = (
            (previous_side < 0 < current_side)
            or
            (previous_side > 0 > current_side)
        )

        if not crossed:
            return False

        now = time.time()

        last_time = self.last_event_time.get(track_id, 0)

        if now - last_time < self.cooldown_seconds:
            return False

        self.last_event_time[track_id] = now

        return True

    def cleanup(self, active_track_ids):
        """
        Remove old track IDs from memory.
        """

        active_track_ids = set(active_track_ids)

        old_ids = [
            track_id
            for track_id in self.previous_side
            if track_id not in active_track_ids
        ]

        for track_id in old_ids:
            self.previous_side.pop(track_id, None)
            self.last_event_time.pop(track_id, None)