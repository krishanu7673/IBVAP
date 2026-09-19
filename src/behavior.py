import math
import time


class BehaviorAnalyzer:
    def __init__(
        self,
        perimeter_line=None,
        restricted_distance=150,
        high_security_distance=75,
        loitering_seconds=20,
        approach_threshold=2,
        early_warning_score=60,
        repeated_approach_threshold=2
    ):
        self.perimeter_line = perimeter_line

        self.restricted_distance = float(
            restricted_distance
        )

        self.high_security_distance = float(
            high_security_distance
        )

        self.loitering_seconds = float(
            loitering_seconds
        )

        self.approach_threshold = float(
            approach_threshold
        )

        self.early_warning_score = float(
            early_warning_score
        )

        self.repeated_approach_threshold = int(
            repeated_approach_threshold
        )

        self.track_history = {}
        self.track_first_seen = {}
        self.track_last_seen = {}

        # Risk history
        self.track_risk_history = {}
        self.track_previous_level = {}

        # -----------------------------------------------------
        # NEW: APPROACH INTELLIGENCE
        # -----------------------------------------------------

        # Number of distinct approach episodes.
        self.track_approach_count = {}

        # Whether the track is currently approaching.
        self.track_currently_approaching = {}

        # Start time of the current approach episode.
        self.track_approach_start = {}

        # Last time an approach was detected.
        self.track_last_approach = {}

        # -----------------------------------------------------
        # NEW: RISK MOMENTUM
        # -----------------------------------------------------

        # Number of consecutive observations where risk increased.
        self.track_risk_increase_streak = {}

        # -----------------------------------------------------
        # NEW: EARLY WARNING STATE
        # -----------------------------------------------------

        self.track_early_warning = {}

    # ---------------------------------------------------------
    # DISTANCE BETWEEN TWO POINTS
    # ---------------------------------------------------------

    def _distance(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    # ---------------------------------------------------------
    # DISTANCE FROM LINE SEGMENT
    # ---------------------------------------------------------

    def _distance_to_line(self, point):
        if not self.perimeter_line:
            return None

        p1 = self.perimeter_line[0]
        p2 = self.perimeter_line[1]

        x, y = point
        x1, y1 = p1
        x2, y2 = p2

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return self._distance(
                point,
                p1
            )

        t = (
            (x - x1) * dx +
            (y - y1) * dy
        ) / (
            dx * dx +
            dy * dy
        )

        t = max(
            0.0,
            min(1.0, t)
        )

        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return self._distance(
            point,
            (closest_x, closest_y)
        )

    # ---------------------------------------------------------
    # MOVEMENT DIRECTION
    # ---------------------------------------------------------

    def _get_direction(
        self,
        previous_point,
        current_point
    ):
        if previous_point is None:
            return "UNKNOWN"

        dx = (
            current_point[0] -
            previous_point[0]
        )

        dy = (
            current_point[1] -
            previous_point[1]
        )

        movement = math.sqrt(
            dx * dx +
            dy * dy
        )

        if movement < 2:
            return "STATIONARY"

        if abs(dx) > abs(dy):
            if dx > 0:
                return "EAST"
            return "WEST"

        if dy > 0:
            return "SOUTH"

        return "NORTH"

    # ---------------------------------------------------------
    # ZONE CLASSIFICATION
    # ---------------------------------------------------------

    def _get_zone(self, distance):
        if distance is None:
            return "UNKNOWN"

        if distance <= self.high_security_distance:
            return "HIGH_SECURITY"

        if distance <= self.restricted_distance:
            return "RESTRICTED"

        return "NORMAL"

    # ---------------------------------------------------------
    # RISK LEVEL
    # ---------------------------------------------------------

    def _get_risk_level(self, risk_score):
        if risk_score >= 75:
            return "CRITICAL"

        if risk_score >= 50:
            return "HIGH"

        if risk_score >= 25:
            return "MEDIUM"

        return "LOW"

    # ---------------------------------------------------------
    # RISK TREND
    # ---------------------------------------------------------

    def _get_risk_trend(
        self,
        track_id,
        current_score
    ):
        history = self.track_risk_history.get(
            track_id,
            []
        )

        if not history:
            return "STABLE"

        # Compare against a short recent window instead
        # of relying only on the immediately previous frame.
        recent = history[-5:]

        if not recent:
            return "STABLE"

        average_recent = (
            sum(recent) /
            len(recent)
        )

        difference = (
            current_score -
            average_recent
        )

        if difference > 5:
            return "INCREASING"

        if difference < -5:
            return "DECREASING"

        return "STABLE"

    # ---------------------------------------------------------
    # RISK ESCALATION
    # ---------------------------------------------------------

    def _is_risk_escalating(
        self,
        track_id,
        current_level
    ):
        previous_level = (
            self.track_previous_level.get(
                track_id
            )
        )

        if previous_level is None:
            return False

        level_order = {
            "LOW": 0,
            "MEDIUM": 1,
            "HIGH": 2,
            "CRITICAL": 3
        }

        previous_value = level_order.get(
            previous_level,
            0
        )

        current_value = level_order.get(
            current_level,
            0
        )

        return current_value > previous_value

    # ---------------------------------------------------------
    # UPDATE APPROACH STATE
    # ---------------------------------------------------------

    def _update_approach_state(
        self,
        track_id,
        approaching,
        current_time
    ):
        if track_id not in self.track_approach_count:
            self.track_approach_count[track_id] = 0

        previous_approaching = (
            self.track_currently_approaching.get(
                track_id,
                False
            )
        )

        # Start a NEW approach episode only when the
        # track changes from not approaching -> approaching.
        if approaching and not previous_approaching:
            self.track_approach_count[track_id] += 1

            self.track_approach_start[
                track_id
            ] = current_time

        if approaching:
            self.track_last_approach[
                track_id
            ] = current_time

        self.track_currently_approaching[
            track_id
        ] = approaching

        approach_duration = 0.0

        if approaching:
            start_time = (
                self.track_approach_start.get(
                    track_id,
                    current_time
                )
            )

            approach_duration = max(
                0.0,
                current_time - start_time
            )

        return {
            "approach_count":
                self.track_approach_count.get(
                    track_id,
                    0
                ),

            "approach_duration":
                approach_duration
        }

    # ---------------------------------------------------------
    # RISK MOMENTUM
    # ---------------------------------------------------------

    def _update_risk_momentum(
        self,
        track_id,
        current_score
    ):
        history = self.track_risk_history.get(
            track_id,
            []
        )

        if not history:
            self.track_risk_increase_streak[
                track_id
            ] = 0

            return 0

        previous_score = history[-1]

        if current_score > previous_score + 2:
            self.track_risk_increase_streak[
                track_id
            ] = (
                self.track_risk_increase_streak.get(
                    track_id,
                    0
                ) + 1
            )

        elif current_score < previous_score - 2:
            self.track_risk_increase_streak[
                track_id
            ] = 0

        return self.track_risk_increase_streak.get(
            track_id,
            0
        )

    # ---------------------------------------------------------
    # EARLY WARNING
    # ---------------------------------------------------------

    def _calculate_early_warning(
        self,
        risk_score,
        zone,
        approaching,
        approach_count,
        approach_duration,
        risk_trend,
        risk_increase_streak,
        loitering
    ):
        warning = False
        reasons = []

        # High risk by itself can trigger early warning.
        if risk_score >= self.early_warning_score:
            warning = True
            reasons.append(
                "Elevated contextual risk"
            )

        # Sustained approach.
        if (
            approaching
            and approach_duration >= 2
        ):
            warning = True
            reasons.append(
                "Sustained approach toward perimeter"
            )

        # Repeated approach behavior.
        if (
            approach_count >=
            self.repeated_approach_threshold
        ):
            warning = True
            reasons.append(
                "Repeated approach toward perimeter"
            )

        # Risk continuously increasing.
        if (
            risk_trend == "INCREASING"
            and risk_increase_streak >= 3
        ):
            warning = True
            reasons.append(
                "Risk continuously increasing"
            )

        # High-security proximity + approach.
        if (
            zone == "HIGH_SECURITY"
            and approaching
        ):
            warning = True
            reasons.append(
                "Approaching from high-security zone"
            )

        # Long-duration loitering.
        if loitering:
            warning = True
            reasons.append(
                "Persistent loitering behavior"
            )

        if not reasons:
            reasons.append(
                "No early-warning indicators"
            )

        return warning, reasons

    # ---------------------------------------------------------
    # MAIN BEHAVIOR ANALYSIS
    # ---------------------------------------------------------

    def analyze(
        self,
        track_id,
        center,
        current_time=None
    ):
        if current_time is None:
            current_time = time.time()

        # -----------------------------------------------------
        # TRACK INITIALIZATION
        # -----------------------------------------------------

        if track_id not in self.track_history:
            self.track_history[track_id] = []

            self.track_first_seen[
                track_id
            ] = current_time

            self.track_risk_history[
                track_id
            ] = []

            self.track_approach_count[
                track_id
            ] = 0

            self.track_currently_approaching[
                track_id
            ] = False

            self.track_risk_increase_streak[
                track_id
            ] = 0

        history = self.track_history[
            track_id
        ]

        previous_point = None

        if history:
            previous_point = history[-1]

        history.append(center)

        # Keep only recent movement history.
        if len(history) > 30:
            history.pop(0)

        self.track_last_seen[
            track_id
        ] = current_time

        # -----------------------------------------------------
        # BASIC BEHAVIOR
        # -----------------------------------------------------

        direction = self._get_direction(
            previous_point,
            center
        )

        distance = self._distance_to_line(
            center
        )

        zone = self._get_zone(
            distance
        )

        dwell_time = (
            current_time -
            self.track_first_seen[track_id]
        )

        # -----------------------------------------------------
        # APPROACH DETECTION
        # -----------------------------------------------------

        approaching = False

        if len(history) >= 3:
            old_distance = (
                self._distance_to_line(
                    history[-3]
                )
            )

            new_distance = distance

            if (
                old_distance is not None
                and new_distance is not None
            ):
                approaching = (
                    new_distance <
                    old_distance -
                    self.approach_threshold
                )

        approach_state = (
            self._update_approach_state(
                track_id,
                approaching,
                current_time
            )
        )

        approach_count = (
            approach_state[
                "approach_count"
            ]
        )

        approach_duration = (
            approach_state[
                "approach_duration"
            ]
        )

        # -----------------------------------------------------
        # LOITERING
        # -----------------------------------------------------

        loitering = False

        if dwell_time >= self.loitering_seconds:
            recent_points = history[-15:]

            if len(recent_points) >= 5:
                movement = 0.0

                for i in range(
                    1,
                    len(recent_points)
                ):
                    movement += self._distance(
                        recent_points[i - 1],
                        recent_points[i]
                    )

                if movement < 100:
                    loitering = True

        # -----------------------------------------------------
        # RISK SCORE
        # -----------------------------------------------------

        risk_score = 0

        # Zone contribution.
        if zone == "NORMAL":
            risk_score += 10

        elif zone == "RESTRICTED":
            risk_score += 30

        elif zone == "HIGH_SECURITY":
            risk_score += 50

        # Approaching perimeter.
        if approaching:
            risk_score += 15

        # Repeated approaches.
        if approach_count >= 2:
            risk_score += 10

        # Sustained approach.
        if approach_duration >= 5:
            risk_score += 5

        # Loitering.
        if loitering:
            risk_score += 20

        # Movement.
        if (
            direction != "UNKNOWN"
            and
            direction != "STATIONARY"
        ):
            risk_score += 5

        # Risk momentum.
        previous_history = (
            self.track_risk_history.get(
                track_id,
                []
            )
        )

        risk_momentum = (
            self._update_risk_momentum(
                track_id,
                risk_score
            )
        )

        if risk_momentum >= 5:
            risk_score += 5

        risk_score = min(
            100,
            risk_score
        )

        # -----------------------------------------------------
        # RISK LEVEL
        # -----------------------------------------------------

        risk_level = self._get_risk_level(
            risk_score
        )

        # -----------------------------------------------------
        # RISK TREND
        # -----------------------------------------------------

        risk_trend = self._get_risk_trend(
            track_id,
            risk_score
        )

        # -----------------------------------------------------
        # RISK ESCALATION
        # -----------------------------------------------------

        risk_escalating = (
            self._is_risk_escalating(
                track_id,
                risk_level
            )
        )

        # -----------------------------------------------------
        # RISK REASONS
        # -----------------------------------------------------

        risk_reasons = []

        if zone == "RESTRICTED":
            risk_reasons.append(
                "Entered restricted zone"
            )

        elif zone == "HIGH_SECURITY":
            risk_reasons.append(
                "Entered high-security zone"
            )

        if approaching:
            risk_reasons.append(
                "Approaching perimeter"
            )

        if approach_count >= 2:
            risk_reasons.append(
                f"Repeated approach detected ({approach_count}x)"
            )

        if approach_duration >= 5:
            risk_reasons.append(
                "Sustained approach behavior"
            )

        if loitering:
            risk_reasons.append(
                "Loitering detected"
            )

        if (
            direction != "UNKNOWN"
            and
            direction != "STATIONARY"
        ):
            risk_reasons.append(
                "Active movement detected"
            )

        if risk_trend == "INCREASING":
            risk_reasons.append(
                "Risk trend increasing"
            )

        if risk_escalating:
            risk_reasons.append(
                "Risk level escalating"
            )

        if not risk_reasons:
            risk_reasons.append(
                "No significant risk indicators"
            )

        # -----------------------------------------------------
        # EARLY WARNING
        # -----------------------------------------------------

        (
            early_warning,
            early_warning_reasons
        ) = self._calculate_early_warning(
            risk_score=risk_score,
            zone=zone,
            approaching=approaching,
            approach_count=approach_count,
            approach_duration=approach_duration,
            risk_trend=risk_trend,
            risk_increase_streak=risk_momentum,
            loitering=loitering
        )

        self.track_early_warning[
            track_id
        ] = early_warning

        # Add early-warning reasons to the main
        # risk explanation without duplicates.
        for reason in early_warning_reasons:
            if (
                reason not in
                risk_reasons
                and
                reason !=
                "No early-warning indicators"
            ):
                risk_reasons.append(
                    reason
                )

        # -----------------------------------------------------
        # UPDATE RISK HISTORY
        # -----------------------------------------------------

        risk_history = (
            self.track_risk_history[
                track_id
            ]
        )

        risk_history.append(
            risk_score
        )

        # Keep recent risk history only.
        if len(risk_history) > 20:
            risk_history.pop(0)

        self.track_previous_level[
            track_id
        ] = risk_level

        # -----------------------------------------------------
        # RETURN ANALYSIS
        # -----------------------------------------------------

        return {
            "track_id": track_id,

            "zone": zone,

            "direction": direction,

            "distance_to_perimeter":
                distance,

            "dwell_time":
                dwell_time,

            "approaching":
                approaching,

            "loitering":
                loitering,

            "risk_score":
                risk_score,

            "risk_level":
                risk_level,

            # Existing intelligence fields.
            "risk_reasons":
                risk_reasons,

            "risk_trend":
                risk_trend,

            "risk_escalating":
                risk_escalating,

            # NEW early-warning intelligence.
            "early_warning":
                early_warning,

            "early_warning_reasons":
                early_warning_reasons,

            "approach_count":
                approach_count,

            "approach_duration":
                approach_duration,

            "risk_increase_streak":
                risk_momentum
        }

    # ---------------------------------------------------------
    # REMOVE OLD TRACKS
    # ---------------------------------------------------------

    def cleanup(
        self,
        max_age=30
    ):
        current_time = time.time()

        expired = []

        for track_id, last_seen in (
            self.track_last_seen.items()
        ):
            if (
                current_time -
                last_seen
            ) > max_age:
                expired.append(
                    track_id
                )

        for track_id in expired:

            self.track_history.pop(
                track_id,
                None
            )

            self.track_first_seen.pop(
                track_id,
                None
            )

            self.track_last_seen.pop(
                track_id,
                None
            )

            self.track_risk_history.pop(
                track_id,
                None
            )

            self.track_previous_level.pop(
                track_id,
                None
            )

            # NEW state cleanup.
            self.track_approach_count.pop(
                track_id,
                None
            )

            self.track_currently_approaching.pop(
                track_id,
                None
            )

            self.track_approach_start.pop(
                track_id,
                None
            )

            self.track_last_approach.pop(
                track_id,
                None
            )

            self.track_risk_increase_streak.pop(
                track_id,
                None
            )

            self.track_early_warning.pop(
                track_id,
                None
            )