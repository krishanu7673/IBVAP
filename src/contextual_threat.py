import time


class ContextualThreatEngine:

    def __init__(
        self,
        correlation_window=60.0,
        cooldown_seconds=20.0
    ):
        self.correlation_window = float(
            correlation_window
        )

        self.cooldown_seconds = float(
            cooldown_seconds
        )

        # ----------------------------------------------------------
        # Event history
        #
        # Key:
        # (camera_id, track_id)
        # ----------------------------------------------------------

        self.event_history = {}

        # ----------------------------------------------------------
        # Last contextual alert time
        # ----------------------------------------------------------

        self.last_alert_time = {}

        # ----------------------------------------------------------
        # Highest contextual score observed
        # ----------------------------------------------------------

        self.peak_context_score = {}

    # ==============================================================
    # EVENT WEIGHTS
    # ==============================================================

    EVENT_WEIGHTS = {
        "ZONE_ENTRY": 20,
        "ZONE_TRANSITION": 15,
        "EARLY_WARNING": 30,
        "BEHAVIOR_RISK": 35,
        "LOITERING_DETECTED": 30,
        "PERIMETER_BREACH": 60
    }

    # ==============================================================
    # RECORD EVENT
    # ==============================================================

    def record_event(
        self,
        camera_id,
        track_id,
        event_type,
        risk_score=0,
        zone=None,
        direction=None,
        timestamp=None
    ):

        if track_id is None or track_id < 0:
            return None

        if timestamp is None:
            timestamp = time.time()

        key = (
            str(camera_id),
            int(track_id)
        )

        if key not in self.event_history:
            self.event_history[key] = []

        try:
            risk_score = float(
                risk_score or 0
            )
        except (
            TypeError,
            ValueError
        ):
            risk_score = 0.0

        event_record = {
            "event_type": event_type,
            "timestamp": timestamp,
            "risk_score": risk_score,
            "zone": zone,
            "direction": direction
        }

        self.event_history[key].append(
            event_record
        )

        self._cleanup_history(
            key,
            timestamp
        )

        return self._evaluate_context(
            key
        )

    # ==============================================================
    # CONTEXT EVALUATION
    # ==============================================================

    def _evaluate_context(
        self,
        key
    ):

        history = self.event_history.get(
            key,
            []
        )

        if not history:
            return None

        now = time.time()

        recent_events = [
            event
            for event in history
            if (
                now - event["timestamp"]
                <= self.correlation_window
            )
        ]

        if not recent_events:
            return None

        # ----------------------------------------------------------
        # Distinct event types
        # ----------------------------------------------------------

        event_types = {
            event["event_type"]
            for event in recent_events
        }

        # ----------------------------------------------------------
        # Calculate contextual score
        #
        # We use the strongest occurrence of each event type
        # instead of repeatedly adding the same event every frame.
        # ----------------------------------------------------------

        contextual_score = 0

        for event_type in event_types:

            weight = self.EVENT_WEIGHTS.get(
                event_type,
                0
            )

            contextual_score += weight

        # ----------------------------------------------------------
        # Risk contribution
        # ----------------------------------------------------------

        maximum_risk = 0.0

        for event in recent_events:

            maximum_risk = max(
                maximum_risk,
                float(
                    event.get(
                        "risk_score",
                        0
                    ) or 0
                )
            )

        # Add a limited contribution from the existing
        # behavioral risk engine.

        contextual_score += min(
            20,
            int(maximum_risk * 0.20)
        )

        # ----------------------------------------------------------
        # Sequence bonuses
        # ----------------------------------------------------------

        sequence_bonus = 0

        if (
            "EARLY_WARNING" in event_types
            and
            (
                "ZONE_ENTRY" in event_types
                or
                "ZONE_TRANSITION" in event_types
            )
        ):
            sequence_bonus += 10

        if (
            (
                "ZONE_ENTRY" in event_types
                or
                "ZONE_TRANSITION" in event_types
            )
            and
            "LOITERING_DETECTED" in event_types
        ):
            sequence_bonus += 10

        if (
            "EARLY_WARNING" in event_types
            and
            "BEHAVIOR_RISK" in event_types
        ):
            sequence_bonus += 10

        if (
            "LOITERING_DETECTED" in event_types
            and
            "BEHAVIOR_RISK" in event_types
        ):
            sequence_bonus += 10

        if (
            "PERIMETER_BREACH" in event_types
        ):
            sequence_bonus += 20

        contextual_score += sequence_bonus

        # ----------------------------------------------------------
        # Cap score
        # ----------------------------------------------------------

        contextual_score = min(
            100,
            contextual_score
        )

        # ----------------------------------------------------------
        # Update peak score
        # ----------------------------------------------------------

        previous_peak = self.peak_context_score.get(
            key,
            0
        )

        self.peak_context_score[key] = max(
            previous_peak,
            contextual_score
        )

        # ----------------------------------------------------------
        # Determine contextual level
        # ----------------------------------------------------------

        if contextual_score >= 75:
            contextual_level = "CRITICAL"

        elif contextual_score >= 50:
            contextual_level = "HIGH"

        elif contextual_score >= 25:
            contextual_level = "MEDIUM"

        else:
            contextual_level = "LOW"

        # ----------------------------------------------------------
        # Minimum evidence requirement
        #
        # A single low-level event should not create a
        # contextual threat.
        # ----------------------------------------------------------

        significant_event_types = {
            "ZONE_ENTRY",
            "ZONE_TRANSITION",
            "EARLY_WARNING",
            "BEHAVIOR_RISK",
            "LOITERING_DETECTED",
            "PERIMETER_BREACH"
        }

        significant_events = (
            event_types
            & significant_event_types
        )

        if len(significant_events) < 2:
            return None

        # ----------------------------------------------------------
        # Contextual alert threshold
        # ----------------------------------------------------------

        if contextual_score < 50:
            return None

        # ----------------------------------------------------------
        # Cooldown
        # ----------------------------------------------------------

        now = time.time()

        last_alert = self.last_alert_time.get(
            key
        )

        if (
            last_alert is not None
            and
            now - last_alert
            < self.cooldown_seconds
        ):
            return None

        # ----------------------------------------------------------
        # Build human-readable reasons
        # ----------------------------------------------------------

        reasons = []

        if "EARLY_WARNING" in event_types:
            reasons.append(
                "early-warning behavior detected"
            )

        if (
            "ZONE_ENTRY" in event_types
            or
            "ZONE_TRANSITION" in event_types
        ):
            reasons.append(
                "zone transition detected"
            )

        if "LOITERING_DETECTED" in event_types:
            reasons.append(
                "prolonged presence detected"
            )

        if "BEHAVIOR_RISK" in event_types:
            reasons.append(
                "elevated behavioral risk detected"
            )

        if "PERIMETER_BREACH" in event_types:
            reasons.append(
                "perimeter breach detected"
            )

        # ----------------------------------------------------------
        # Determine latest zone/direction
        # ----------------------------------------------------------

        latest_event = recent_events[-1]

        latest_zone = latest_event.get(
            "zone"
        )

        latest_direction = latest_event.get(
            "direction"
        )

        # ----------------------------------------------------------
        # Register alert
        # ----------------------------------------------------------

        self.last_alert_time[key] = now

        return {
            "camera_id": key[0],
            "track_id": key[1],

            "contextual_score":
                contextual_score,

            "contextual_level":
                contextual_level,

            "peak_context_score":
                self.peak_context_score[key],

            "event_count":
                len(recent_events),

            "event_types":
                sorted(
                    list(event_types)
                ),

            "reasons":
                reasons,

            "zone":
                latest_zone,

            "direction":
                latest_direction,

            "correlation_window":
                self.correlation_window,

            "timestamp":
                now
        }

    # ==============================================================
    # CLEANUP HISTORY
    # ==============================================================

    def _cleanup_history(
        self,
        key,
        current_time
    ):

        history = self.event_history.get(
            key,
            []
        )

        self.event_history[key] = [
            event
            for event in history
            if (
                current_time
                - event["timestamp"]
                <= self.correlation_window
            )
        ]

    # ==============================================================
    # TRACK CLEANUP
    # ==============================================================

    def cleanup(
        self,
        active_track_ids,
        camera_id,
        max_age=120
    ):

        active_track_ids = {
            int(track_id)
            for track_id in (
                active_track_ids or []
            )
            if track_id is not None
        }

        camera_id = str(
            camera_id
        )

        current_time = time.time()

        stale_keys = []

        for key, history in list(
            self.event_history.items()
        ):

            key_camera, key_track = key

            if key_camera != camera_id:
                continue

            if key_track in active_track_ids:
                continue

            if not history:
                stale_keys.append(
                    key
                )
                continue

            latest_timestamp = max(
                event["timestamp"]
                for event in history
            )

            if (
                current_time
                - latest_timestamp
                > max_age
            ):
                stale_keys.append(
                    key
                )

        for key in stale_keys:

            self.event_history.pop(
                key,
                None
            )

            self.last_alert_time.pop(
                key,
                None
            )

            self.peak_context_score.pop(
                key,
                None
            )

    # ==============================================================
    # STATISTICS
    # ==============================================================

    def get_statistics(self):

        return {
            "active_tracks":
                len(
                    self.event_history
                ),

            "tracks_with_context_alerts":
                len(
                    self.last_alert_time
                ),

            "peak_context_scores":
                dict(
                    self.peak_context_score
                )
        }