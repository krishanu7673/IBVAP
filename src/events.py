import time


class EventManager:

    def __init__(
        self,
        default_cooldown=5.0
    ):
        self.default_cooldown = float(
            default_cooldown
        )

        self.last_event = {}

        # ----------------------------------------------------
        # Priority-specific cooldowns
        # ----------------------------------------------------

        self.priority_cooldowns = {
            "CRITICAL": 3.0,
            "HIGH": 5.0,
            "MEDIUM": 10.0,
            "LOW": 20.0
        }

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self.total_events_seen = 0
        self.total_events_emitted = 0
        self.total_events_suppressed = 0

    # ========================================================
    # DETERMINE EVENT PRIORITY
    # ========================================================

    def get_priority(
        self,
        event_type,
        risk_score=None,
        risk_level=None
    ):

        # ----------------------------------------------------
        # Explicit perimeter breach
        # ----------------------------------------------------

        if event_type == "PERIMETER_BREACH":
            return "CRITICAL"

        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        if risk_level == "CRITICAL":
            return "CRITICAL"

        if risk_level == "HIGH":
            return "HIGH"

        if risk_level == "MEDIUM":
            return "MEDIUM"

        # ----------------------------------------------------
        # Risk score
        # ----------------------------------------------------

        if risk_score is not None:

            try:

                score = float(
                    risk_score
                )

                if score >= 75:
                    return "CRITICAL"

                if score >= 50:
                    return "HIGH"

                if score >= 25:
                    return "MEDIUM"

            except (
                TypeError,
                ValueError
            ):
                pass

        # ----------------------------------------------------
        # Event-specific priority
        # ----------------------------------------------------

        if event_type == "LOITERING_DETECTED":
            return "MEDIUM"

        if event_type == "ANPR_DETECTION":
            return "LOW"

        return "LOW"

    # ========================================================
    # GET COOLDOWN
    # ========================================================

    def get_cooldown(
        self,
        priority
    ):

        return self.priority_cooldowns.get(
            priority,
            self.default_cooldown
        )

    # ========================================================
    # SHOULD EMIT
    # ========================================================

    def should_emit(
        self,
        camera_id,
        event_type,
        track_id=None,
        cooldown=None,
        priority=None
    ):

        self.total_events_seen += 1

        # ----------------------------------------------------
        # Determine cooldown
        # ----------------------------------------------------

        if cooldown is None:

            if priority is not None:

                cooldown = self.get_cooldown(
                    priority
                )

            else:

                cooldown = (
                    self.default_cooldown
                )

        cooldown = float(
            cooldown
        )

        # ----------------------------------------------------
        # Deduplication key
        # ----------------------------------------------------

        key = (
            camera_id,
            event_type,
            track_id
        )

        now = time.time()

        previous = self.last_event.get(
            key
        )

        # ----------------------------------------------------
        # First occurrence
        # ----------------------------------------------------

        if previous is None:

            self.last_event[
                key
            ] = now

            self.total_events_emitted += 1

            return True

        # ----------------------------------------------------
        # Cooldown check
        # ----------------------------------------------------

        elapsed = (
            now - previous
        )

        if elapsed < cooldown:

            self.total_events_suppressed += 1

            return False

        # ----------------------------------------------------
        # Cooldown expired
        # ----------------------------------------------------

        self.last_event[
            key
        ] = now

        self.total_events_emitted += 1

        return True

    # ========================================================
    # CREATE EVENT
    # ========================================================

    def create_event(
        self,
        camera_id,
        event_type,
        details,
        track_id=None,
        confidence=None,
        risk_score=None,
        risk_level=None,
        zone=None,
        direction=None
    ):

        # ----------------------------------------------------
        # Determine priority
        # ----------------------------------------------------

        priority = self.get_priority(
            event_type=event_type,
            risk_score=risk_score,
            risk_level=risk_level
        )

        # ----------------------------------------------------
        # Check duplicate / cooldown
        # ----------------------------------------------------

        if not self.should_emit(
            camera_id=camera_id,
            event_type=event_type,
            track_id=track_id,
            priority=priority
        ):

            return None

        # ----------------------------------------------------
        # Create normalized event
        # ----------------------------------------------------

        event = {
            "camera_id": camera_id,
            "event_type": event_type,
            "track_id": track_id,
            "confidence": confidence,
            "details": details,
            "timestamp": time.time(),

            # Phase A additions
            "priority": priority,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "zone": zone,
            "direction": direction
        }

        return event

    # ========================================================
    # CLEANUP OLD EVENT KEYS
    # ========================================================

    def cleanup(
        self,
        max_age=300
    ):

        now = time.time()

        expired_keys = []

        for key, timestamp in (
            self.last_event.items()
        ):

            if (
                now - timestamp
                > max_age
            ):

                expired_keys.append(
                    key
                )

        for key in expired_keys:

            del self.last_event[
                key
            ]

    # ========================================================
    # STATISTICS
    # ========================================================

    def get_statistics(self):

        return {
            "total_events_seen":
                self.total_events_seen,

            "total_events_emitted":
                self.total_events_emitted,

            "total_events_suppressed":
                self.total_events_suppressed,

            "active_deduplication_keys":
                len(self.last_event)
        }